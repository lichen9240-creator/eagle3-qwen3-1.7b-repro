#!/usr/bin/env python3
"""AutoDL 多规格抢卡（Pro API）。

同一进程内按权重轮询多个算力规格，串行发送创建请求：
  - 严格串行 => 同一时刻只有一个 create 在飞 => 不可能同时开出两台。
  - 任一规格抢到后立即停止，其余规格不再尝试。
  - 无库存继续轮询；余额不足/鉴权失败/参数错误立即停止。
  - 网络错误或响应丢失时回查实例列表，防止重复创建。

用法：
  python autodl/grab.py --specs 4090D:2,v-48g:1 --interval 0.3 --duration 14400
"""
import argparse, json, random, socket, sys, time, urllib.error, urllib.request
from datetime import datetime
from pathlib import Path

BASE = "https://api.autodl.com"
ROOT = Path(__file__).resolve().parent
TOKEN_FILE = ROOT.parent / ".secrets" / "autodl_token.txt"
LOG_FILE = ROOT / "grab.log"
STATE_FILE = ROOT / "grab_state.json"
STOP_FILE = ROOT / "STOP"

FATAL_HINTS = (
    "余额", "欠费", "充值", "balance", "insufficient",
    "认证", "实名", "token", "鉴权", "未登录", "权限",
    "算力配置规格不存在",
)


def now():
    return datetime.now().strftime("%H:%M:%S")


def log(msg):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


class Api:
    def __init__(self, token, timeout=20.0):
        self.token = token
        self.timeout = timeout

    def call(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            BASE + path, data=data,
            headers={"Authorization": self.token, "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw, status = r.read().decode("utf-8", "replace"), r.status
        except urllib.error.HTTPError as e:
            raw, status = e.read().decode("utf-8", "replace"), e.code
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            return None, {"code": "NetworkError", "msg": str(e), "data": None}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"code": f"HTTP{status}", "msg": raw[:300], "data": None}
        return status, payload

    def list_instances(self):
        st, p = self.call("POST", "/api/v1/dev/instance/pro/list", {"page_index": 1, "page_size": 50})
        if p.get("code") != "Success":
            return None
        return (p.get("data") or {}).get("list") or []

    def snapshot(self, uuid):
        st, p = self.call("GET", f"/api/v1/dev/instance/pro/snapshot?instance_uuid={uuid}")
        return p.get("data") if p.get("code") == "Success" else None


def parse_specs(text):
    """'4090D:2,v-48g:1' -> ['4090D','4090D','v-48g']"""
    schedule = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            spec, w = part.rsplit(":", 1)
            weight = max(1, int(w))
        else:
            spec, weight = part, 1
        schedule.extend([spec.strip()] * weight)
    return schedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="4090D:2,v-48g:1")
    ap.add_argument("--image", default="base-image-l2t43iu6uk")
    ap.add_argument("--cuda", default=118, type=int)
    ap.add_argument("--gpus", default=1, type=int)
    ap.add_argument("--disk", default=0, type=int)
    ap.add_argument("--dc", default="")
    ap.add_argument("--name", default="grab-4090b")
    ap.add_argument("--interval", default=0.3, type=float)
    ap.add_argument("--duration", default=14400, type=float)
    ap.add_argument("--max-backoff", default=30.0, type=float)
    ap.add_argument("--allow-duplicate", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    schedule = parse_specs(args.specs)
    if not schedule:
        log("FATAL 规格列表为空")
        return 2

    api = Api(TOKEN_FILE.read_text(encoding="utf-8").strip())
    dc = [s.strip() for s in args.dc.split(",") if s.strip()]

    def body_for(spec):
        b = {
            "req_gpu_amount": args.gpus,
            "expand_system_disk_by_gb": args.disk,
            "gpu_spec_uuid": spec,
            "image_uuid": args.image,
            "cuda_v_from": args.cuda,
            "instance_name": args.name,
        }
        if dc:
            b["data_center_list"] = dc
        return b

    log("=" * 70)
    log(f"轮询规格={args.specs} -> 调度序列={schedule}")
    log(f"镜像={args.image} cuda>={args.cuda} GPU={args.gpus} 地域={dc or '自动'} 间隔={args.interval}s 上限={args.duration}s")
    if args.dry_run:
        for s in dict.fromkeys(schedule):
            log(f"  body[{s}]={json.dumps(body_for(s), ensure_ascii=False)}")
        return 0

    baseline = api.list_instances()
    if baseline is None:
        log("FATAL 读取实例列表失败（token 或网络），退出")
        return 2
    baseline_uuids = {i.get("uuid") for i in baseline}
    log(f"基线实例 {len(baseline)} 台")
    for i in baseline:
        log(f"  已有 {i.get('uuid')} | {i.get('name')} | {i.get('gpu_spec_uuid')} | {i.get('region_sign')} | {i.get('status')}")

    dup = [i for i in baseline if i.get("name") == args.name and i.get("status") in ("running", "starting", "creating")]
    if dup and not args.allow_duplicate:
        log(f"ABORT 已存在同名运行中实例 {[i.get('uuid') for i in dup]}（要强制再开加 --allow-duplicate）")
        return 3

    deadline = time.time() + args.duration
    started = time.time()
    attempt = 0
    per_spec = {s: 0 for s in schedule}
    backoff = 0.0
    disabled = set()

    while time.time() < deadline:
        if STOP_FILE.exists():
            log("检测到 STOP 文件，退出")
            return 4
        keys = [s for s in schedule if s not in disabled]
        if not keys:
            log("FATAL 所有规格都被禁用，退出")
            return 5

        spec = keys[attempt % len(keys)]
        attempt += 1
        per_spec[spec] = per_spec.get(spec, 0) + 1

        status, p = api.call("POST", "/api/v1/dev/instance/pro/create", body_for(spec))
        code, msg = p.get("code"), (p.get("msg") or "").strip()

        if code == "Success":
            uuid = p.get("data")
            log(f"SUCCESS 第 {attempt} 次尝试，规格 {spec} 抢到实例 {uuid}")
            STATE_FILE.write_text(json.dumps({
                "ok": True, "instance_uuid": uuid, "spec": spec, "attempts": attempt,
                "request_id": p.get("request_id"), "body": body_for(spec),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return 0

        low = (msg + " " + str(code)).lower()
        if any(h in low for h in FATAL_HINTS):
            log(f"FATAL 规格 {spec} 返回不可重试错误 code={code} msg={msg}，退出")
            STATE_FILE.write_text(json.dumps({
                "ok": False, "attempts": attempt, "code": code, "msg": msg,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return 6

        if "规格不存在" in msg:
            disabled.add(spec)
            log(f"  规格 {spec} 不存在，从队列移除")

        # 只有真·无响应才回查列表，避免每轮多一次请求拖慢节奏
        if status is None or (status and status >= 500 and not p.get("request_id")):
            cur = api.list_instances()
            if cur is not None:
                new_ones = [i for i in cur if i.get("uuid") not in baseline_uuids]
                if new_ones:
                    i = new_ones[0]
                    log(f"SUCCESS 发现新实例（上次响应丢失）{i.get('uuid')} | {i.get('gpu_spec_uuid')}")
                    STATE_FILE.write_text(json.dumps({
                        "ok": True, "instances": new_ones, "attempts": attempt,
                        "finished_at": datetime.now().isoformat(timespec="seconds"),
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                    return 0

        if attempt <= 3 or attempt % 5 == 0 or msg != last_msg:
            log(f"  #{attempt} {spec} {code} {msg[:48]}")
        last_msg = msg

        if status == 429 or (status and status >= 500) or code == "NetworkError":
            backoff = min(args.max_backoff, max(2.0, backoff * 2 or 2.0))
            sleep_for = backoff
            log(f"  ! 退避 {backoff:.0f}s（status={status} code={code}）")
        else:
            backoff = 0.0
            sleep_for = args.interval

        if attempt % 60 == 0:
            el = time.time() - started
            log(f"  ...{attempt} 次 / {el/60:.1f} 分钟 ({attempt/max(el,1e-9)*60:.0f} 次/分)  分布={per_spec}  最近: {spec} {msg or code}")
        time.sleep(sleep_for + random.uniform(0, 0.15))

    log(f"TIMEOUT 达上限 {args.duration}s，{attempt} 次未抢到，分布={per_spec}")
    return 7


if __name__ == "__main__":
    sys.exit(main())
