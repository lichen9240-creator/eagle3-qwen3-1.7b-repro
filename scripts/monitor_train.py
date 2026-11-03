"""训练实时监控：采样训练日志 + GPU，检测异常。

每 interval 秒连一次实例，解析新出现的所有 (step, loss, acc)，维护历史：
- 硬异常：loss nan/inf、rolling(100) 均值 > 2.5x 前 500 均值、loss>30、
          进程消失、Traceback/OOM/CUDA error、步数停滞 >8 分钟
- 软异常：GPU 利用率 <30%、温度 >88C、步速 <40% 中位数（连续 3 次才退出）
- 训练正常结束（G8_DONE）：退出码 0

退出码：0 正常结束 / 2 硬异常 / 3 连续软告警 / 1 超时
"""
import json, os, re, statistics, sys, time
from collections import deque
from datetime import datetime
import paramiko

HOST, PORT, USER = "connect.westd.seetacloud.com", 11570, "root"
PW = os.environ.get("AUTODL_SSH_PW", "")
BASE = os.path.dirname(os.path.abspath(__file__))
LOG = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/eagle3-qwen/logs/train_30k.log"
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 120.0
MAX_SECONDS = float(sys.argv[3]) if len(sys.argv) > 3 else 8 * 3600
OUT = os.path.join(BASE, "monitor_status.json")
CSV = os.path.join(BASE, "train_monitor.csv")

LINE_RE = re.compile(
    r"Training Epoch (\d+):.*?\|\s*(\d+)/(\d+) .*?([\d.]+)(it/s|s/it)"
    r".*?loss=([-\d.eEnanNANINF]+),\s*acc=([\d.]+)"
)


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30,
              banner_timeout=30, auth_timeout=30)
    return c


def run(c, cmd, timeout=60):
    _, o, _ = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode("utf-8", "replace")
    o.channel.recv_exit_status()
    return out


def sample(c):
    tail = run(c, f"tail -c 8000 {LOG} | tr '\\r' '\\n'")
    alive = run(c, "pgrep -c -f train_eagle3.py || true").strip()
    g = run(c, "nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu "
               "--format=csv,noheader,nounits").strip()
    try:
        u, m, t = [float(x) for x in g.split(",")]
    except Exception:
        u = m = t = None
    return tail, alive, u, m, t


def main():
    start = time.time()
    hist = deque(maxlen=4000)      # (step, loss, acc)
    seen = set()
    last_step, last_progress = None, time.time()
    warn_streak = 0
    c = connect()
    csv_new = not os.path.exists(CSV)
    cf = open(CSV, "a", encoding="utf-8")
    if csv_new:
        cf.write("ts,epoch,step,loss,acc,it_s,gpu_util,gpu_mem\n")

    while time.time() - start < MAX_SECONDS:
        ts = datetime.now().strftime("%H:%M:%S")
        try:
            tail, alive, util, mem, temp = sample(c)
        except Exception as ex:
            print(f"[{ts}] reconnect: {ex}", flush=True)
            try:
                c.close()
            except Exception:
                pass
            time.sleep(5)
            c = connect()
            continue

        if "G8_DONE" in tail:
            print(f"[{ts}] TRAINING_DONE", flush=True)
            json.dump({"ts": ts, "state": "done", "samples": len(hist)}, open(OUT, "w"))
            cf.close()
            return 0

        hard = None
        if "Traceback" in tail or "out of memory" in tail.lower() or "CUDA error" in tail:
            hard = "日志出现 Traceback/OOM/CUDA error"
        if alive in ("0", ""):
            hard = f"训练进程不存在（pgrep={alive}）"

        new = []
        for line in tail.splitlines():
            m = LINE_RE.search(line)
            if not m:
                continue
            ep, step = int(m.group(1)), int(m.group(2))
            rate = float(m.group(4))
            unit = m.group(5)
            it_s = (1.0 / rate) if unit == "it/s" and rate else rate
            loss_s = m.group(6).lower()
            loss = float("nan") if loss_s in ("nan", "inf", "-inf") else float(m.group(6))
            acc = float(m.group(7))
            if step not in seen:
                seen.add(step)
                new.append((ep, step, loss, acc, it_s))

        new.sort(key=lambda x: x[1])
        for ep, step, loss, acc, it_s in new:
            hist.append((step, loss, acc))
            cf.write(f"{ts},{ep},{step},{loss},{acc},{it_s},{util},{mem}\n")
            if loss != loss:
                hard = f"step {step} loss=NaN"
            elif loss > 30:
                hard = f"step {step} loss={loss:.1f} 异常大"
        cf.flush()

        cur = None
        if new:
            ep, step, loss, acc, it_s = new[-1]
            cur = {"epoch": ep, "step": step, "loss": loss, "acc": acc, "it_s": it_s}
            if step != last_step:
                last_step, last_progress = step, time.time()

        losses = [x[1] for x in hist if x[1] == x[1]]
        if len(losses) >= 400 and hard is None:
            recent = statistics.mean(losses[-100:])
            base = statistics.mean(losses[-600:-100]) if len(losses) >= 600 else statistics.mean(losses[:-100])
            if base > 0 and recent > 2.5 * base and recent > 3.0:
                hard = f"loss 发散：近 100 均值 {recent:.2f} vs 前 500 均值 {base:.2f}"

        if hard is None and time.time() - last_progress > 480 and "Evaluating" not in tail and "Saving" not in tail:
            hard = f"步数停滞 {int(time.time() - last_progress)}s（step={last_step}）"

        soft = []
        if util is not None and util < 20 and (last_step or 0) > 500:
            soft.append(f"GPU 利用率 {util:.0f}%")
        if temp is not None and temp > 88:
            soft.append(f"GPU 温度 {temp:.0f}C")

        warn_streak = warn_streak + 1 if soft else 0

        status = {"ts": ts, "state": "running", "alive": alive, "gpu_util": util,
                  "gpu_mem": mem, "gpu_temp": temp, "cur": cur,
                  "loss_mean_100": round(statistics.mean(losses[-100:]), 3) if losses else None,
                  "loss_base_500": round(statistics.mean(losses[-600:-100]), 3) if len(losses) >= 600 else None,
                  "samples": len(hist), "warn": soft, "warn_streak": warn_streak,
                  "elapsed_min": round((time.time() - start) / 60, 1)}
        json.dump(status, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

        if cur:
            print(f"[{ts}] step {cur['step']} loss={cur['loss']:.2f} acc={cur['acc']:.3f} "
                  f"{cur['it_s']:.2f}s/step gpu={util}% mem={mem}MB "
                  f"loss100={status['loss_mean_100']}" + (f" WARN={soft}" if soft else ""), flush=True)
        else:
            print(f"[{ts}] no new step line; gpu={util}% mem={mem}MB", flush=True)

        if hard:
            print(f"ANOMALY: {hard}", flush=True)
            json.dump({"ts": ts, "state": "anomaly", "reason": hard, "cur": cur},
                      open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            cf.close()
            return 2
        if warn_streak >= 3:
            print(f"ANOMALY_WARN: 连续软告警 {soft}", flush=True)
            cf.close()
            return 3

        time.sleep(INTERVAL)

    print("MONITOR_TIMEOUT", flush=True)
    cf.close()
    return 1


if __name__ == "__main__":
    sys.exit(main())
