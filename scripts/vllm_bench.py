"""vLLM 侧 benchmark：走 OpenAI /v1/completions，input_ids 直传，另抓 /metrics 算 acceptance。"""
import argparse, asyncio, json, statistics, time, urllib.request
from pathlib import Path
import aiohttp


def metrics(base):
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            txt = r.read().decode("utf-8", "replace")
    except Exception:
        return {}
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for key in ["spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total",
                    "spec_decode_num_emitted_tokens_total", "num_requests_total"]:
            if line.startswith("vllm:" + key):
                try:
                    out[key] = float(line.rsplit(" ", 1)[1])
                except Exception:
                    pass
    return out


async def one(session, url, item, max_tokens, sem, model):
    async with sem:
        payload = {
            "model": model,
            "prompt": item["input_ids"],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
            "stream": False,
        }
        t0 = time.perf_counter()
        try:
            async with session.post(url, json=payload) as r:
                d = await r.json()
            dt = time.perf_counter() - t0
            if "usage" not in d:
                return None
            return {"ok": True, "out": d["usage"]["completion_tokens"], "dt": dt,
                    "in": d["usage"]["prompt_tokens"]}
        except Exception as e:
            return {"ok": False, "err": repr(e)[:120]}


async def run(base, data, max_tokens, conc, nreq, model):
    url = base + "/v1/completions"
    items = [json.loads(l) for l in Path(data).read_text(encoding="utf-8").split("\n") if l.strip()]
    items = items[:nreq]
    sem = asyncio.Semaphore(conc)
    async with aiohttp.ClientSession() as s:
        # warmup
        for it in items[:2]:
            await one(s, url, it, 16, sem, model)
        m0 = metrics(base)
        t0 = time.perf_counter()
        res = await asyncio.gather(*[one(s, url, it, max_tokens, sem, model) for it in items])
        wall = time.perf_counter() - t0
        m1 = metrics(base)
    ok = [r for r in res if r and r.get("ok")]
    out_tok = sum(r["out"] for r in ok)
    tput = out_tok / wall
    lat = [r["dt"] for r in ok]
    acc = None
    if m0 and m1 and m1.get("spec_decode_num_draft_tokens_total", 0) > m0.get("spec_decode_num_draft_tokens_total", 0):
        da = m1["spec_decode_num_accepted_tokens_total"] - m0["spec_decode_num_accepted_tokens_total"]
        dd = m1["spec_decode_num_draft_tokens_total"] - m0["spec_decode_num_draft_tokens_total"]
        acc = 1 + da / dd
    return {"conc": conc, "max_tokens": max_tokens, "n": len(ok),
            "throughput_tokens_s": tput, "latency_mean_ms": statistics.mean(lat) * 1000 if lat else None,
            "accept_len": acc}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="qwen3")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--requests", type=int, default=60)
    ap.add_argument("--conc", type=int, nargs="+", default=[1])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    for c in a.conc:
        r = asyncio.run(run(a.base, a.data, a.max_tokens, c, a.requests, a.model))
        print(json.dumps(r, ensure_ascii=False))
        (out / f"{a.tag}_c{c}_summary.json").write_text(json.dumps(r, ensure_ascii=False, indent=1))
