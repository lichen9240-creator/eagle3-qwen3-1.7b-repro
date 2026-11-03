import json, time, urllib.request, urllib.error, pathlib

token = pathlib.Path(".secrets/autodl_token.txt").read_text(encoding="utf-8").strip()
BASE = "https://api.autodl.com"
UUID = "pro-79360d2314fc"

def call(method, path):
    req = urllib.request.Request(BASE + path,
                                 headers={"Authorization": token, "Content-Type": "application/json"},
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return json.loads(e.read().decode())
    except Exception as e:
        return {"code": "ERR", "msg": str(e)}

def gb(n):
    return f"{n/1024**3:.1f}GB" if isinstance(n, int) else n

for i in range(40):
    st = call("GET", f"/api/v1/dev/instance/pro/status?instance_uuid={UUID}").get("data")
    print(f"[{i*3:3d}s] status={st}")
    if st == "running":
        break
    time.sleep(3)

s = call("GET", f"/api/v1/dev/instance/pro/snapshot?instance_uuid={UUID}").get("data") or {}
print("\n=== 实例详情 ===")
for k in ["name", "region_sign", "snapshot_gpu_alias_name", "chip_corp",
          "payg_price", "origin_pay_price"]:
    print(f"  {k:<24} {s.get(k)}")
print(f"  {'系统盘总容量':<22} {gb(s.get('system_init_disk_size'))} + 扩容 {gb(s.get('expand_system_disk_size'))}")
print(f"  {'数据盘':<24} {gb(s.get('data_disk_total_size'))}")
ui = s.get("usage_info") or {}
print(f"  {'CPU/内存':<23} {ui.get('cpu_usage_percent')}% / {ui.get('mem_usage_percent')}%  (上限 {gb(ui.get('mem_limit'))})")
print("\n=== 连接信息 ===")
print(f"  ssh        {s.get('ssh_command')}")
print(f"  密码       {s.get('root_password')}")
print(f"  JupyterLab https://{s.get('jupyter_domain')}/lab?token={s.get('jupyter_token')}")
