"""等待远端日志出现指定结束标记。

用法: python autodl/watch.py <远端日志> <结束标记> <超时秒> [轮询秒]
找到标记退出码 0；超时退出码 1。
"""
import os, sys, time, paramiko

HOST, PORT, USER = "connect.westd.seetacloud.com", 11570, "root"
PW = os.environ.get("AUTODL_SSH_PW", "")

log, marker = sys.argv[1], sys.argv[2]
timeout = float(sys.argv[3]) if len(sys.argv) > 3 else 4 * 3600
interval = float(sys.argv[4]) if len(sys.argv) > 4 else 60

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)

deadline = time.time() + timeout
last = ""
while time.time() < deadline:
    try:
        _, o, _ = c.exec_command(f"tail -c 3000 {log} | tr '\\r' '\\n' | tail -5", timeout=30)
        txt = o.read().decode("utf-8", "replace")
        if marker in txt:
            print("MARKER_FOUND")
            print(txt[-1500:])
            sys.exit(0)
        if txt.strip() != last:
            last = txt.strip()
            print("[progress]", txt.strip().splitlines()[-1][:160] if txt.strip() else "", flush=True)
    except Exception as e:
        print("[warn]", e, flush=True)
    time.sleep(interval)

print("TIMEOUT_WAITING")
sys.exit(1)
