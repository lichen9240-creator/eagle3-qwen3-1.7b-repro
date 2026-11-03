"""把本地 repro/ 目录上传到实例的 $ROOT/repro。"""
import os, sys, stat, paramiko

HOST, PORT, USER = "connect.westd.seetacloud.com", 11570, "root"
PW = os.environ.get("AUTODL_SSH_PW", "")
ROOT = "/root/autodl-tmp/eagle3-qwen"
LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "Qwen3-1.7B_EAGLE3_repro_bundle", "repro")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30)
sftp = c.open_sftp()


def mkdirs(remote):
    parts = remote.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


n = 0
for dirpath, dirnames, filenames in os.walk(LOCAL):
    dirnames[:] = [d for d in dirnames if d != "__pycache__"]
    rel = os.path.relpath(dirpath, LOCAL).replace("\\", "/")
    rdir = ROOT + "/repro" + ("" if rel == "." else "/" + rel)
    mkdirs(rdir)
    for f in filenames:
        if f.endswith(".pyc"):
            continue
        lp = os.path.join(dirpath, f)
        rp = rdir + "/" + f
        sftp.put(lp, rp)
        n += 1
        print(f"  {rp}  ({os.path.getsize(lp)} B)")

print(f"uploaded {n} files")
c.close()
