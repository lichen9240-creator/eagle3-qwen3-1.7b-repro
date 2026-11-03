"""在对端实例上执行命令（主机/端口/密码通过参数传入）。"""
import argparse, base64, os, sys, paramiko

ap = argparse.ArgumentParser()
ap.add_argument("--host", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--pw", required=True)
ap.add_argument("--file")
ap.add_argument("--cmd")
ap.add_argument("--tail")
ap.add_argument("-n", type=int, default=40)
a = ap.parse_args()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(a.host, port=a.port, username="root", password=a.pw, timeout=30,
          banner_timeout=30, auth_timeout=30)


def run(cmd, timeout=600):
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    rc = o.channel.recv_exit_status()
    return rc, out, err


if a.tail:
    rc, out, err = run(f"tail -n {a.n} {a.tail}", 120)
    sys.stdout.write(out)
else:
    if a.file:
        src = open(a.file, "rb").read()
    elif a.cmd:
        src = a.cmd.encode("utf-8")
    else:
        src = sys.stdin.buffer.read()
    script = "/tmp/_peer_step.sh"
    b64 = base64.b64encode(src).decode()
    rc, out, err = run(f"echo {b64} | base64 -d > {script} && bash {script}", 1800)
    sys.stdout.write(out)
    if err.strip():
        sys.stdout.write("\n[stderr]\n" + err)
    print(f"\n[exit {rc}]")
c.close()
