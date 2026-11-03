"""在 AutoDL 实例上执行远端 bash 脚本。

用法：
  python autodl/rc.py --file autodl/steps/00_recon.sh
  python autodl/rc.py --file autodl/steps/x.sh --bg /root/autodl-tmp/eagle3-qwen/logs/x.log
  python autodl/rc.py --tail /root/autodl-tmp/eagle3-qwen/logs/x.log -n 60
  python autodl/rc.py --cmd "nvidia-smi -L"

  --bg  : nohup 后台执行，立刻返回（长任务用）
  --tail: 读取远端日志尾部
  --wait: 与 --bg 配合时，等到远端 PID 结束再返回
"""
import argparse, base64, os, sys, time, uuid, paramiko

HOST, PORT, USER = "connect.westd.seetacloud.com", 11570, "root"
PW = os.environ.get("AUTODL_SSH_PW", "")
ROOT = "/root/autodl-tmp/eagle3-qwen"


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PW, timeout=30,
              banner_timeout=30, auth_timeout=30)
    return c


def run(c, cmd, timeout=3600):
    _, o, e = c.exec_command(cmd, timeout=timeout, get_pty=False)
    out = o.read().decode("utf-8", "replace")
    err = e.read().decode("utf-8", "replace")
    rc = o.channel.recv_exit_status()
    return rc, out, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--cmd")
    ap.add_argument("--bg", default=None, help="远端日志路径；将脚本放后台执行")
    ap.add_argument("--tail", default=None)
    ap.add_argument("-n", type=int, default=40)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    c = connect()
    try:
        if args.tail:
            rc, out, err = run(c, f"tail -n {args.n} {args.tail} 2>&1", 120)
            sys.stdout.write(out)
            if err.strip():
                sys.stdout.write("[stderr] " + err)
            return rc

        if args.file:
            src = open(args.file, "rb").read()
        elif args.cmd:
            src = args.cmd.encode("utf-8")
        else:
            src = sys.stdin.buffer.read()

        b64 = base64.b64encode(src).decode()
        script = f"/tmp/_dsh_step_{uuid.uuid4().hex[:10]}.sh"
        write_cmd = (
            f"mkdir -p {ROOT}/logs && echo {b64} | base64 -d > {script} && "
            f"sed -i 's/\\r$//' {script} && wc -c {script}"
        )

        if args.bg:
            rc, out, err = run(c, write_cmd, 120)
            if rc != 0:
                print("写入脚本失败:", out, err)
                return rc
            pidfile = args.bg + ".pid"
            cmd = (f"nohup bash -c 'echo $$ > {pidfile}; bash {script}' "
                   f"> {args.bg} 2>&1 & echo started_pid=$!")
            print(run(c, cmd, 120)[1])
            if args.wait:
                deadline = time.time() + args.timeout
                while time.time() < deadline:
                    rc, out, _ = run(c, f"kill -0 $(cat {pidfile}) 2>/dev/null && echo ALIVE || echo DONE", 60)
                    if "DONE" in out:
                        print("[remote] finished")
                        break
                    time.sleep(30)
        else:
            rc, out, err = run(c, f"{write_cmd} >/dev/null && bash {script}", args.timeout)
            sys.stdout.write(out)
            if err.strip():
                sys.stdout.write("\n[stderr]\n" + err)
            print(f"\n[exit {rc}]")
            return rc
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
