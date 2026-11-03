import os, paramiko
c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("connect.westd.seetacloud.com", port=11570, username="root",
          password=os.environ["AUTODL_SSH_PW"], timeout=25)
for title, cmd in [
    ("shell", "cat /etc/shells >/dev/null; echo SHELL=$SHELL; ls /root/miniconda3/bin/python* 2>&1 | head"),
    ("torch", "/root/miniconda3/bin/python -c \"import sys,torch;print(sys.version.split()[0],'torch',torch.__version__,'cuda',torch.version.cuda,'gpu',torch.cuda.get_device_name(0),torch.cuda.get_device_properties(0).total_memory//2**30,'GB')\" 2>&1 | tail -2"),
    ("cgroup", "cat /sys/fs/cgroup/cpu.max 2>/dev/null; cat /sys/fs/cgroup/memory.max 2>/dev/null"),
    ("image", "cat /etc/autodl_version 2>/dev/null || cat /root/.autodl_version 2>/dev/null || echo none"),
]:
    print(f"===== {title} =====")
    _, o, e = c.exec_command(cmd, timeout=60)
    print((o.read().decode("utf-8","replace").strip() or "(空)"))
    err = e.read().decode("utf-8","replace").strip()
    if err: print("[stderr]", err[:300])
c.close()
