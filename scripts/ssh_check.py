import os, sys, paramiko

HOST, PORT, USER = "connect.westd.seetacloud.com", 11570, "root"
PW = os.environ.get("AUTODL_SSH_PW", "")

cmds = [
    ("GPU 卡型/显存/算力", "nvidia-smi --query-gpu=name,memory.total,memory.used,compute_cap,driver_version --format=csv"),
    ("GPU 拓扑(是否 MIG/切片)", "nvidia-smi -L && nvidia-smi --query-gpu=mig.mode.current --format=csv"),
    ("GPU 进程/利用率", "nvidia-smi --query-gpu=utilization.gpu,power.draw,power.limit --format=csv"),
    ("磁盘", "df -h / /root/autodl-tmp 2>&1 | tail -n +1"),
    ("CPU/内存", "nproc && free -g | head -2"),
    ("Python/torch", "python -V; python -c \"import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),torch.cuda.get_device_name(0))\" 2>&1 | tail -3"),
    ("镜像标识", "cat /etc/autodl_version 2>/dev/null; ls -d /root/autodl-tmp 2>&1"),
]

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=25, banner_timeout=25, auth_timeout=25)

for title, cmd in cmds:
    print(f"\n===== {title} =====")
    stdin, stdout, stderr = c.exec_command(cmd, timeout=40)
    out = stdout.read().decode("utf-8", "replace").strip()
    err = stderr.read().decode("utf-8", "replace").strip()
    print(out or "(空)")
    if err:
        print("[stderr]", err[:400])
c.close()
