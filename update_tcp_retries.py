import json
import subprocess
import os

# 1. 您提供的基础函数
def run_nsenter_cmd(pid, shell_cmd, async_mode=False):
    """使用 nsenter 进入容器执行任意 Shell 命令"""
    # 注意：确保 pid 是字符串，防止 f-string 报错
    pid = str(pid)
    
    # -n 进入网络命名空间 (必须，因为我们要改网络参数)
    # -m 进入挂载命名空间 (必须，因为 sysctl 需要访问 /proc/sys)
    full_cmd = f"nsenter -t {pid} -n -m {shell_cmd}"
    
    if async_mode:
        subprocess.Popen(full_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None
    else:
        # 使用 capture_output=True 来捕获 sysctl 的输出结果
        return subprocess.run(full_cmd, shell=True, capture_output=True, text=True)

# 2. 批量处理主逻辑
def apply_sysctl_to_all(json_file_path):
    # 检查文件是否存在
    if not os.path.exists(json_file_path):
        print(f"❌ 错误: 找不到文件 {json_file_path}")
        return

    print(f"📂 正在读取 {json_file_path} ...")
    
    try:
        with open(json_file_path, 'r') as f:
            container_map = json.load(f)
    except json.JSONDecodeError:
        print("❌ 错误: JSON 文件格式不正确")
        return

    # 设定的目标参数
    target_cmd = "sysctl -w net.ipv4.tcp_retries2=100"
    
    success_count = 0
    fail_count = 0

    print("-" * 60)
    print(f"{'Container Name':<30} | {'PID':<10} | {'Result':<10}")
    print("-" * 60)

    # 遍历字典 (假设 json 格式为 {"container_name": pid, ...})
    # 如果您的 json 是 list 格式 [{"name": "xxx", "pid": 123}], 请告诉我调整代码
    for name, pid in container_map.items():
        # 执行命令 (同步模式，以便确认结果)
        result = run_nsenter_cmd(pid, target_cmd, async_mode=False)
        
        if result.returncode == 0:
            print(f"{name:<30} | {pid:<10} | ✅ Success")
            # 可选：打印详细输出，例如 "net.ipv4.tcp_retries2 = 100"
            # print(f"   └─ Output: {result.stdout.strip()}")
            success_count += 1
        else:
            print(f"{name:<30} | {pid:<10} | ❌ Failed")
            print(f"   └─ Error: {result.stderr.strip()}")
            fail_count += 1

    print("-" * 60)
    print(f"🎉 完成。成功: {success_count} 个, 失败: {fail_count} 个。")

if __name__ == "__main__":
    # 请确保文件名和路径正确
    JSON_FILE = "container_pids.json"
    apply_sysctl_to_all(JSON_FILE)