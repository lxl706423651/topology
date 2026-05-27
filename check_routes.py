import subprocess
import re

def get_network_count():
    """获取 Docker 网络总数"""
    try:
        # 使用 -q 只列出 ID，方便计数
        result = subprocess.run(
            ["docker", "network", "ls", "-q"],
            capture_output=True, text=True, check=True
        )
        # 过滤空行并计数
        networks = [line for line in result.stdout.split('\n') if line.strip()]
        return len(networks)
    except subprocess.CalledProcessError as e:
        print(f"Error getting networks: {e}")
        return 0

def get_running_containers():
    """获取所有运行中的容器名称"""
    try:
        # format={{.Names}} 只获取容器名
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, check=True
        )
        containers = [line for line in result.stdout.split('\n') if line.strip()]
        return containers
    except subprocess.CalledProcessError as e:
        print(f"Error getting containers: {e}")
        return []

def get_bird_route_count(container_name):
    """在指定容器中执行 birdc show route count 并解析结果"""
    cmd = ["docker", "exec", container_name, "birdc", "show", "route", "count"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            # 容器可能没有安装 bird 或者命令失败
            return "N/A (Error)"
        
        output = result.stdout.strip()
        
        # birdc 输出通常格式: "12 of 12 routes for 10 networks"
        # 我们想提取 "10" 这个数字
        match = re.search(r'for\s+(\d+)\s+networks', output)
        if match:
            return int(match.group(1))
        
        # 如果格式只有 "12 routes" (旧版本或特殊情况)
        match_simple = re.search(r'^(\d+)\s+routes', output)
        if match_simple:
            return int(match_simple.group(1))
            
        return f"Unknown output: {output}"
        
    except Exception as e:
        return f"Exec failed: {e}"

def main():
    print("-" * 60)
    print("正在分析 Docker 网络仿真拓扑...")
    print("-" * 60)

    # 1. 获取网络总数
    total_networks = get_network_count()
    # 通常 Docker 有 3 个默认网络 (bridge, host, none)，仿真网络数通常是 总数 - 3
    sim_networks = max(0, total_networks - 3) 
    
    print(f"Docker Networks 总数: {total_networks}")
    print(f"推测的仿真链路/网段数 (Total - 3): {sim_networks}")
    print("-" * 60)
    print(f"{'Container Name':<30} | {'Reachable Networks':<20} | {'Status'}")
    print("-" * 60)

    # 2. 获取容器并循环检查
    containers = get_running_containers()
    
    # 对容器名排序，方便查看
    containers.sort()

    for container in containers:
        route_count = get_bird_route_count(container)
        
        # 简单的状态判断
        status = "OK"
        if isinstance(route_count, int):
            # 如果学到的路由数接近网络总数，标记为 Full，否则 Partial
            if route_count >= sim_networks and sim_networks > 0:
                status = "✅ Full Mesh"
            elif route_count == 0:
                status = "❌ No Routes"
            else:
                status = "⚠️ Partial"
        else:
            status = "❓ Unknown"

        print(f"{container:<30} | {str(route_count):<20} | {status}")

if __name__ == "__main__":
    main()