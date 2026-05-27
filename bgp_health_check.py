import subprocess
import re
import datetime
import sys

# ================= 配置区域 =================
# 容器名称匹配正则: 匹配 as1269brd-r32... 这种格式
# ^as       : 以 as 开头
# \d+       : 接着是数字 (AS号)
# brd-r     : 接着是固定字符串 brd-r
# .* : 后面可以是任意字符
CONTAINER_NAME_REGEX = r"^as\d+brd-r.*"

# BIRD 命令
CMD_CHECK_PROTOCOLS = "birdc show protocols"
# ===========================================

class Logger:
    def __init__(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"logs/bgp_health_check/bgp_issues_{timestamp}.log"
        self.file = open(self.filename, "w", encoding='utf-8')
        
        header = f"=== BGP Health Check Report ===\nTime: {timestamp}\nTarget Pattern: {CONTAINER_NAME_REGEX}\n===============================\n"
        print(header, end='')
        self.file.write(header)

    def log_file(self, message):
        """只记录到文件"""
        self.file.write(message + "\n")
        self.file.flush()

    def log_both(self, message):
        """同时记录到屏幕和文件"""
        print(message)
        self.file.write(message + "\n")
        self.file.flush()
        
    def close(self):
        self.file.write("\n=== End of Report ===\n")
        self.file.close()
        print(f"\n[Done] Report saved to: {self.filename}")

def get_target_containers():
    """获取符合命名规则的容器"""
    try:
        # 获取所有容器名称
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], stdout=subprocess.PIPE, text=True)
        all_names = result.stdout.strip().split('\n')
        
        # 正则筛选
        pattern = re.compile(CONTAINER_NAME_REGEX)
        targets = [name for name in all_names if pattern.match(name)]
        return targets
    except Exception as e:
        print(f"Error fetching containers: {e}")
        return []

def check_node_bgp(container_name):
    """
    检查单个节点的 BGP 状态
    返回: (是否有问题, 问题协议列表)
    """
    try:
        cmd = f"docker exec {container_name} {CMD_CHECK_PROTOCOLS}"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            return True, [f"ERROR: Command failed (Bird not running?) - {result.stderr.strip()}"]

        lines = result.stdout.strip().split('\n')
        failed_protocols = []
        
        # 解析输出
        # Typical output:
        # Name       Proto    Table    State  Since       Info
        # to_rr_1    BGP      master4  up     10:00:00    Established
        # x_as2      BGP      master4  start  10:00:00    Idle
        
        for line in lines:
            parts = line.split()
            if len(parts) < 6: continue
            
            name, proto, _, state, _, info = parts[0], parts[1], parts[2], parts[3], parts[4], parts[-1]
            
            # 这里的 info 可能是时间，也可能是状态，取决于 bird 版本和列宽
            # 更稳妥的方法是看整行是否包含 'BGP' 且不包含 'Established'
            
            if proto == "BGP":
                # 关键判断：如果是 BGP 协议，且状态栏里没有 Established
                if "Established" not in line:
                    # 记录这个有问题的协议
                    # 格式: 协议名 (当前状态)
                    # 尝试提取最后的状态信息
                    status_info = info if info not in ["Running", "up", "start"] else "Unknown/Idle"
                    failed_protocols.append(f"{name} [{status_info}]")

        if failed_protocols:
            return True, failed_protocols
        else:
            return False, []

    except Exception as e:
        return True, [f"ERROR: Execution exception - {str(e)}"]

def main():
    logger = Logger()
    
    # 1. 发现容器
    logger.log_both("🔍 Scanning for target containers...")
    targets = get_target_containers()
    
    if not targets:
        logger.log_both("❌ No containers found matching the pattern.")
        return

    logger.log_both(f"✅ Found {len(targets)} nodes. Starting BGP check...\n")
    
    # 2. 遍历检查
    total_checked = 0
    nodes_with_issues = 0
    total_failed_sessions = 0
    
    for i, cname in enumerate(targets):
        # 进度条效果
        sys.stdout.write(f"\rchecking {i+1}/{len(targets)}: {cname}...")
        sys.stdout.flush()
        
        has_issue, failures = check_node_bgp(cname)
        total_checked += 1
        
        if has_issue:
            nodes_with_issues += 1
            total_failed_sessions += len(failures)
            
            # 记录详细日志
            logger.log_file(f"\n[NODE] {cname}")
            for fail in failures:
                logger.log_file(f"  ❌ {fail}")
        else:
            # 如果没问题，可以选择不记录，或者只记录一行 OK
            # logger.log_file(f"[NODE] {cname} - All Clean")
            pass

    # 3. 总结输出
    sys.stdout.write("\n") # 换行
    logger.log_both("-" * 40)
    logger.log_both("📊 Summary Report")
    logger.log_both(f"   Total Nodes Checked: {total_checked}")
    logger.log_both(f"   Healthy Nodes:       {total_checked - nodes_with_issues}")
    logger.log_both(f"   Nodes with Issues:   {nodes_with_issues}")
    logger.log_both(f"   Total Down Sessions: {total_failed_sessions}")
    logger.log_both("-" * 40)
    
    if nodes_with_issues > 0:
        logger.log_both(f"⚠️  Detailed error list has been saved to the log file.")
    else:
        logger.log_both(f"🎉  All BGP sessions are Established!")

    logger.close()

if __name__ == "__main__":
    main()