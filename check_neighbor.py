import json
import subprocess
import re
import os
import time

# ========== 你原有的函数，原样保留 ==========
def run_nsenter_cmd(pid, shell_cmd, async_mode=False):
    """使用 nsenter 进入容器执行任意 Shell 命令"""
    # -n 网络命名空间, -m 挂载命名空间
    full_cmd = f"nsenter -t {pid} -n -m {shell_cmd}"
    if async_mode:
        subprocess.Popen(full_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return None
    else:
        return subprocess.run(full_cmd, shell=True, capture_output=True, text=True)

# ========== 核心配置（按需修改这3个即可） ==========
JSON_FILE_PATH = "./container_pids.json"  # 你的容器pid文件路径
TARGET_AS_NUM = 1272                      # 要检测的AS号
OSPF_INSTANCE = "ospf1"                   # 固定检测ospf1实例
LOG_DIR = "./logs"                        # 日志存放目录

# ========== 新增：日志输出函数（控制台+文件双写） ==========
def print_and_log(log_content, log_file, end='\n'):
    """控制台打印+日志文件写入，统一输出方法"""
    print(log_content, end=end)
    log_file.write(log_content + end)

# ========== 主检测逻辑 ==========
def check_ospf_neighbors_status():
    # 1. 自动创建logs目录，不存在则创建
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        print(f"✅ 自动创建日志目录: {LOG_DIR}")

    # 2. 日志文件命名：ospf_check_年月日_时分秒.log
    log_file_name = f"ospf_check_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(LOG_DIR, log_file_name)

    # 3. 读取并解析container_pids.json
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            container_pid_dict = json.load(f)
        init_info = f"✅ 成功读取容器PID文件，共加载 {len(container_pid_dict)} 个容器信息"
    except FileNotFoundError:
        init_info = f"❌ 错误：未找到文件 {JSON_FILE_PATH}，请检查文件路径！"
        print(init_info)
        return
    except json.JSONDecodeError:
        init_info = f"❌ 错误：{JSON_FILE_PATH} 文件格式不是合法的JSON，请检查！"
        print(init_info)
        return

    # 4. 打开日志文件，开始写入+检测
    with open(log_file_path, 'w', encoding='utf-8') as log_f:
        # 写入日志头部信息
        print_and_log("="*80, log_f)
        print_and_log(f"OSPF邻居状态检测日志 - AS号: {TARGET_AS_NUM}", log_f)
        print_and_log(f"检测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}", log_f)
        print_and_log(f"日志文件: {log_file_path}", log_f)
        print_and_log("="*80, log_f)
        print_and_log(init_info, log_f)

        # 统计变量
        total_as_container = 0    # 匹配到的目标AS容器数
        total_abnormal = 0        # 异常容器总数
        abnormal_container_list = [] # 异常容器清单

        # 5. 遍历所有容器，筛选目标AS号的容器
        for container_name, pid in container_pid_dict.items():
            # 匹配规则：容器名以 as{AS号} 开头，精准匹配如 as1272xxxx
            if container_name.startswith(f"as{TARGET_AS_NUM}"):
                total_as_container += 1
                split_line = "-" * 80
                print_and_log(split_line, log_f)
                print_and_log(f"\n📌 正在检测容器：【{container_name}】，PID: {pid}", log_f)
                
                # 6. 进入容器执行birdc命令，查询ospf1邻居
                bird_cmd = f"birdc show ospf neighbors"
                cmd_result = run_nsenter_cmd(pid, bird_cmd)
                
                # 7. 命令执行结果判断
                if cmd_result.returncode != 0:
                    err_msg = cmd_result.stderr.strip() if cmd_result.stderr else "执行命令无返回结果"
                    err_info = f"  ❌ 容器执行失败：{err_msg}"
                    print_and_log(err_info, log_f)
                    total_abnormal +=1
                    abnormal_container_list.append(f"{container_name} - birdc命令执行失败")
                    continue

                cmd_output = cmd_result.stdout.strip()
                if not cmd_output or OSPF_INSTANCE not in cmd_output:
                    empty_info = f"  ⚠️  该容器无OSPF邻居信息 / 未配置ospf1实例"
                    print_and_log(empty_info, log_f)
                    continue

                # 8. 正则匹配OSPF邻居的核心信息（完美适配你的输出格式）
                # 匹配行：10.0.1.158        1     Full/PtP        2986.894        net_99_158 5.248.1.254
                neighbor_pattern = re.compile(
                    r'(\d+\.\d+\.\d+\.\d+)\s+'          # 分组1: Router ID
                    r'(\d+)\s+'                         # 分组2: Pri
                    r'(\w+\/\w+)\s+'                    # 分组3: State (核心校验字段)
                    r'([\d\.]+)\s+'                     # 分组4: DTime
                    r'(\w+_\d+_\d+)\s+'                 # 分组5: Interface
                    r'(\d+\.\d+\.\d+\.\d+)'             # 分组6: Router IP
                )
                neighbor_matches = neighbor_pattern.findall(cmd_output)
                
                if not neighbor_matches:
                    no_neighbor_info = f"  ⚠️  该容器未发现有效OSPF邻居"
                    print_and_log(no_neighbor_info, log_f)
                    continue

                # 9. 逐个校验邻居状态，Full开头=正常，其他=异常
                container_has_abnormal = False
                normal_info = f"  ✅ 共发现 {len(neighbor_matches)} 个OSPF邻居，状态检测如下："
                print_and_log(normal_info, log_f)
                for match in neighbor_matches:
                    router_id, pri, state, dtime, interface, router_ip = match
                    # 核心判定规则：State以Full开头 即为正常
                    if state.startswith("Full"):
                        neighbor_ok = f"     ✔ RouterID:{router_id} | 状态:{state} | 接口:{interface} | 邻居IP:{router_ip} → 正常"
                        print_and_log(neighbor_ok, log_f)
                    else:
                        neighbor_err = f"     ❌ RouterID:{router_id} | 状态:{state} | 接口:{interface} | 邻居IP:{router_ip} → 异常【非Full状态】"
                        print_and_log(neighbor_err, log_f)
                        container_has_abnormal = True

                # 10. 记录异常容器
                if container_has_abnormal:
                    total_abnormal +=1
                    abnormal_container_list.append(f"{container_name} - 存在非Full状态的OSPF邻居")

        # ========== 最终检测汇总报告 ==========
        print_and_log("\n" + "="*80, log_f)
        print_and_log(f"\n📊 OSPF邻居状态检测【汇总报告】- 目标AS号: {TARGET_AS_NUM}", log_f)
        print_and_log(f"📥 本次检测的容器总数：{total_as_container} 个", log_f)
        print_and_log(f"❌ 存在异常的容器数：{total_abnormal} 个", log_f)
        if abnormal_container_list:
            print_and_log(f"\n异常容器明细：", log_f)
            for idx, abnormal in enumerate(abnormal_container_list, 1):
                print_and_log(f"  {idx}. {abnormal}", log_f)
        else:
            print_and_log(f"\n🎉 所有 {total_as_container} 个容器的OSPF邻居全部为【Full正常状态】，检测通过！", log_f)
        print_and_log("\n" + "="*80, log_f)

    # 最后打印日志文件保存路径，方便查找
    print(f"\n✅ 本次检测日志已完整保存至：{log_file_path}")

# 执行脚本
if __name__ == "__main__":
    check_ospf_neighbors_status()