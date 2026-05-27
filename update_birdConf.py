import os
import re
import yaml
import subprocess
import sys

# ================= 配置区域 =================

# 1. 文件路径配置
TARGET_FILENAME = "2b0ae038330eccd43095538618caee7d"  # BIRD 配置文件名(Hash)
BASE_DIR = "./output"                                  # output 目录路径
COMPOSE_FILE = os.path.join(BASE_DIR, "docker-compose.yml")
CONTAINER_DEST_PATH = "/etc/bird/bird.conf"            # 容器内目标路径

# 2. BGP 参数配置 (修改此处数值; 如果不想修改某项，请设置为 None)
BGP_KEEPALIVE = 60     # 单位: 秒
BGP_HOLD_TIME = 36000     # 单位: 秒

# 3. OSPF 参数配置 (修改此处数值; 如果不想修改某项，请设置为 None)
OSPF_HELLO = 30         # Hello interval
OSPF_DEAD = 36000         # Dead interval
OSPF_RETRANSMIT = 20    # Retransmit interval
OSPF_TICK = 3 
# ===========================================

def load_docker_compose_map(compose_path):
    """
    解析 docker-compose.yml，返回 {service_name: container_name} 的字典
    """
    if not os.path.exists(compose_path):
        print(f"Error: 找不到 {compose_path}")
        sys.exit(1)
        
    print(f"正在解析 {compose_path} ...")
    try:
        with open(compose_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            
        mapping = {}
        if 'services' in data:
            for service_name, config in data['services'].items():
                if 'container_name' in config:
                    mapping[service_name] = config['container_name']
        return mapping
    except Exception as e:
        print(f"Error: 解析 docker-compose.yml 失败: {e}")
        sys.exit(1)

def update_config_content(content):
    """
    使用正则根据全局配置修改内容
    """
    new_content = content

    # --- 修改 BGP 参数 ---
    if BGP_KEEPALIVE is not None:
        # 匹配 keepalive time <数字>; 并替换
        new_content = re.sub(r'(keepalive time\s+)\d+;', f'\\g<1>{BGP_KEEPALIVE};', new_content)
    
    if BGP_HOLD_TIME is not None:
        new_content = re.sub(r'(hold time\s+)\d+;', f'\\g<1>{BGP_HOLD_TIME};', new_content)

    # --- 修改 OSPF 参数 ---
    if OSPF_HELLO is not None:
        new_content = re.sub(r'(hello\s+)\d+;', f'\\g<1>{OSPF_HELLO};', new_content)
        
    if OSPF_DEAD is not None:
        new_content = re.sub(r'(dead\s+)\d+;', f'\\g<1>{OSPF_DEAD};', new_content)
        
    if OSPF_RETRANSMIT is not None:
        new_content = re.sub(r'(retransmit\s+)\d+;', f'\\g<1>{OSPF_RETRANSMIT};', new_content)

    if OSPF_RETRANSMIT is not None:
        new_content = re.sub(r'(tick\s+)\d+;', f'\\g<1>{OSPF_TICK};', new_content)

    return new_content

def main():
    # 打印当前配置确认
    print("-" * 30)
    print(f"即将应用以下配置:")
    print(f"BGP: Keepalive={BGP_KEEPALIVE}, Hold={BGP_HOLD_TIME}")
    print(f"OSPF: Hello={OSPF_HELLO}, Dead={OSPF_DEAD}, Retransmit={OSPF_RETRANSMIT}")
    print("-" * 30)

    # 1. 获取 Service 到 Container 的映射
    service_to_container = load_docker_compose_map(COMPOSE_FILE)
    
    # 2. 遍历 output 目录
    if not os.path.exists(BASE_DIR):
        print(f"Error: 目录 {BASE_DIR} 不存在")
        sys.exit(1)

    count = 0
    # 获取目录下所有文件夹
    for folder_name in os.listdir(BASE_DIR):
        folder_path = os.path.join(BASE_DIR, folder_name)
        
        # 过滤条件：必须是目录，且名字包含 'brdnode'
        if os.path.isdir(folder_path) and "brdnode" in folder_name:
            target_file = os.path.join(folder_path, TARGET_FILENAME)
            
            # 检查那个哈希文件是否存在
            if not os.path.exists(target_file):
                # 某些 brdnode 目录可能没有这个特定的 conf 文件，跳过不报错，避免刷屏
                continue

            # 3. 读取并修改文件
            with open(target_file, 'r') as f:
                original_content = f.read()
            
            modified_content = update_config_content(original_content)
            
            # 如果内容有变化才写入
            if modified_content != original_content:
                with open(target_file, 'w') as f:
                    f.write(modified_content)
                # print(f"[修改] {folder_name} 本地文件已更新")
            else:
                pass # 内容无需修改

            # 4. 查找对应的 container_name 并执行 docker cp
            # 假设文件夹名就是 service 名
            service_name = folder_name
            container_name = service_to_container.get(service_name)
            
            if container_name:
                cmd = [
                    "docker", "cp", 
                    target_file, 
                    f"{container_name}:{CONTAINER_DEST_PATH}"
                ]
                try:
                    # 执行命令
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"[CP成功] {service_name} -> {container_name}")
                        count += 1
                    else:
                        print(f"[CP失败] {service_name}: {result.stderr.strip()}")
                except Exception as e:
                    print(f"[执行异常] {service_name}: {e}")
            else:
                print(f"[警告] 找不到 Service: {service_name} 对应的 Container Name")

    print(f"\n全部处理完成。共更新并上传了 {count} 个容器。")
    print("提示: 请视情况重启容器或执行 'birdc configure' 使配置生效。")

if __name__ == "__main__":
    main()