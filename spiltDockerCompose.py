import os
from pathlib import Path
from ruamel.yaml import YAML

# --- 配置 ---

# 源文件路径
SOURCE_FILE = Path("output/docker-compose.yml")

# 新配置文件的输出目录
OUTPUT_DIR = SOURCE_FILE.parent

# 每个批次文件包含多少个服务
BATCH_SIZE = 50

# --- 脚本 ---

def split_docker_compose():
    
    # 1. 初始化 YAML 处理器 (保留格式)
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 800
    yaml.indent(mapping=2, sequence=4, offset=2) # 更好的格式化

    # 2. 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"输出目录已准备: {OUTPUT_DIR}")
    print("将删除旧的 batch 文件...")
    # 清理旧文件，防止新旧混淆
    for f in OUTPUT_DIR.glob("services-batch-*.yml"):
        f.unlink()

    # 3. 加载源 YAML 文件
    try:
        with open(SOURCE_FILE, 'r') as f:
            data = yaml.load(f)
    except Exception as e:
        print(f"错误: 无法读取或解析 {SOURCE_FILE}\n{e}")
        return

    if not data:
        print(f"错误: {SOURCE_FILE} 为空或格式不正确。")
        return

    original_version = data.get('version')

    # 4. (步骤一) 处理和保存 'networks'
    # ！！！更新：不再在 common-networks.yml 中写入 version
    # (根据您的日志，version 属性已过时)
    if 'networks' in data:
        networks_data = {'networks': data['networks']}
        networks_path = OUTPUT_DIR / "common-networks.yml"
        
        try:
            with open(networks_path, 'w') as f:
                yaml.dump(networks_data, f)
            print(f"✅ 成功: 已创建 {networks_path} (不含 version)")
        except Exception as e:
            print(f"错误: 无法写入 {networks_path}\n{e}")
    else:
        print("⚠️ 警告: 在源文件中未找到 'networks' 块。")

    # 5. (步骤二) 批量处理和保存 'services'
    if 'services' in data and data['services']:
        
        # ！！！核心修复：
        # 在拆分之前，遍历所有服务，移除 'depends_on' 键。
        # 'depends_on' 只用于 'up'，对 'build' 无用且有害。
        print("\n正在移除 'depends_on' 键以确保构建有效...")
        services_to_split = data['services']
        for service_name, service_config in services_to_split.items():
            if 'depends_on' in service_config:
                del service_config['depends_on']
        
        # 将 services 字典转换为 (key, value) 列表以便切片
        all_services = list(services_to_split.items())
        total_services = len(all_services)
        print(f"正在处理 {total_services} 个服务 (每批 {BATCH_SIZE} 个)...")

        for i in range(0, total_services, BATCH_SIZE):
            batch_services_list = all_services[i : i + BATCH_SIZE]
            batch_services_dict = {key: value for key, value in batch_services_list}
            
            # ！！！更新：也不在 batch 文件中写入 version
            batch_data = {'services': batch_services_dict}
            
            batch_num = (i // BATCH_SIZE) + 1
            batch_filename = f"services-batch-{batch_num:02d}.yml"
            batch_path = OUTPUT_DIR / batch_filename
            
            try:
                with open(batch_path, 'w') as f:
                    yaml.dump(batch_data, f)
                print(f"  ✅ 成功: 已创建 {batch_path} (已移除 depends_on)")
            except Exception as e:
                print(f"  ❌ 错误: 写入 {batch_path} 失败\n{e}")
                
        print("\n🎉 所有服务批次均已拆分完毕！")

    else:
        print("⚠️ 警告: 在源文件中未找到 'services' 块。")

if __name__ == "__main__":
    split_docker_compose()