import os
import json
import sys

# 尝试导入 PyYAML，如果不存在则提示用户安装
try:
    import yaml
except ImportError:
    print("❌ 错误: 缺少 'pyyaml' 库。")
    print("请运行: pip install pyyaml")
    sys.exit(1)

# ================= 配置区域 =================
# docker-compose.yml 文件路径
COMPOSE_FILE = os.path.join("output", "docker-compose.yml")

# 输出的 JSON 文件名
OUTPUT_FILE = "container_meta.json"

# Service 名称过滤前缀
SERVICE_PREFIX = "brdnode"

# 需要提取的 Label 键名 (对应 JSON 中的 key 和 Label 中的 key)
TARGET_LABELS = {
    "net_1_address": "org.seedsecuritylabs.seedemu.meta.net.1.address",
    "net_2_address": "org.seedsecuritylabs.seedemu.meta.net.2.address"
}
# ===========================================

def main():
    # 1. 检查文件是否存在
    if not os.path.exists(COMPOSE_FILE):
        print(f"❌ 找不到文件: {COMPOSE_FILE}")
        print("请确保脚本运行在包含 'output' 目录的文件夹中。")
        return

    print(f"📂 正在读取: {COMPOSE_FILE} ...")

    # 2. 解析 YAML
    try:
        with open(COMPOSE_FILE, 'r', encoding='utf-8') as f:
            compose_data = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ 解析 YAML 失败: {e}")
        return

    extracted_data = []
    
    # 获取所有 services
    services = compose_data.get('services', {})
    if not services:
        print("⚠️  文件中没有找到 'services' 定义。")
        return

    print(f"🔍 正在扫描 {len(services)} 个服务，寻找以 '{SERVICE_PREFIX}' 开头的服务...")

    # 3. 遍历并提取信息
    match_count = 0
    for service_name, config in services.items():
        # 过滤 service 名称
        if not service_name.startswith(SERVICE_PREFIX):
            continue

        match_count += 1
        
        # 获取 container_name
        container_name = config.get('container_name')
        if not container_name:
            # 如果没有显式设置 container_name，通常 docker-compose 会自动生成
            # 这里我们标记为 Unknown 或跳过，视需求而定，通常 SeedEmu 会指定它
            container_name = f"{service_name}_(implicit)"

        # 获取 labels
        labels = config.get('labels', {})
        # labels 可能是 list 格式 ("key=value") 也可能是 dict 格式
        # docker-compose 规范允许两种，通常是 dict。这里做兼容处理。
        labels_dict = {}
        if isinstance(labels, list):
            for label_str in labels:
                if "=" in label_str:
                    k, v = label_str.split("=", 1)
                    labels_dict[k.strip()] = v.strip()
        elif isinstance(labels, dict):
            labels_dict = labels

        # 构建数据对象
        entry = {
            "service_name": service_name,
            "container_name": container_name
        }

        # 提取目标 Label
        for json_key, label_key in TARGET_LABELS.items():
            # 使用 .get(key, None) 以防 label 不存在
            entry[json_key] = labels_dict.get(label_key, None)

        extracted_data.append(entry)

    # 4. 输出 JSON
    if extracted_data:
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=4, ensure_ascii=False)
            print(f"✅ 成功提取 {len(extracted_data)} 条数据。")
            print(f"💾 结果已保存至: {os.path.abspath(OUTPUT_FILE)}")
            
            # 打印前 2 条作为预览
            print("\n👀 数据预览 (前2条):")
            print(json.dumps(extracted_data[:2], indent=2))
        except IOError as e:
            print(f"❌ 写入文件失败: {e}")
    else:
        print(f"⚠️  未找到任何以 '{SERVICE_PREFIX}' 开头的服务。")

if __name__ == "__main__":
    main()