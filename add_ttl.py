import sys
from ruamel.yaml import YAML

# 定义要处理的文件名
INPUT_FILE = './output/docker-compose.yml'
OUTPUT_FILE = './output/docker-compose.yml'

def add_ttl_to_services():
    # 初始化 YAML 实例
    yaml = YAML()
    yaml.preserve_quotes = True  # 尽量保留引号格式
    yaml.indent(mapping=4, sequence=4, offset=2) # 设置缩进格式，尽量匹配你的风格

    try:
        print(f"正在读取 {INPUT_FILE} ...")
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = yaml.load(f)

        # 检查是否存在 services 字段
        if 'services' not in data:
            print("错误：在文件中未找到 'services' 字段！")
            return

        count = 0
        # 遍历所有服务
        for service_name, service_config in data['services'].items():
            # 目标配置
            ttl_config = "net.ipv4.ip_default_ttl=255"

            # 检查该服务是否已有 sysctls 字段
            if 'sysctls' not in service_config:
                # 如果没有，直接创建列表
                service_config['sysctls'] = [ttl_config]
                count += 1
            else:
                # 如果已有，先检查是不是列表格式
                if isinstance(service_config['sysctls'], list):
                    # 避免重复添加
                    if ttl_config not in service_config['sysctls']:
                        service_config['sysctls'].append(ttl_config)
                        count += 1
                elif isinstance(service_config['sysctls'], dict):
                    # 有些 compose 版本支持字典格式，我们也兼容一下
                    if "net.ipv4.ip_default_ttl" not in service_config['sysctls']:
                        service_config['sysctls']["net.ipv4.ip_default_ttl"] = "255"
                        count += 1

        print(f"处理完成！共修改了 {count} 个服务。")
        
        # 保存到新文件（为了安全，不直接覆盖原文件）
        print(f"正在保存到 {OUTPUT_FILE} ...")
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(data, f)
            
        print("\n✅ 成功！请检查 'docker-compose-fixed.yml' 文件。")
        print("确认无误后，可以将原文件重命名备份，然后将新文件改名为 docker-compose.yml")

    except FileNotFoundError:
        print(f"错误：找不到文件 {INPUT_FILE}，请确保脚本和yml文件在同一目录下。")
    except Exception as e:
        print(f"发生未知错误：{e}")

if __name__ == "__main__":
    add_ttl_to_services()