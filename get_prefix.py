import json
import requests
import re
import sys
import pickle
import time
RIS_PREFIXLIST_URL = 'https://stat.ripe.net/data/announced-prefixes/data.json'
def get_prefix(asn):
    try:
        #print(asn)
        rslt = requests.get(RIS_PREFIXLIST_URL, {
            'resource': asn
        }).json()
        # 尝试获取前缀
        prefix = rslt['data']['prefixes'][0]['prefix']
        return prefix
    except IndexError:
        print(asn,'IndexError')
        # 当列表索引越界时，返回当前的rslt
        return None
    except Exception as e:
        # 其他异常也可以返回rslt或错误信息（可选）
        print(asn,f"发生其他错误: {e}")
        return None  # 或根据需要返回其他标识，如None

def load_topology_data(filename: str) -> dict:
    """从文件加载拓扑数据（兼容 key: value 格式）"""
    with open(filename, 'r') as f:
        content = f.read().strip()
    
    # 处理格式：添加外层大括号，替换冒号为冒号+引号，处理列表
    # 1. 替换 key: 为 'key':
    content = re.sub(r'^(\w+):', r'"\1":', content, flags=re.MULTILINE)
    # 2. 每行末尾添加逗号（最后一行除外）
    lines = content.split('\n')
    lines = [line + ',' for line in lines[:-1]] + [lines[-1]] if lines else []
    content = '\n'.join(lines)
    # 3. 包裹成字典
    content = '{' + content + '}'
    
    # 安全解析为字典
    try:
        return eval(content)  # 此处使用eval是因为处理后的格式已符合Python字典规范
    except Exception as e:
        raise ValueError(f"解析拓扑数据失败: {e}")

t=time.time()
try:
    TOPOLOGY_DATA = load_topology_data('real_topology.txt')
except FileNotFoundError:
    print("错误: 未找到real_topology.txt文件")
    sys.exit(1)
    
prefix_dict = {}
for asn in TOPOLOGY_DATA['transit_asns']:
    asn_int = int(asn)  # 转换为整数ASN
    result = get_prefix(asn_int)
    
    # if isinstance(result, dict):
    #     raise ValueError(f"获取ASN {asn_int} 的前缀失败，返回了JSON数据: {result}")
    
    # 若不是字典，则认为是有效前缀（字符串类型），存入字典
    prefix_dict[asn_int] = result

print(f"获取前缀耗时: {time.time() - t} 秒")
with open("my_dict.pkl", "wb") as f:
    pickle.dump(prefix_dict, f)
print("字典已成功存储为 my_dict.pkl")