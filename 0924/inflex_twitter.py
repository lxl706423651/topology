import networkx as nx
import numpy as np
import math
import heapq

import heapq
import networkx as nx

class InflexAlgorithm:
    def __init__(self, G, num_topics, index_size=1000, ell=200):
        """
        :param G: NetworkX DiGraph
        :param num_topics: 话题总数 (对应你说的索引点数量)
        :param index_size: 每个话题的索引大小 (即 RR-Set 的数量/采样次数)
        :param ell: 种子集规模上限 (Max Seed Budget)，即 k 的最大值
        """
        self.G = G
        self.num_topics = num_topics
        self.index_size = index_size  # 这里 index_size 决定了采样的丰富度
        self.ell = ell                # 这里 ell 是种子数量上限 (200)
        
        # 存储反向可达集 (RR-Sets)
        # 结构: { topic_id: [RR_Set_1, RR_Set_2, ..., RR_Set_IndexSize] }
        # 这就是你要求的：每个 topic 对应一组独立的索引数据
        self.topic_rr_sets = defaultdict(list)
        self.nodes_list = list(G.nodes())

    def _generate_rr_set(self, target_node, topic_idx):
        """生成针对特定 Topic 的反向可达集"""
        active_nodes = {target_node}
        queue = [target_node]
        
        while queue:
            curr = queue.pop(0)
            # 反向遍历
            for u in self.G.predecessors(curr):
                if u in active_nodes:
                    continue
                
                # 获取边上的 topic_distribution
                # 假设格式是 [prob_topic_0, prob_topic_1, ...]
                dist = self.G[u][curr].get('topic_distribution', [])
                
                if topic_idx < len(dist):
                    prob = dist[topic_idx]
                else:
                    prob = 0.0
                
                # 蒙特卡洛采样
                if random.random() < prob:
                    active_nodes.add(u)
                    queue.append(u)
        return active_nodes

    def build_index(self):
        """
        离线阶段：为每个 Topic 构建 index_size 个 RR-Set
        """
        print(f"Building index: {self.num_topics} topics, {self.index_size} RR-sets per topic...")
        
        for z in range(self.num_topics):
            # 针对每个话题 z，生成 index_size 个 RR-Set
            # 这里的 index_size 对应你代码中计算出的 int(max(len(node_list)/100,1000))
            for _ in range(self.index_size):
                root_node = random.choice(self.nodes_list)
                rr_set = self._generate_rr_set(root_node, z)
                self.topic_rr_sets[z].append(rr_set)

    def query(self, query_gamma, k_target):
        """
        在线阶段：基于 query_gamma 组合不同 Topic 的 RR-Set 进行贪心选择
        """
        # 简单校验 k_target 是否超过上限，虽然超过了也能跑，但通常 ell 是设计的最大值
        k = min(k_target, self.ell)
        
        # 归一化权重
        gamma = np.array(query_gamma)
        if np.sum(gamma) > 0:
            gamma = gamma / np.sum(gamma)
        
        S = []
        
        # 记录覆盖状态: covered_status[topic_id][rr_idx]
        covered_status = {
            z: [False] * len(self.topic_rr_sets[z]) 
            for z in range(self.num_topics)
        }
        
        # -----------------------------------------------------
        # 优化：建立 节点 -> (topic, rr_idx) 的倒排索引
        # 这样不用每次遍历所有节点，只遍历出现在相关 RR-Set 中的节点
        # -----------------------------------------------------
        node_occurrence = defaultdict(list)
        for z in range(self.num_topics):
            if gamma[z] <= 0: continue # 权重为0的话题不参与
            for rr_idx, rr_set in enumerate(self.topic_rr_sets[z]):
                for node in rr_set:
                    node_occurrence[node].append((z, rr_idx))

        # 贪心选择 k 个种子
        for _ in range(k):
            best_node = -1
            max_gain = -1.0
            
            # 计算边际增益 (Marginal Gain)
            # 实际上只需要检查 node_occurrence 中的节点
            # 为了进一步加速，标准的 CELF 算法会维护一个优先队列，
            # 这里为了代码简洁，使用每次重新计算 (对于 Python 来说，如果 index_size 不大尚可接受)
            
            # 这里的 candidates 设为 node_occurrence.keys() 可以大幅减少无效计算
            candidates = [n for n in node_occurrence.keys() if n not in S]
            
            for node in candidates:
                gain = 0.0
                # 检查该节点能覆盖哪些尚未覆盖的 (topic, rr_set)
                for (z, rr_idx) in node_occurrence[node]:
                    if not covered_status[z][rr_idx]:
                        # 增益 = 话题权重 * 1 (因为是覆盖计数)
                        # 严格来说应该是 gamma[z] * (N / index_size)，但 N 和 index_size 是常数，不影响相对排序
                        gain += gamma[z]
                
                if gain > max_gain:
                    max_gain = gain
                    best_node = node
            
            if best_node != -1 and max_gain > 0:
                S.append(best_node)
                # 更新覆盖状态
                for (z, rr_idx) in node_occurrence[best_node]:
                    covered_status[z][rr_idx] = True
            else:
                break
                
        return S
def count_communities(path,top_count=10):
    # 用于存储每个社区的节点数量
    community_counts = {}
    file_path="./dataset/"+path+'/'+path+"_mem.txt"
    with open(file_path, 'r') as file:
        for line in file:
            # 去除行首尾的空白字符
            line = line.strip()
            if not line:
                continue
                
            # 分割行数据，取前两个数字作为节点编号和社区号
            parts = line.split()
            if len(parts) >= 2:
                try:
                    # 节点编号我们不需要实际使用，只需要社区号
                    community_id = int(parts[1])
                    
                    # 统计社区数量
                    if community_id in community_counts:
                        community_counts[community_id] += 1
                    else:
                        community_counts[community_id] = 1
                except ValueError:
                    # 忽略格式不正确的行
                    continue
    
    # 按社区节点数量从大到小排序
    sorted_communities = sorted(community_counts.items(), 
                               key=lambda x: x[1], 
                               reverse=True)
    community_list=[]
    # 取前10个，或者如果总社区数不足10个则取全部
    top_count = 10
    for i in range(top_count):
        community_id, count = sorted_communities[i]
        community_list.append(str(community_id))
    return community_list

def update_weight(g,F):
    def calculate_h_uv(Fv, F, b_uv, q_uv):# 计算 h_uv(Fv, F)
        overlap = len(set(Fv) & set(F))  # 计算 Fv 和 F 的交集大小
        if(q_uv==0): max_increase=0
        else:max_increase = (1 - b_uv) / q_uv  # 根据公式1限制激活概率的上限
        return min(max_increase, overlap)
    G=g.copy()
    # 更新图中每条边的激活概率
    for u, v, data in G.edges(data=True):
        b_uv = data['b_uv']  # 获取基础激活概率
        q_uv = data['q_uv']  # 获取边的增益系数
        F_v = G.nodes[v]['community']  # 获取目标节点 v 的社区集合
        # 根据公式计算新的激活概率
        h_uv_value = calculate_h_uv(F_v, F, b_uv, q_uv)
        new_p_uv = b_uv + q_uv * h_uv_value
        # 更新边的激活概率
        if(new_p_uv!=G[u][v]['weight']):
            G[u][v]['weight'] = new_p_uv
    return G

def forward_single(seed,g):#理论上不需要层级
    #s=time.time()
    result = []
    result.extend(seed)
    # 保存激活的状态
    checked_a = np.zeros(nodenum,dtype=bool)
    for t in result:
        checked_a[t] = 1
    # 当前节点不为空，进行影响
    while(len(result)>0):
        current_node = result.pop(0)
        for nbr in g.successors(current_node): # 得到当前节点的邻居节点
            if checked_a[nbr] == 0 :
                wt = g.get_edge_data(current_node,nbr)
                if random.uniform(0,1)<wt['weight'] :
                    result.append(nbr)
                    checked_a[nbr] = 1
    #print('forward:',time.time()-s)
    return (checked_a != 0).sum()
def pg_influence(G,k,seed_set):#转为列表存储，可以评估方差
    sss=0
    for i in range(k):
        sss+=forward_single(seed_set,G)
    print("influence:",sss/k)
    return sss/k

import copy
import itertools
import random
import math
import networkx as nx
import pandas as pd
import numpy as np
import time
import pickle
from collections import Counter
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import reduce
import gc
path='twitter'
with open("./dataset/"+path+'/'+"sh.pkl", "rb") as f:
    sh=pickle.load(f)
with open("./dataset/"+path+'/'+"G.pkl", "rb") as f:
    G=pickle.load(f)   
with open("./dataset/"+path+'/'+"community_list.pkl", "rb") as f:
    community_list=pickle.load(f)
node_list=list(G.nodes)
total_nodes=len(G.nodes)#总节点数
total_edges=len(G.edges)#总边数
nodenum=sorted(node_list,reverse=True)[0]+1
print('len(community_list)',len(community_list))
print('total_nodes:',total_nodes, 'total_edges:',total_edges)

num_communities = len(community_list)

# 1. 创建社区ID到索引的映射字典，大幅提高查找速度
# 格式: {'社区ID字符串': 0, '社区ID字符串': 1, ...}
comm_to_index = {comm_id: i for i, comm_id in enumerate(community_list)}

# 2. 遍历所有边，构建 topic_distribution
for u, v, data in G.edges(data=True):
    # 初始化一个全为0的列表，长度为 社区总数 + 1
    # 使用 0.0 确保数据类型为 float，如果需要省内存可以用 0
    dist = [0.0] * (num_communities + 1)
    
    # 设置列表第一个数为 b_uv
    dist[0] = data['b_uv']
    
    # 获取 q_uv 的值
    q_val = data['q_uv']
    
    # 获取节点 v 的社区列表，使用 get 防止某些节点没有社区属性
    v_communities = G.nodes[v].get('community', [])
    
    # 遍历节点 v 所属的每一个社区
    for comm_id in v_communities:
        # 查找该社区在 community_list 中的索引
        if comm_id in comm_to_index:
            idx = comm_to_index[comm_id]
            # 根据需求，设置第 x+1 个数为 q_uv
            dist[idx + 1] = q_val
            
    # 将构建好的列表赋值给边的 topic_distribution 属性
    G[u][v]['topic_distribution'] = dist

NUM_TOPICS = num_communities + 1     # 话题数量
INDEX_SIZE = 100000    # 索引中预存的点数

# 3. 初始化 INFLEX 算法
t1=time.time()
inflex = InflexAlgorithm(G, NUM_TOPICS, index_size=INDEX_SIZE, ell=200)

# 4. 构建索引 (离线阶段)
inflex.build_index()
print('inflex_offline_time:',time.time()-t1)
# 2. 设置查询
if path=='gnutella' or path=='soc-Epinions1':
    F=count_communities(path,top_count=10)
else:
    F=community_list[:10]
newG=update_weight(G,F)
query_gamma = [0.0] * (num_communities + 1)
query_gamma[0]=1
for comm_id in F:
    # 查找该社区在 community_list 中的索引
    if comm_id in comm_to_index:
        idx = comm_to_index[comm_id]
        # 根据需求，设置第 x+1 个数为 q_uv
        query_gamma[idx + 1] = 0.099

for k in [20, 30, 50, 100, 150, 200]:
    print(k, '#########################################################################')
    repeat = 100
    
    t2 = time.time()
    # 3. 运行 inflex 算法 (在线查询)
    print("Running inflex Algorithm...")

    # query 中不再需要 seed_budget 参数，因为 k_target 动态传入
    seeds = inflex.query(query_gamma, k_target=k)
    
    print('time:', time.time() - t2, 'inflex influence:')
    pg_influence(newG, repeat, seeds)