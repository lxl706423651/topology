import networkx as nx
import numpy as np
import math
import heapq

import heapq
import networkx as nx

class TopicAwareInfluence:
    def __init__(self, graph, theta=0.01):
        """
        初始化 TopicAwareInfluence 模型
        :param graph: networkx.DiGraph, 边属性包含 'topic_distribution'
        :param theta: 剪枝阈值
        """
        self.graph = graph
        self.theta = theta
        
        # 获取节点列表和最大ID，用于数组索引
        self.nodes = list(graph.nodes())
        if not self.nodes:
            self.max_node_id = 0
            self.num_topics = 0
        else:
            self.max_node_id = max(self.nodes) + 1
            # 获取话题数量 (根据第一条边的 topic_distribution 长度)
            try:
                first_edge = next(iter(graph.edges(data=True)))
                self.num_topics = len(first_edge[2]['topic_distribution'])
            except StopIteration:
                self.num_topics = 0

        # U[u][z] 存储用户 u 在话题 z 下的影响力上界
        # 使用列表而不是字典以提高访问速度 (假设节点ID是整数)
        self.U = [[0.0] * self.num_topics for _ in range(self.max_node_id)]
        
        # 初始化时立即构建离线索引
        self.offline_indexing()

    def offline_indexing(self):
        """
        实现论文中的 Neighborhood-Based Bound Estimation (离线部分)
        计算 U_u^z
        """
        # print("Building offline index (Neighborhood-based)...")
        
        # 1. 计算每个节点在每个话题下的最大出度和最大入度概率
        # pp_out_max[u][z]: u 在话题 z 下对任意邻居的最大影响概率
        # pp_in_max[v][z]: v 在话题 z 下受任意邻居的最大影响概率
        pp_out_max = [[0.0] * self.num_topics for _ in range(self.max_node_id)]
        pp_in_max = [[0.0] * self.num_topics for _ in range(self.max_node_id)]

        # 遍历图中的所有边更新 max 概率
        # NetworkX 遍历边: (u, v, data)
        for u, v, data in self.graph.edges(data=True):
            probs = data.get('topic_distribution', [])
            if not probs: continue
            
            for z in range(self.num_topics):
                val = probs[z]
                if val > pp_out_max[u][z]:
                    pp_out_max[u][z] = val
                if val > pp_in_max[v][z]:
                    pp_in_max[v][z] = val

        # 2. 计算所有节点入度最大概率的总和 (用于优化计算)
        # S_in[z] = sum(pp_in_max[v][z] for all v)
        S_in = [0.0] * self.num_topics
        for z in range(self.num_topics):
            sum_val = 0.0
            for v in self.nodes:
                sum_val += pp_in_max[v][z]
            S_in[z] = sum_val

        # 3. 计算 U[u][z]
        # 公式: U_u^z = (pp_u^o * S_in^z) - sum_neighbors(pp_u^o * pp_v^i) + sum_neighbors(max(pp_u^o, pp_v^i))
        for u in self.nodes:
            # 预先获取 u 的出边邻居，避免重复调用
            neighbors = list(self.graph.successors(u))
            
            for z in range(self.num_topics):
                pp_u_o = pp_out_max[u][z]
                if pp_u_o == 0:
                    continue
                
                # 基础项：假设 u 对所有节点的影响力上界都是 pp_u^o * pp_v^i
                # S_in[z] 包含了所有 v 的 pp_v^i 之和
                bound_sum = pp_u_o * S_in[z]
                
                # 修正项：对于实际相连的邻居 v，我们知道更精确的上界是 max(pp_u^o, pp_v^i)
                # 而不是 loose bound (pp_u^o * pp_v^i)
                # 所以我们要减去 loose bound，加上 tight bound
                for v in neighbors:
                    pp_v_i = pp_in_max[v][z]
                    
                    product_term = pp_u_o * pp_v_i
                    
                    # 获取该边实际的 topic probability 用于判断是否紧密相连
                    # 论文中：如果 (u,v) 存在，则 bound 是 max(pp_u^o, pp_v^i)
                    # 注意：这里我们直接用 max，因为边确实存在
                    tight_bound = max(pp_u_o, pp_v_i)
                    
                    bound_sum = bound_sum - product_term + tight_bound
                
                self.U[u][z] = bound_sum

    def get_edge_prob(self, u, v, gamma):
        """计算边 (u,v) 在查询话题分布 gamma 下的综合传播概率"""
        data = self.graph[u][v]
        probs = data.get('topic_distribution', [])
        # pp(u,v|gamma) = sum(pp_uv^z * gamma^z)
        prob = 0.0
        for z, g in enumerate(gamma):
            if g > 0:
                prob += probs[z] * g
        return prob

    def est_inf_ub(self, u, gamma):
        """
        在线：估计初始影响力的上界
        hat_sigma(u|gamma) = sum(gamma^z * U_u^z)
        """
        ub = 0.0
        u_vals = self.U[u]
        for z, g in enumerate(gamma):
            if g > 0:
                ub += g * u_vals[z]
        return ub

    def est_margin_ub(self, u, ap_u, initial_ub):
        """
        在线：估计边际影响力的上界
        delta_sigma(u|S) <= (1 - ap(u|S)) * hat_sigma(u)
        """
        return (1.0 - ap_u) * initial_ub

    def calc_margin_exact(self, u, seeds, gamma, current_ap):
        """
        在线：计算精确的边际影响力
        使用 Dijkstra/BFS 扩展计算 MIP (Maximum Influence Path)
        """
        marginal_gain = 0.0
        
        # max_prob[v]: 从 u 到 v 的最大路径概率
        max_prob = {u: 1.0}
        # 优先队列存储 (-prob, node)
        pq = [(-1.0, u)] 
        visited = set()

        while pq:
            curr_prob, curr_node = heapq.heappop(pq)
            curr_prob = -curr_prob 
            
            if curr_node in visited:
                continue
            visited.add(curr_node)
            
            # 如果路径概率小于阈值，停止该路径扩展
            if curr_prob < self.theta:
                continue

            # 计算增益：只有当 curr_node 之前没有被 S 完全激活的部分，才能被 u 激活
            # Gain = prob(u activates v) * prob(v not activated by S)
            old_ap = current_ap.get(curr_node, 0.0)
            
            # u 自身不计入 spread (假设种子已经是 active，目的是激活其他人)
            if curr_node != u: 
                 marginal_gain += (1.0 - old_ap) * curr_prob

            # 扩展邻居
            # 使用 networkx 的 successors 获取邻居
            for neighbor in self.graph.successors(curr_node):
                if neighbor not in seeds:
                    edge_p = self.get_edge_prob(curr_node, neighbor, gamma)
                    if edge_p == 0: continue
                    
                    new_prob = curr_prob * edge_p
                    if new_prob >= self.theta:
                        if new_prob > max_prob.get(neighbor, 0.0):
                            max_prob[neighbor] = new_prob
                            heapq.heappush(pq, (-new_prob, neighbor))
                            
        return marginal_gain

    def update_ap(self, new_seed, seeds, gamma, current_ap):
        """
        更新所有节点的激活概率 ap(v|S)
        """
        max_prob = {new_seed: 1.0}
        pq = [(-1.0, new_seed)]
        visited = set()
        
        while pq:
            curr_prob, curr_node = heapq.heappop(pq)
            curr_prob = -curr_prob
            
            if curr_node in visited:
                continue
            visited.add(curr_node)
            
            # 更新全局 AP
            old_ap = current_ap.get(curr_node, 0.0)
            # P(active|S U {u}) = 1 - (1 - P(active|S)) * (1 - P(active|u))
            new_ap_val = 1.0 - (1.0 - old_ap) * (1.0 - curr_prob)
            current_ap[curr_node] = new_ap_val
            
            if curr_prob < self.theta:
                continue
                
            for neighbor in self.graph.successors(curr_node):
                edge_p = self.get_edge_prob(curr_node, neighbor, gamma)
                if edge_p == 0: continue
                
                new_prob = curr_prob * edge_p
                if new_prob > max_prob.get(neighbor, 0.0):
                    max_prob[neighbor] = new_prob
                    heapq.heappush(pq, (-new_prob, neighbor))

    def best_effort_algorithm(self, gamma, k):
        """
        运行 BESTEFFORT 算法
        :param gamma: 查询话题分布列表
        :param k: 种子数量
        :return: 种子集合 set
        """
        # 1. 初始化优先队列 L (基于离线计算的上界)
        L = []
        for u in self.nodes:
            ub = self.est_inf_ub(u, gamma)
            # 只有上界大于0的才有必要放入
            if ub > 0:
                heapq.heappush(L, (-ub, u))
            
        # 2. 在线搜索
        H = [] # 在线堆: (-marginal_gain, node_id, state)
        STATE_INITIAL = 0
        STATE_BOUNDED = 1
        STATE_EXACT = 2
        
        seeds = set()
        current_ap = {u: 0.0 for u in self.nodes} # ap(v|S)
        
        for i in range(k):
            # 将 H 中所有元素状态重置为 Initial (懒惰更新)
            new_H = []
            while H:
                val, u, _ = heapq.heappop(H)
                heapq.heappush(new_H, (val, u, STATE_INITIAL))
            H = new_H
            
            candidate_found = None
            
            while True:
                # 步骤 1: 将 L 中的候选者移入 H
                # 如果 H 为空，或者 L 中最好的上界比 H 中最好的当前值还大
                if L and (not H or -L[0][0] > -H[0][0]):
                    neg_ub, u = heapq.heappop(L)
                    if u not in seeds:
                        heapq.heappush(H, (neg_ub, u, STATE_INITIAL))
                    continue
                
                if not H: 
                    break

                # 步骤 2: 从 H 中寻找种子
                val, u, state = heapq.heappop(H)
                
                if u in seeds: 
                    continue
                
                if state == STATE_INITIAL:
                    # 计算边际上界 (Bounded)
                    init_ub = self.est_inf_ub(u, gamma)
                    ap_u = current_ap.get(u, 0.0)
                    margin_ub = self.est_margin_ub(u, ap_u, init_ub)
                    heapq.heappush(H, (-margin_ub, u, STATE_BOUNDED))
                    
                elif state == STATE_BOUNDED:
                    # 计算精确边际影响力 (Exact)
                    exact_margin = self.calc_margin_exact(u, seeds, gamma, current_ap)
                    heapq.heappush(H, (-exact_margin, u, STATE_EXACT))
                    
                elif state == STATE_EXACT:
                    # 找到最优
                    candidate_found = u
                    break
            
            if candidate_found is not None:
                seeds.add(candidate_found)
                # 更新全局 AP
                self.update_ap(candidate_found, seeds, gamma, current_ap)
            else:
                break
                
        return seeds
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
path='wiki10'
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


# 1. 初始化算法模块
# theta 设为 0.05 加速演示，实际论文建议 0.001
t1=time.time()
model = TopicAwareInfluence(G, theta=0.001)
print('TopicAwareInfluence model build time:',time.time()-t1)
# 2. 设置查询
if path=='gnutella' or path=='soc-Epinions1':
    F=count_communities(path,top_count=5)
else:
    F=community_list[:5]
newG=update_weight(G,F)
query_gamma = [0.0] * (num_communities + 1)
query_gamma[0]=1
for comm_id in F:
    # 查找该社区在 community_list 中的索引
    if comm_id in comm_to_index:
        idx = comm_to_index[comm_id]
        # 根据需求，设置第 x+1 个数为 q_uv
        query_gamma[idx + 1] = 0.099

for k in [20,30,50,100,150,200]:
    print(k,'#########################################################################')
    repeat=100
    
    t2=time.time()
    # 3. 运行 Best-Effort 算法
    print("Running Best-Effort Algorithm...")

    seeds = model.best_effort_algorithm(query_gamma, k)
    print('time:',time.time()-t2,'TopicAwareInfluence influence:')
    pg_influence(newG,repeat,seeds)