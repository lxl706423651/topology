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
import heapq

ccs=0.1
path='gnutella'
datasets=[]
f=open("./dataset/"+path+'/'+path+".txt")
data=f.read()
f.close()
rows=data.split('\n')
for row in rows:
    split_row=row.split(' ')
    name=(int(split_row[0]),int(split_row[1]),0.01)
    datasets.append(name)

G=nx.DiGraph()
G.add_weighted_edges_from(datasets)   #根据数据集创建有向图

del data,rows,datasets
gc.collect()

node_list=list(G.nodes)
nodenum=sorted(node_list,reverse=True)[0]+1

total_nodes=len(G.nodes)#总节点数
total_edges=len(G.edges)#总边数
print('total_nodes:',total_nodes, 'total_edges:',total_edges)

# --- 修改开始 ---

k = 120  # 【在此处设置】你想要随机选择的社区数量
print(f"Plan to select {k} random communities.")

# 1. 第一步：读取文件并统计所有出现的社区
raw_community_dict = {}
all_communities_pool = set()

with open("./dataset/"+path+'/'+path+"_mem.txt", 'r') as f:
    for line in f:
        data = line.strip().split()
        node_id = int(data[0])
        group_ids = data[1:]
        
        # 暂时存入原始字典
        raw_community_dict[node_id] = group_ids
        # 收集所有社区ID用于抽样
        all_communities_pool.update(group_ids)

# 2. 第二步：从所有社区中随机抽取 k 个
if len(all_communities_pool) <= k:
    selected_communities = list(all_communities_pool)
    print(f"Warning: k is larger than total communities. Selected all {len(selected_communities)} communities.")
else:
    selected_communities = list(all_communities_pool)[:k]

print(f"Successfully selected {len(selected_communities)} communities.")

# 3. 第三步：更新图节点，只保留属于 selected_communities 的社区
community_set = set() # 用于最后统计实际用到的社区

# 遍历图中的所有节点进行初始化和赋值
for node in G.nodes():
    # 默认为空列表（即没有社区）
    G.nodes[node]['community'] = []
    
    # 如果该节点在原始文件中有记录
    if node in raw_community_dict:
        # 核心逻辑：取交集。只保留那些“被选中”的社区ID
        # list comprehension: 遍历节点原有的社区，只有在 selected_communities 里的才保留
        filtered_groups = [gid for gid in raw_community_dict[node] if gid in selected_communities]
        
        if filtered_groups:
            G.nodes[node]['community'] = filtered_groups
            community_set.update(filtered_groups)

no_community_num = sum(1 for n in G.nodes() if len(G.nodes[n]['community']) == 0)

print(f"没有社区的节点数量: {no_community_num}")
print(f"占比: {no_community_num / total_nodes:.2%}") # 顺便看下比例

# --- 修改结束 ---

# 下面是原本的边权重计算逻辑（保持不变，但因为上面确保了每个节点都有 community 属性，所以不会报错）
for u, v, data in G.edges(data=True):
    G[u][v]['b_uv'] = G[u][v]['weight']
    # 注意：这里引用了 G.nodes[v]['community']，上面必须确保所有节点都被初始化过
    if len(G.nodes[v]['community']) == 0: 
        G[u][v]['q_uv'] = 0
    else:
        G[u][v]['q_uv'] = ccs * (1 - G[u][v]['b_uv']) / len(G.nodes[v]['community'])

community_list = list(community_set)
print('community_list len:', len(community_list))

def RRS(g=G,t=node_list):# rr-seed
    seed = random.choice(t)
    result_list = []
    result_list.append(seed)
    RRS=[]
    RRS.append(seed)
    # 保存激活的状态
    checked = np.zeros(nodenum,dtype=bool) 
    for node in result_list:
        checked[node] = 1
    # 当前节点不为空，进行影响
    while len(result_list) != 0:
        current_node = result_list.pop(0)
        for nbr in g.predecessors(current_node): # 得到当前节点的邻居节点
            if checked[nbr] == 0:
                wt = g.get_edge_data(nbr,current_node)
                if random.uniform(0,1) < wt['weight'] :
                    result_list.append(nbr)
                    checked[nbr] = 1
                    RRS.append(nbr)
    return RRS

def generate_1_hop_vertex_cover(G):
    # 创建图的副本，避免修改原图
    graph = G.copy()
    vertex_cover = set()
    
    # 当图中还有边时继续迭代
    while graph.edges():
        # 随机选择一条边
        edge = random.choice(list(graph.edges()))
        
        # 随机选择边的一个顶点
        vertex = random.choice(edge)
        
        # 将顶点加入覆盖集合
        vertex_cover.add(vertex)
        
        # 从图中删除该顶点及其所有关联边
        graph.remove_node(vertex)
    
    return vertex_cover
def generate_2_hop_vertex_cover(G):
    def two_hop_neighbors(G, v):
        # 获取顶点v的邻居
        neighbors_v = set(G.neighbors(v))

        # 获取邻居的邻居
        two_hop_neigh = set()
        for neighbor in neighbors_v:
            two_hop_neigh.update(G.neighbors(neighbor))

        # 两跳的邻居包括邻居的邻居，排除v自己和v的直接邻居
        two_hop_neigh -= neighbors_v
        two_hop_neigh.discard(v)  # 确保不包含v本身

        return two_hop_neigh
    graph=G.copy()
    sh = []
    flag=1
    while(1):
        out_degrees = graph.out_degree()
        max_node = max(out_degrees, key=lambda x: x[1])  # (节点, 出度)
        node=max_node[0]
        visited=two_hop_neighbors(graph, node)
        if(len(visited)==0):
            break
        graph.remove_node(node)
        sh.append(node)
        sh.extend(visited)
        graph.remove_nodes_from(visited)
    print(len(graph.nodes()),len(graph.edges()))
    return sh

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

def find_two_hop_paths_with_max_probability(graph, start): #p1 
    # 存储每个目标节点的前两条最大概率路径
    paths = defaultdict(list)
    
    # 深度优先搜索DFS来查找两跳路径
    def dfs(node, path, prob, depth):
        if depth == 2:
            target = path[-1]
            # 如果两跳路径的目标已经存在，则将新路径插入
            paths[target].append((path, prob))
            return
        
        if node in graph:
            for neighbor in graph.successors(node):
                # 获取当前边的权重（概率）
                edge_prob = graph[node][neighbor]['weight']
                dfs(neighbor, path + [neighbor], prob * edge_prob, depth + 1)
    
    # 查找直接的两跳路径
    for target in graph.successors(start):
        prob = graph[start][target]['weight']
        paths[target].append(([start, target], prob))
    
    # 开始DFS遍历图
    dfs(start, [start], 1, 0)
    
    # 对每个目标节点按概率排序，并仅保留概率最大的两条路径
    result = {}
    for target in paths:
        # 按照路径的总概率排序
        sorted_paths = sorted(paths[target], key=lambda x: x[1], reverse=True)
        result[target] = sorted_paths[:2]  # 取概率最大的两条路径
    
    return result
def find_reverse_two_hop_paths_with_max_probability(graph, start):
    # 创建反向图
    reverse_graph = graph.reverse()  # 反向图
    
    # 存储每个目标节点的前两条最大概率路径
    paths = defaultdict(list)
    
    # 深度优先搜索DFS来查找反向两跳路径
    def dfs(node, path, prob, depth):
        if depth == 2:
            target = path[-1]
            # 如果反向两跳路径的目标已经存在，则将新路径插入
            paths[target].append((path, prob))
            return
        
        if node in reverse_graph:
            for neighbor in reverse_graph.successors(node):
                # 获取当前边的权重（概率）
                edge_prob = reverse_graph[node][neighbor]['weight']  # 默认权重为1
                dfs(neighbor, path + [neighbor], prob * edge_prob, depth + 1)
    
    # 查找直接的反向两跳路径
    for target in reverse_graph.successors(start):
        prob = reverse_graph[start][target]['weight']
        paths[target].append(([start, target], prob))
    
    # 开始DFS遍历反向图
    dfs(start, [start], 1, 0)
    
    # 对每个目标节点按概率排序，并仅保留概率最大的两条路径
    result = {}
    for target in paths:
        # 按照路径的总概率排序
        sorted_paths = sorted(paths[target], key=lambda x: x[1], reverse=True)
        result[target] = sorted_paths[:2]  # 取概率最大的两条路径
    
    return result

def pf_path(result,s,path):
    path[s]={}
    for target, paths in result.items():
        if len(paths)==1: path[s][target]=[paths[0][0]]
        else:
            path[s][target]=[paths[0][0],paths[1][0]]
    return path

def get_forward_probability(G,path_list):
    if len(path_list)==2:
        return G[path_list[0]][path_list[1]]['weight']
    elif len(path_list)==3:
        return G[path_list[0]][path_list[1]]['weight']*G[path_list[1]][path_list[2]]['weight']
    else:
        print(path_list,"get_forward_probability有误！")
        return None

def get_reverse_probability(G,path_list):
    if len(path_list)==2:
        return G[path_list[1]][path_list[0]]['weight']
    elif len(path_list)==3:
        return G[path_list[2]][path_list[1]]['weight']*G[path_list[1]][path_list[0]]['weight']
    else:
        print(path_list,"get_reverse_probability有误！")
        return None

#计算IMM
def RRS_ori(g=G,t=node_list):#最原始的rr-seed 
    seed = random.choice(t)
    result_list = []
    result_list.append(seed)
    RRS=[]
    RRS.append(seed)
    # 保存激活的状态
    checked = np.zeros(nodenum,dtype=bool) 
    for node in result_list:
        checked[node] = 1
    # 当前节点不为空，进行影响
    while len(result_list) != 0:
        current_node = result_list.pop(0)
        for nbr in g.predecessors(current_node): # 得到当前节点的邻居节点
            if checked[nbr] == 0:
                wt = g.get_edge_data(nbr,current_node)
                if random.uniform(0,1) < wt['weight'] :
                    result_list.append(nbr)
                    checked[nbr] = 1
                    RRS.append(nbr)
    return RRS
def combination(n, k):
    if k < 0 or k > n:
        return 0
    # 利用对称性，C(n, k) == C(n, n-k)
    if k > n - k:
        k = n - k
    result = 1
    for i in range(1, k + 1):
        result = result * (n - i + 1) // i
    return result
def lnc(n,k):
    return math.log(combination(n, k))
def nodeselect(RR,k):#对RRSet作种子选择
    s=time.time()
    R=RR.copy()
    js=0
    SEED=[]
    for _ in range(k):
        flat_map = [item for subset in R for item in subset]
        if(len(flat_map))==0:
            return SEED,0
        seed = Counter(flat_map).most_common()[0][0]
        js=js+Counter(flat_map).most_common()[0][1]
        SEED.append(seed)
        R = [rrs for rrs in R if seed not in rrs]
    return SEED,js
def IMM_RRS(G,k,eps,ell,node_list=node_list,n=total_nodes):
    s = time.time()
    # 参数计算（保持不变）
    ell=ell*(1+math.log(2)/math.log(n))
    eps1=math.sqrt(2) * eps
    alpha = math.sqrt(ell * math.log(n) + math.log(2))
    beta = math.sqrt((1.0 - 1.0 / math.exp(1)) * (lnc(n,k) + alpha * alpha))
    lamba1 = (2*n*(1+1/3*eps1) * (lnc(n,k)+ell*math.log(n)+math.log(math.log(n)/math.log(2))))/(eps1*eps1)
    lamba2 = 2 * n * pow(((1.0 - 1.0 / math.exp(1)) * alpha + beta),2) / (eps*eps)
    LB=1
    # 初始化映射（b直接存储RR集，a和cover_counts实时更新）
    a = defaultdict(list)  # a[node] = [rrset_idx1, ...]
    b = []  # b[rr_idx] = [node1, ...]（替代原R）
    cover_counts = defaultdict(int)  # 节点覆盖计数
    
    len_b=0
    # 校准阶段
    for i in range(1, int(math.log(n-1) / math.log(2)) + 1):
        x = n / pow(2, i)
        ti = int(lamba1 / x)
        # 生成RR集并实时更新映射
        
        while len_b < ti:
            rrset = RRS_ori(G, node_list)
            b.append(rrset)
            for node in rrset:
                a[node].append(len_b)
                cover_counts[node] += 1
            len_b=len(b)
        
        # 无需复制，直接传递原始映射（函数内部不会修改它们）
        remaining_rrs = set(range(len(b)))
        SEED, js = optimized_nodeselect(k, a, b, cover_counts, remaining_rrs)
        
        if n * js / len(b) >= (1 + eps1) * x:
            LB = n * js / ((1 + eps1) * len(b))
            break
    
    # 最终采样阶段
    theta = int(lamba2 / LB)
    print(f"len(RR): {theta}")
    
    # 继续生成RR集至目标数量
    
    while len_b < theta:
        rrset = RRS_ori(G, node_list)
        b.append(rrset)
        for node in rrset:
            a[node].append(len_b)
            cover_counts[node] += 1
        len_b=len(b)
    
    # 最终种子选择（同样无需复制）
    remaining_rrs = set(range(len(b)))
    seed, js = optimized_nodeselect(k, a, b, cover_counts, remaining_rrs)
    
    print(f"IMM_RRS COST TIME: {time.time() - s:.2f}s")
    return seed

def optimized_nodeselect(k, a, b, cover_counts, remaining_rrs):
    """无需复制原始映射，仅在内部维护临时状态"""
    s = time.time()
    SEED = []
    total_covered = 0
    
    # 仅复制需要修改的状态（覆盖计数和剩余RR集）
    current_cover = cover_counts.copy()  # 复制需要修改的计数
    current_remaining = remaining_rrs.copy()  # 复制需要修改的集合
    
    for _ in range(k):
        if not current_remaining or not current_cover:
            break
        
        # 选择覆盖最多的节点
        seed = max(current_cover, key=lambda x: current_cover[x])
        SEED.append(seed)
        
        # 找到覆盖的RR集
        covered_rrs = [rr_idx for rr_idx in a[seed] if rr_idx in current_remaining]
        total_covered += len(covered_rrs)
        
        # 更新剩余RR集
        for rr_idx in covered_rrs:
            current_remaining.discard(rr_idx)
        
        # 更新其他节点的覆盖计数
        for rr_idx in covered_rrs:
            for node in b[rr_idx]:
                if node != seed and node in current_cover:
                    current_cover[node] -= 1
                    if current_cover[node] == 0:
                        del current_cover[node]
        
        # 移除已选节点
        del current_cover[seed]
    
    #print(f"节点选择耗时: {time.time() - s:.2f}秒")
    return SEED, total_covered

with open("./dataset/"+path+'/'+"sh.pkl", "rb") as f:
    sh=pickle.load(f)
    
G1=update_weight(G,community_list)#G1是F=所有topic
t=time.time()
pf = {}
pr = {}

for s in sh:
    result=find_two_hop_paths_with_max_probability(G1,s)
    if result == None: continue
    pf=pf_path(result,s,pf)




print('find_two_hop_paths_with_max_probability',time.time()-t)
# del reverse_graph
# gc.collect()

t=time.time()
graph={}
for i in community_list:
    graph[i]=update_weight(G,i)
    
print('gragh:',time.time()-t)

t=time.time()
If={}

for node in sh:
    If[node]={}
    for key,value in pf[node].items():
        x0=0
        flag=len(value)
        temp_dict={}
        if flag==1: x0=get_forward_probability(G,value[0])
        elif flag==2: 
            x0=1-(1-get_forward_probability(G,value[0]))*(1-get_forward_probability(G,value[1]))
        for i in community_list:
            # 里面其实还可以存一个标识 表示是只有一条路径得到的概率，还是2条路径得到的概率（后续可以采用不同的计算方法）
            if flag==1: temp_dict[i]=get_forward_probability(graph[i],value[0])
            elif flag==2: 
                w=1-(1-get_forward_probability(graph[i],value[0]))*(1-get_forward_probability(graph[i],value[1]))
                temp_dict[i]=w
        If[node][key]=[flag,x0,temp_dict]

del pf
gc.collect()

Ir={}
# for node in sh:
#     Ir[node]={}
#     for nbr in G.predecessors(node): # 得到当前节点的邻居节点
#         x0 = G.get_edge_data(nbr,node)
#         flag=1
#         temp_dict={}
#         for i in community_list:
#             temp_dict[i]=graph[i].get_edge_data(nbr,node)
#         Ir[node][nbr]=[flag,x0,temp_dict]

print('Ir If cost time:',time.time()-t)

print('len(sh):',len(sh))

num=0
for node in sh:
    num=num+len(If[node])
print('len(If[node])+len(Ir[node])',num)

#获得最新的索引表
def update_p_weight(F,flag,x0,temp):
    X=[]
    for i in F:
        X.append(temp[i])
    if flag == 1:
        return sum(X)-(len(X)-1)*x0
    elif x0==1:
        return 1
    else:
        m = len(X) - 1  # 这里假设 X 里有 m+1 个元素，X[0] 是 X_0，X[m] 是 X_m
        sum_part1 = sum(1 - X[j] for j in range(m))  # 计算第一部分的和
        sum_part2 = 0
        for i in range(m):
            for j in range(i+1, m):
                sum_part2 += (1 - x0) + (1 - X[i]) * (1 - X[j]) / (1 - x0)  # 计算第二部分的和
        part3 = (m - 1) * math.sqrt(1 - x0)  # 第三部分
        result = 1 - (math.sqrt(sum_part1 + sum_part2) - part3) ** 2  # 最终计算结果
        return max(result,max(X))
def update_index_table(If,Ir,F,sh):
    t=time.time()
    new_forward_label={}
    for node in sh:
        new_forward_label[node]={}
        for key,value in If[node].items():
            flag=value[0]
            x0=value[1]
            temp_dict=value[2]
            new_forward_label[node][key]=update_p_weight(F,flag,x0,temp_dict)
    
    print(f"new_forward_label cost time:{time.time()-t}")
    return new_forward_label

def reverse_dict(Ir):
    I = {}
    # 遍历 Ir 字典
    for key1, value1 in Ir.items():
        for key2, value2 in value1.items():
            if key2 not in I:
                I[key2] = {}
            I[key2][key1] = value2
    return I

#通过If,Ir中不受topic影响的概率值也就是x0部分将其存储成ori_forward_label,ori_reverse_label来给出一个排序

#算两跳内的影响力
def calculate_two_hop_probability(graph, start_node):
    hop_probabilities = {}
    
    # 计算第一跳的激活概率
    first_hop_probs = {}
    for neighbor in graph.successors(start_node):  # 只考虑a的邻居
        # 计算第一跳传播概率
        hop_probabilities[neighbor] = graph[start_node][neighbor]['weight']  # 保存第一跳的概率
    # 计算第二跳的激活概率
    for node in graph.successors(start_node):
        for second_node in graph.successors(node):  # 第二跳从node的邻居开始
            # 计算第二跳的传播概率
            if(second_node not in hop_probabilities.keys()):hop_probabilities[second_node]=graph[node][second_node]['weight']*graph[start_node][node]['weight']
            else:
                hop_probabilities[second_node]=1-(1-hop_probabilities[second_node])*(1-graph[node][second_node]['weight']*graph[start_node][node]['weight']) 
    return hop_probabilities

t22=time.time()
ori_forward_label={}
for key1 in If.keys():
    ori_forward_label[key1]={}
    for key2 in If[key1].keys():
        ori_forward_label[key1][key2]=If[key1][key2][1]

# ori_reverse_label={}
# for key1 in Ir.keys():
#     ori_reverse_label[key1]={}
#     for key2 in Ir[key1].keys():
#         ori_reverse_label[key1][key2]=Ir[key1][key2][1]
# ori_reverse_label=reverse_dict(ori_reverse_label)

ori_yxl={}
for node in node_list:
    temp={}
    if node in ori_forward_label.keys():
        for key,value in ori_forward_label[node].items():
            if key in temp.keys(): temp[key]=max(value,temp[key])
            else:temp[key]=value
            if key in ori_forward_label.keys():
                for key1,value1 in ori_forward_label[key].items():
                    if key1 in temp.keys(): temp[key1]=max(value1*temp[key],temp[key1])
                    else:temp[key]=value1*temp[key]
    else:
        t=0
        temp=calculate_two_hop_probability(G, node)#获取两跳内的影响力 calculate_two_hop_probability
        temp1=temp.copy()
        if len(temp)!=0:
            for key,value in temp1.items():
                if key in ori_forward_label.keys():
                    for key1,value1 in ori_forward_label[key].items():
                        if key1 in temp.keys(): temp[key1]=max(value1*value,temp[key1])
                        else:
                            temp[key1]=value1*value
    ori_yxl[node]=sum(temp.values())
sorted_list = sorted(ori_yxl.items(), key=lambda x: x[1], reverse=True)

print('sorted_list cost time:',time.time()-t22)

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

def insert_list(new_tuple,new_sorted_list,k):
    for i in range(0,k-1):
        if(new_sorted_list[k-2-i][1]>new_tuple[1]):
            new_sorted_list.insert(k-1-i,new_tuple)
            new_sorted_list=new_sorted_list[:k]
            return new_sorted_list
    new_sorted_list.insert(0,new_tuple)
    new_sorted_list=new_sorted_list[:k]
    return new_sorted_list
def update_node_influence(G,node,new_forward_label):
    temp={}
    if node in new_forward_label.keys():
        for key,value in new_forward_label[node].items():
            if key in temp.keys(): temp[key]=max(value,temp[key])
            else:temp[key]=value
            if key in new_forward_label.keys():
                for key1,value1 in new_forward_label[key].items():
                    if key1 in temp.keys(): temp[key1]=max(value1*temp[key],temp[key1])
                    else:temp[key]=value1*temp[key]
    else:
        t=0
        temp=calculate_two_hop_probability(G, node)#获取两跳内的影响力 calculate_two_hop_probability
        temp1=temp.copy()
        if len(temp)!=0:
            for key,value in temp1.items():
                if key in new_forward_label.keys():
                    for key1,value1 in new_forward_label[key].items():
                        if key1 in temp.keys(): temp[key1]=max(value1*value,temp[key1])
                        else:
                            temp[key1]=value1*value
    return sum(temp.values())

def prune_labels(new_forward_label):
    t=time.time()
    best_for_b = {}
    pruned = {}
    for a, b_dict in new_forward_label.items():
        pruned[a]={}
        for b, value in b_dict.items():
            # 如果 b 尚未出现，或当前 value 更大，则更新
            if b not in best_for_b or value > best_for_b[b][1]:
                best_for_b[b] = (a, value)
    for b, (a, value) in best_for_b.items():
        pruned[a][b] = value
    for key,temp in pruned.items():
        pruned[key]=sum(temp.values())+1
    print(f"jz cost time:{time.time()-t}")
    return pruned

def update_node_influence2(G,node,pruned,new_forward_label):
    temp=0
    if node in new_forward_label.keys():
        for key in new_forward_label[node].keys():
            if key in pruned.keys(): temp+=new_forward_label[node][key]*pruned[key]
            else: temp+=new_forward_label[node][key]
    else:
        for nbr in G.successors(node): # 得到当前节点的邻居节点
            value = G.get_edge_data(node,nbr)['weight']
            if nbr in pruned.keys():
                temp+=value*pruned[nbr]
            else:
                temp+=value
    return temp

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
    for i in range(top_count):
        community_id, count = sorted_communities[i]
        community_list.append(str(community_id))
    return community_list

if path=='gnutella' or path=='soc-Epinions1':
    F=count_communities(path,top_count=10)
else:
    F=community_list[:10]
F=community_list[10:15]
newG=update_weight(G,F)

t2=time.time()
new_forward_label=update_index_table(If,Ir,F,sh)
tt1=time.time()-t2
pruned=prune_labels(new_forward_label)
tt2=time.time()-t2-tt1
#将方向取正
print('new_forward_label',tt1)
print('prune_labels',tt2)

for k in [20,200]:
    print(k,'#########################################################################')
    repeat=100
    t1=time.time()
    seed4=IMM_RRS(newG,k,0.5,1)
    print('time:',time.time()-t1,'IMM influence:')
    pg_influence(newG,repeat,seed4)
    t2=time.time()
    new_sorted_list=[]
    for i in range(k):
        node=sorted_list[i][0]
        yxl=update_node_influence(newG,node,new_forward_label)
        new_sorted_list.append((node,yxl))
    new_sorted_list = sorted(new_sorted_list, key=lambda x: x[1], reverse=True)
    temp=0
    for i in range(k,5*k):
        node=sorted_list[i][0]
        
        yxl=update_node_influence(newG,node,new_forward_label)
        if(yxl>new_sorted_list[k-1][1]):
            temp=0
            new_tuple=(node,yxl)
            new_sorted_list=insert_list(new_tuple,new_sorted_list,k)
        else:
            temp=temp+1
            if(temp>=k):
                print(i)
                break
    seed3=[item[0] for item in new_sorted_list[:k]]
    print('time:',time.time()-t2+tt1,'stop round:',i,'my method1 influence:')
    pg_influence(newG,repeat,seed3)
    
    
    t3=time.time()
    new_sorted_list=[]
    for i in range(k):
        node=sorted_list[i][0]
        yxl=update_node_influence2(newG,node,pruned,new_forward_label)
        new_sorted_list.append((node,yxl))
    new_sorted_list = sorted(new_sorted_list, key=lambda x: x[1], reverse=True)

    temp=0
    for i in range(k,10*k):
        node=sorted_list[i][0]
        yxl=update_node_influence2(newG,node,pruned,new_forward_label)
        if(yxl>new_sorted_list[k-1][1]):
            temp=0
            new_tuple=(node,yxl)
            new_sorted_list=insert_list(new_tuple,new_sorted_list,k)
        else:
            temp=temp+1
            if(temp>=2*k):
                break
    seed3=[item[0] for item in new_sorted_list[:k]]
    print('time:',time.time()-t3+tt1+tt2,'stop round:',i,'my method2 influence:')
    pg_influence(newG,repeat,seed3)

