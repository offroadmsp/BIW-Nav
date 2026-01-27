import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
import time

# 参数配置
UPDATE_INTERVAL = 1.0  # 每次更新间隔（秒）
DIST_THRESHOLD = 3.0   # 建立边的距离阈值
NUM_NODES = 30         # 模拟节点总数
CLUSTER_NUM = 3        # 抽象层簇数

# 初始化图
G = nx.Graph()
positions = []

def update_graph(new_point):
    """在地图中添加新节点并更新连接关系"""
    idx = len(positions)
    positions.append(new_point)
    G.add_node(idx, pos=new_point)
    # 更新边
    for i in range(idx):
        dist = np.linalg.norm(np.array(positions[i]) - new_point)
        if dist < DIST_THRESHOLD:
            G.add_edge(i, idx, weight=dist)

def build_abstract_graph():
    """构建抽象层关系"""
    if len(positions) < 3:
        return None, None  # 不足以聚类
    Z = linkage(np.array(positions), method='ward')
    clusters = fcluster(Z, t=CLUSTER_NUM, criterion='maxclust')
    abstract_graph = nx.Graph()
    for c in np.unique(clusters):
        abstract_graph.add_node(c)
    for i, j in G.edges():
        ci, cj = clusters[i], clusters[j]
        if ci != cj:
            abstract_graph.add_edge(ci, cj)
    return clusters, abstract_graph

# 模拟动态建图过程
plt.ion()  # 开启交互式绘图
for step in range(NUM_NODES):
    # 模拟机器人探索新位置
    new_point = np.random.rand(2) * 10
    update_graph(new_point)
    clusters, abstract_graph = build_abstract_graph()

    # 绘图
    plt.clf()
    if clusters is not None:
        pos_dict = {i: positions[i] for i in G.nodes()}
        plt.subplot(121)
        nx.draw(G, pos_dict, node_color=clusters, with_labels=True, cmap=plt.cm.Set3)
        plt.title(f"Step {step+1}: Node-Level Map")
        
        plt.subplot(122)
        nx.draw(abstract_graph, with_labels=True, node_color="lightblue", node_size=800)
        plt.title("Abstract-Level Relationship Graph")
    else:
        plt.text(0.3, 0.5, f"Building initial map ({step+1}/{NUM_NODES})", fontsize=12)
    
    plt.pause(UPDATE_INTERVAL)

plt.ioff()
plt.show()
