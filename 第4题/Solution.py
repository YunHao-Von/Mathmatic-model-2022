import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from numpy import linalg as LA
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
plt.style.use('fivethirtyeight')

class Solution:
    def __init__(self) -> None:
        self.community = pd.read_excel('Data/community.xlsx')
        self.intersection_node = pd.read_excel('Data/intersection_node.xlsx')
        self.intersection_route = pd.read_excel('Data/intersection_route.xlsx')
    
    def run(self):
        # self.draw_map()  # 画出全市地图
        pass
    
    
    
    def build_graph(self):
        """本函数用于构建无向图"""
        community_route = self.community_neraest_intersection()
        intersection_route = self.intersection_route
        G = nx.Graph()
        intersection_edges = list()
        community_edges = list()
        for i in range(intersection_route.shape[0]):
            start = intersection_route.iloc[i,1]  # 路线起点
            end = intersection_route.iloc[i,2]
            weight = intersection_route.iloc[i,3]
            intersection_edges.append(('i'+str(start),'i'+str(end),weight))
        for i in range(community_route.shape[0]):
            start = community_route.iloc[i,0]  # 路线起点
            end = community_route.iloc[i,8]
            weight = community_route.iloc[i,9]
            community_edges.append(('c'+str(start),'i'+str(end),weight))
        G.add_weighted_edges_from(intersection_edges)
        G.add_weighted_edges_from(community_edges)
        return G
    
    
    
    
    
    
    def community_neraest_intersection(self):
        """本函数用于计算每个小区最近的路口"""
        if os.path.exists('TempResult/社区路口连接数据.csv'):
            return pd.read_csv('TempResult/社区路口连接数据.csv')
        else:
            intersection = self.intersection_node
            community = self.community
            intersection_index = np.array(intersection[['路口横坐标','路口纵坐标']])
            community_index = np.array(community[['小区横坐标','小区纵坐标']])
            nearest_intersection = list()
            distance = list()
            for i in range(community_index.shape[0]):
                community_index_i = np.array([community_index[i]])
                res = LA.norm(np.array([community_index_i[:, 0] - intersection_index[:, 0], 
                                        community_index_i[:, 1] - intersection_index[:, 1]]), axis=0)
                nearest_intersection.append(np.argmin(res)+1)
                distance.append(np.min(res)* 320)
            community['最近路口节点'] = nearest_intersection
            community['路口路线距离'] = distance
            community.to_csv("TempResult/社区路口连接数据.csv",index=False)
            return community
            
            
            
    def draw_map(self):
        intersection_route = self.intersection_route
        intersection_node = self.intersection_node
        community = self.community
        routes = intersection_route[['路线起点', '路线终点']]
        x_index = intersection_node['路口横坐标'].tolist()
        y_index = intersection_node['路口纵坐标'].tolist()
        area = list(community['所属区域'].value_counts().index)
        colors = ['red', 'yellow', 'black', 'green',
                  'orange', 'purple', 'blueviolet', 'cyan', 'tan']
        area_color_dict = dict(zip(area, colors))
        plt.figure(figsize=(10, 10))
        for i in range(routes.shape[0]):
            try:
                start = routes.iloc[i, 0]-1
                end = routes.iloc[i, 1]-1
                plt.plot([x_index[start], x_index[end]], [y_index[start],
                                                          y_index[end]], color='b', alpha=0.6, lw=0.2, zorder=0.2)
            except:
                print(start, end)
        for i in range(community.shape[0]):
            x_index = community.iloc[i, 4]
            y_index = community.iloc[i, 5]
            area = community.iloc[i, 7]
            plt.scatter(x_index, y_index, color=area_color_dict[area], s=2)
        plt.show()
    
    def draw_solution(self):
        sort_intersection = pd.read_csv('无货车结果(集散中心).csv')['节点编号'].tolist()
        intersection_route = self.intersection_route
        intersection_node = self.intersection_node
        community = self.community
        routes = intersection_route[['路线起点', '路线终点']]
        x_index = intersection_node['路口横坐标'].tolist()
        y_index = intersection_node['路口纵坐标'].tolist()
        area = list(community['所属区域'].value_counts().index)
        colors = ['red', 'yellow', 'black', 'green',
                  'orange', 'purple', 'blueviolet', 'cyan', 'tan']
        area_color_dict = dict(zip(area, colors))
        plt.figure(figsize=(10, 10))
        for i in range(routes.shape[0]):
            try:
                start = routes.iloc[i, 0]-1
                end = routes.iloc[i, 1]-1
                plt.plot([x_index[start], x_index[end]], [y_index[start],
                                                          y_index[end]], color='b', alpha=0.6, lw=0.2, zorder=0.2)
            except:
                print(start, end)
        for i in sort_intersection:
            xi = x_index[i-1]
            yi = y_index[i-1]
            plt.scatter(xi,yi,marker='*', color='red', s=100)
        for i in range(community.shape[0]):
            x_index = community.iloc[i, 4]
            y_index = community.iloc[i, 5]
            area = community.iloc[i, 7]
            plt.scatter(x_index, y_index, color=area_color_dict[area], s=2)
        cangku = pd.read_csv('无货车结果(仓库).csv')
        for i in range(cangku.shape[0]):
            x_index = cangku.iloc[i, 2]
            y_index = cangku.iloc[i, 3]
            plt.scatter(x_index, y_index, color='blue',marker='*', s=200)
        plt.show()

    def draw_solution_car(self):
        sort_intersection = pd.read_csv('有货车结果(集散中心).csv')['节点编号'].tolist()
        intersection_route = self.intersection_route
        intersection_node = self.intersection_node
        community = self.community
        routes = intersection_route[['路线起点', '路线终点']]
        x_index = intersection_node['路口横坐标'].tolist()
        y_index = intersection_node['路口纵坐标'].tolist()
        area = list(community['所属区域'].value_counts().index)
        colors = ['red', 'yellow', 'black', 'green',
                  'orange', 'purple', 'blueviolet', 'cyan', 'tan']
        area_color_dict = dict(zip(area, colors))
        plt.figure(figsize=(10, 10))
        for i in range(routes.shape[0]):
            try:
                start = routes.iloc[i, 0]-1
                end = routes.iloc[i, 1]-1
                plt.plot([x_index[start], x_index[end]], [y_index[start],
                                                          y_index[end]], color='b', alpha=0.6, lw=0.2, zorder=0.2)
            except:
                print(start, end)
        for i in sort_intersection:
            xi = x_index[i-1]
            yi = y_index[i-1]
            plt.scatter(xi,yi,marker='*', color='red', s=100)
        for i in range(community.shape[0]):
            x_index = community.iloc[i, 4]
            y_index = community.iloc[i, 5]
            area = community.iloc[i, 7]
            plt.scatter(x_index, y_index, color=area_color_dict[area], s=2)
        cangku = pd.read_csv('有货车结果(仓库).csv')
        for i in range(cangku.shape[0]):
            x_index = cangku.iloc[i, 2]
            y_index = cangku.iloc[i, 3]
            plt.scatter(x_index, y_index, color='blue',marker='*', s=200)
        plt.show()