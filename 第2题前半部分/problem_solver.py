import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
plt.style.use('fivethirtyeight')


class Solver:
    def __init__(self) -> None:
        self.community = pd.read_excel('Data/community.xlsx')
        self.intersection_node = pd.read_excel('Data\intersection_node.xlsx')
        self.intersection_route = pd.read_excel('Data\intersection_route.xlsx')

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

    def community_connect_intersection(self):
        community = self.community
        intersection_node = self.intersection_node
        community_intersection = list()
        for i in range(community.shape[0]):
            community_node_x_index = community.iloc[i, 4]
            community_node_y_index = community.iloc[i, 5]
            distance = list()
            for j in range(intersection_node.shape[0]):
                intersection_node_x = intersection_node.iloc[j, 1]
                intersection_node_y = intersection_node.iloc[j, 2]
                di = np.sqrt((community_node_x_index - intersection_node_x)
                             ** 2 + (community_node_y_index - intersection_node_y)**2)
                distance.append(di)
            community_intersection.append(np.argmin(distance) + 1)
        community['最近路口节点'] = community_intersection
        community.to_csv('TempResult\社区路口连接数据.csv', index=False)

    def intersection_area(self):
        """本函数用于计算有小区连着的路口节点所属的行政区"""
        community_intersection = pd.read_csv('TempResult\社区路口连接数据.csv')
        a = community_intersection.groupby(
            ['最近路口节点', '所属区域'], as_index=False).count()
        groups = a.groupby('最近路口节点')
        intersection_node = list()
        intersection_areas = list()
        intersection_communities = list()
        for i in groups:
            node, node_data = i
            areas = node_data['所属区域'].tolist()
            counts = node_data['小区编号'].tolist()
            intersection_node.append(node)
            intersection_areas.append(areas[np.argmax(counts)])
            intersection_communities.append(np.sum(counts))
        result = pd.DataFrame({
            '路口节点': intersection_node,
            '路口节点所属行政区': intersection_areas,
            '路口节点连接社区数': intersection_communities})
        result.to_csv('TempResult\路口连接行政区数据.csv', index=False)

    def intersection_area_update(self):
        """本函数通过从上一个函数得出的结果，给所有路口节点赋予行政区的标签"""
        label_data = pd.read_csv('TempResult\路口连接行政区数据.csv')
        unlabel_data = pd.read_excel('Data\intersection_node.xlsx')
        train_data = pd.merge(left=unlabel_data,
                              right=label_data, how="right", left_on='节点编号', right_on='路口节点')[['路口节点', '路口节点所属行政区', '路口节点连接社区数', '路口横坐标', '路口纵坐标']]
        # train_data.to_csv('TempResult\路口训练集.csv', index=False)
        areas = list(set(train_data['路口节点所属行政区'].tolist()))
        areas_number = list(range(9))
        areas2number = dict(zip(areas, areas_number))
        number2areas = dict(zip(areas_number, areas))
        train_data['路口节点所属行政区'] = train_data['路口节点所属行政区'].apply(
            lambda x: areas2number[x])
        x_train = train_data[['路口横坐标', '路口纵坐标']]
        y_train = train_data['路口节点所属行政区']
        test_data = pd.merge(left=unlabel_data,
                             right=label_data, how="left", left_on='节点编号', right_on='路口节点')[['节点编号', '路口横坐标', '路口纵坐标', '路口节点连接社区数']]
        test_data['路口节点连接社区数'] = test_data['路口节点连接社区数'].fillna(0)
        test_data = test_data[test_data['路口节点连接社区数'] == 0]
        knn = KNeighborsClassifier().fit(x_train, y_train)
        x_test = test_data[['路口横坐标', '路口纵坐标']]
        y_test_pre = knn.predict(x_test)
        test_data['路口节点所属行政区'] = y_test_pre
        train_data['节点编号'] = train_data['路口节点']
        train_data = train_data[['节点编号', '路口横坐标',
                                 '路口纵坐标', '路口节点连接社区数', '路口节点所属行政区']]
        res = pd.concat([train_data, test_data])
        res = res.sort_values(by='节点编号', ascending=True)
        res['路口节点所属行政区'] = res['路口节点所属行政区'].apply(
            lambda x: number2areas[x])
        res.to_csv('TempResult\路口行政区小区数据.csv', index=False)

    def route_area(self):
        """本函数用于计算路线的行政区划,同时计算公路密度"""
        route_data = pd.read_excel('Data\intersection_route.xlsx')
        intersection_data = pd.read_csv('TempResult\路口行政区小区数据.csv')
        start_area = list()
        end_area = list()
        for i in range(route_data.shape[0]):
            start = route_data.iloc[i, 1]
            end = route_data.iloc[i, 2]
            start_area.append(intersection_data.iloc[start-1, 4])
            end_area.append(intersection_data.iloc[end-1, 4])
        route_data['起点行政区'] = start_area
        route_data['终点行政区'] = end_area
        # route_data.to_csv('TempResult\带行政区的公路线.csv', index=False)
        route_data = route_data[['路线距离(m)', '起点行政区']]
        area_route_length = route_data.groupby(
            ['起点行政区'], as_index=False)['路线距离(m)'].sum()
        area_route_length.to_csv('TempResult\各行政区公路线总长度.csv', index=False)
        area_route_length.to_excel('TempResult\各行政区公路线总长度.xlsx', index=False)

    def draw_map_with_area(self):
        intersection_route = pd.read_csv('TempResult\带行政区的公路线.csv')
        intersection_node = self.intersection_node
        community = self.community
        routes = intersection_route[['路线起点', '路线终点', '起点行政区']]
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
                area = routes.iloc[i, 2]
                color = area_color_dict[area]
                plt.plot([x_index[start], x_index[end]], [y_index[start],
                                                          y_index[end]], color=color, alpha=0.6, lw=0.2, zorder=0.2)
            except:
                print(start, end)
        for i in range(community.shape[0]):
            x_index = community.iloc[i, 4]
            y_index = community.iloc[i, 5]
            area = community.iloc[i, 7]
            plt.scatter(x_index, y_index, color=area_color_dict[area], s=2)
        plt.show()

    def cluster(self):
        """计算该路口所连接的路线数"""
        route_data = pd.read_excel('Data\intersection_route.xlsx')
        G = nx.DiGraph()
        for i in range(route_data.shape[0]):
            start_idx = route_data.iloc[i, 1]
            end_idx = route_data.iloc[i, 2]
            weight = route_data.iloc[i, 3]
            G.add_edge(start_idx, end_idx, weight=weight)
        a = nx.pagerank(G)
        node_list = list(a.keys())
        pr_list = list()
        for i in node_list:
            pr_list.append(a[i])
        pr_list = np.array(pr_list)
        pr_list = (pr_list-np.min(pr_list))/(np.max(pr_list)-np.min(pr_list))
        aa = pd.DataFrame({
            '节点编号': node_list,
            'pr值': pr_list
        })
        b = dict(G.degree())
        node_list = list(b.keys())
        degree_list = list()
        for i in node_list:
            degree_list.append(b[i])
        degree_list = np.array(degree_list)
        degree_list = (degree_list-np.min(degree_list)) / \
            (np.max(degree_list)-np.min(degree_list))
        bb = pd.DataFrame({
            '节点编号': node_list,
            '度值': degree_list
        })
        res = pd.merge(aa, bb, on='节点编号', how="inner")
        left = pd.read_csv('TempResult\路口行政区小区数据.csv')
        cluster_data = pd.merge(left, res, on='节点编号', how='left')
        right = pd.read_csv('TempResult\社区路口连接数据.csv')
        right = right.groupby(['最近路口节点'], as_index=False)['小区人口数（人）'].sum()
        right['节点编号'] = right['最近路口节点']
        right = right.drop(columns=['最近路口节点'])[['节点编号', '小区人口数（人）']]
        cluster_data = pd.merge(cluster_data, right, on='节点编号', how='left')
        cluster_data['小区人口数（人）'] = cluster_data['小区人口数（人）'].fillna(0)
        temp = np.array(cluster_data['小区人口数（人）'])
        cluster_data['小区人口数（人）'] = (
            temp-np.min(temp))/(np.max(temp)-np.min(temp))
        temp = np.array(cluster_data['路口节点连接社区数'])
        cluster_data['路口节点连接社区数'] = (
            temp-np.min(temp))/(np.max(temp)-np.min(temp))
        train_data = cluster_data[['路口节点连接社区数', 'pr值', '度值', '小区人口数（人）']]
        # k = list(range(2,10))
        # pinggu =list()
        # for ki in k:
        #     kmeans = KMeans(n_clusters=ki, random_state=42)
        #     y_pred = kmeans.fit_predict(train_data)
        #     pinggu.append(kmeans.inertia_)
        # plt.plot(k,pinggu)
        kmeans = KMeans(n_clusters=5, random_state=42)
        y_pred = kmeans.fit_predict(train_data)
        cluster_data['节点类别'] = y_pred
        label_translate = dict(
            zip([0, 1, 2, 3, 4], ['一般', '冷清', '较冷清', '繁忙', '较繁忙']))
        cluster_data['节点类别'] = cluster_data['节点类别'].apply(
            lambda x: label_translate[x])
        cluster_data.to_csv('TempResult\行政区统计数据.csv', index=False)

    def compute_area_info(self):
        data = pd.read_csv('TempResult\行政区统计数据.csv')
        groups = data.groupby(['路口节点所属行政区'])
        area_list = list()
        mean_degree = list()
        busy_percent = list()
        for i in groups:
            area, group_data = i
            area_list.append(area)
            mean_degree.append(group_data['度值'].mean())
            length = group_data.shape[0]
            busy_number = group_data[group_data['节点类别'] == '繁忙'].shape[0] + \
                group_data[group_data['节点类别'] == '较繁忙'].shape[0]
            busy_percent.append(busy_number/length)
        res = pd.DataFrame({
            '区域名称': area_list,
            '平均度': mean_degree,
            '繁忙百分比': busy_percent
        })
        data = pd.read_excel('TempResult\行政区指标.xlsx')
        result = pd.merge(res, data, on='区域名称', how='inner')
        result['指标'] = result['生活物资投放点数量'] / \
            result['隔离人口数（万人）'] * result['公路距离'] / \
            result['行政面积'] * result['平均度']*result['繁忙百分比']
        result.to_csv('Result/first_result.csv', index=False)
        result.to_excel('Result/first_result.xlsx', index=False)
