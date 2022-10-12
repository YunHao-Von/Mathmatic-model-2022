from cnames import cnames
import pandas as pd
import gurobipy as gb
from numpy import linalg as LA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import numpy as np
import os
plt.style.use('fivethirtyeight')
colors = list(cnames.keys())


class ProblemSolver:
    def __init__(self) -> None:
        self.XiaoQu = pd.read_excel('Data\XiaoQu.xlsx')  # 读取小区的坐标和需求数据

    def run(self):
        sortcenter_number = self.get_number_of_SortingCenter()
        # self.draw_cluster_xiaoqu_data()
        return sortcenter_number

    def get_number_of_SortingCenter(self):
        """本函数用于对小区进行聚类从而确定配送中心的数量"""
        XiaoQu = pd.read_excel('Data\XiaoQu.xlsx')
        x_index = np.array(XiaoQu['横坐标'])
        y_index = np.array(XiaoQu['纵坐标'])
        train_data = pd.DataFrame({
            '横坐标': x_index,
            '纵坐标': y_index
        })
        if os.path.exists('Temp\XiaoQu_Cluster.csv'):
            XiaoQu = pd.read_csv('Temp\XiaoQu_Cluster.csv')
        else:
            for i in range(2, 100):
                clust = AgglomerativeClustering(
                    n_clusters=i, linkage='complete')
                y_pred = clust.fit_predict(train_data)
                train_data['节点类别'] = y_pred
                train_data['人口数'] = XiaoQu['小区人口数']
                cluster_population = np.array(
                    train_data.groupby(['节点类别'])['人口数'].sum().tolist())
                decide = all(cluster_population <= 100000)
                if decide:
                    break
            XiaoQu['节点类别'] = y_pred
            XiaoQu.to_csv('Temp\XiaoQu_Cluster.csv', index=False)
        return len(list(set(XiaoQu['节点类别'])))

    def draw_cluster_xiaoqu_data(self):
        """在确定配送中心数量后为地图上色"""
        XiaoQu = pd.read_csv('Temp\XiaoQu_Cluster.csv')
        sortcenter = list(set(XiaoQu['节点类别']))
        color_dict = dict(zip(sortcenter, colors[:len(sortcenter)]))
        x_index = np.array(XiaoQu['横坐标'])
        y_index = np.array(XiaoQu['纵坐标'])
        for i in range(XiaoQu.shape[0]):
            x_i = x_index[i]
            y_i = y_index[i]
            color_i = XiaoQu.iloc[i, 6]
            color_i = color_dict[color_i]
            plt.scatter(x_i, y_i, color=color_i, s=2)
        plt.show()

    def draw_result(self):
        sort_intersection = pd.read_csv('Result\物资分发中心坐标.csv')
        intersection_route = pd.read_excel('Data\intersection_route.xlsx')
        intersection_node = pd.read_excel('Data\intersection_node.xlsx')
        community = pd.read_excel('Data\community.xlsx')
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
        for i in range(sort_intersection.shape[0]):
            xi = sort_intersection.iloc[i, 0]
            yi = sort_intersection.iloc[i, 1]
            plt.scatter(xi, yi, marker='*', color='red', s=100)
        for i in range(community.shape[0]):
            x_index = community.iloc[i, 4]
            y_index = community.iloc[i, 5]
            area = community.iloc[i, 7]
            plt.scatter(x_index, y_index, color=area_color_dict[area], s=2)

    def optimize(self):
        number_sort = 71
        sorting_index = 100 * \
            np.random.RandomState(152).random(size=(number_sort, 2))  # 分拣中心的坐标
        """计算路径并且确认"""
        chengben = list()
        for xun in range(20):
            xiaoqu = pd.read_csv('Temp\XiaoQu_Cluster.csv')
            p = xiaoqu['需求'].tolist()
            q = 0.2
            v = 100000 * 0.4
            xiaoqu_index = np.array(xiaoqu[['横坐标', '纵坐标']])
            distance = list()
            for i in range(sorting_index.shape[0]):
                sorting_i = np.array([sorting_index[i]])
                res = LA.norm(np.array(
                    [sorting_i[:, 0] - xiaoqu_index[:, 0], sorting_i[:, 1] - xiaoqu_index[:, 1]]), axis=0)
                distance.append(res.tolist())
            distance = np.array(distance)*0.32  # 比例尺问题
            model = gb.Model()
            w = model.addVars(number_sort, 1409, lb=0.0, ub=40000,
                              vtype=gb.GRB.CONTINUOUS, name='w')  # 添加变量
            model.setObjective(gb.quicksum(w[i, j] * distance[i, j] * q for i in range(number_sort)
                                           for j in range(1409)), gb.GRB.MINIMIZE)  # 添加目标函数
            model.addConstrs(w.sum('*', j) == p[j] for j in range(1409))
            model.addConstrs(w.sum(i, '*') <= v for i in range(number_sort))
            model.Params.LogToConsole = True  # 显示求解过程
            model.Params.TimeLimit = 100  # 限制求解时间为 100s
            model.optimize()
            cost = model.objVal
            """更新坐标"""
            for i in range(number_sort):
                sorting_index[i, 0] = (sum(w[i, j].X * q * xiaoqu_index[j, 0]/distance[i, j] for j in range(
                    1409))+0.0000001) / (sum(w[i, j].X * q / distance[i, j] for j in range(1409))+0.0000001) + 0.1*np.random.RandomState(152).random()
                sorting_index[i, 1] = (sum(w[i, j].X * q * xiaoqu_index[j, 1]/distance[i, j] for j in range(
                    1409))+0.0000001) / (sum(w[i, j].X * q / distance[i, j] for j in range(1409))+0.000001) + 0.1*np.random.RandomState(152).random()
            chengben.append(cost)
        print("总花费为", cost)
        all_renkou = pd.read_csv('Temp\XiaoQu_Cluster.csv')['小区人口数'].tolist()
        all_xuqiu = pd.read_csv('Temp\XiaoQu_Cluster.csv')['需求'].tolist()
        all_quyu = pd.read_csv('Temp\XiaoQu_Cluster.csv')['所属区域'].tolist()
        give_plan = list()
        row_index = list()
        column_index = list()
        distance_plan = list()
        Population = list()
        area = list()
        for i in range(number_sort):
            give = sum(w[i, j].X for j in range(1409))
            if give != 0:
                row_index.append(sorting_index[i, 0])
                column_index.append(sorting_index[i, 1])
                give_plan.append([w[i, j].X for j in range(1409)])
                distance_plan.append([distance[i, j] for j in range(1409)])
        give_plan = np.array(give_plan)
        distance_plan = np.array(distance_plan)
        xiaoqugeshu = list()
        xuanzhibanjing = list()
        for i in range(39):
            count = 0
            banjing = list()
            renkou = list()
            quyu = list()
            for j in range(1409):
                give = give_plan[i, j]
                if give != 0:
                    count += 1
                    banjing.append(distance_plan[i, j])
                    renkou.append(all_renkou[j]*give/all_xuqiu[j])
                    quyu.append(all_quyu[j])
            xiaoqugeshu.append(count)
            xuanzhibanjing.append(max(banjing))
            Population.append(int(sum(renkou)))
            area.append(max(set(quyu), key=quyu.count))
        result = pd.DataFrame({
            '横坐标': row_index,
            '纵坐标': column_index,
            '选址半径(km)': xuanzhibanjing,
            '辖区小区数': xiaoqugeshu,
            '辖区内人口数': Population,
            '所在区域': area
        })
        result.to_csv('Result/物资分发中心坐标.csv', index=False)
        result.to_excel('Result/物资分发中心坐标.xlsx', index=False)
