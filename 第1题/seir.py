from scipy.integrate import odeint  # 导入 scipy.integrate 模块
import numpy as np
import pandas as pd


def dySEIR(y, t, lamda, theta, mu):  # SEIR 模型，导数函数
    s, e, i = y
    ds_dt = -lamda*i

    de_dt = lamda*i - theta*e
    di_dt = theta*e - mu*i
    return np.array([ds_dt, de_dt, di_dt])


class LambdaSolver:
    def __init__(self) -> None:
        self.actually_injected = pd.read_csv(
            'Data\Day_Population.csv')['合计'].to_list()
        # self.actually_injected = pd.read_excel(
        #     'Data\Day_Shanghai.xlsx')['合计'].to_list()

    def loss(self, lamda, pre_value, after_value):
        """计算当前lambda造成的损失值
        lamda:当前的lambda值,
        pre_value:前一天的确诊数,
        after_value:后一天的确诊数"""
        population = 9066906  # 总人数
        theta = 0.15  # 日发病率，每天密接转阳的比率
        mu = 0.75  # 日治愈率, 每天治愈的患病者人数占患病者总数的比例
        tEnd = 2  # 预测日期长度
        i0 = pre_value/population  # 患病者比例的初值
        e0 = i0*8  # 潜伏者比例的初值
        s0 = 1-i0-e0  # 易感者比例的初值
        Y0 = (s0, e0, i0)  # 微分方程组的初值
        # odeint 数值解，求解微分方程初值问题
        t = np.arange(0.0, tEnd, 1)  # (start,stop,step)
        ySEIR = odeint(dySEIR, Y0, t, args=(lamda, theta, mu))  # SEIR 模型
        pre_injected = ySEIR[:, 2] * population
        pre_injected = pre_injected[1] - pre_value
        ture_injected = after_value
        return (pre_injected-ture_injected)**2

    def found_best_lambda(self):
        """
        使用简单的for循环来优化每天的lambda
        """
        actually_injected = self.actually_injected
        lambda_result = list()
        for i in range(len(actually_injected) - 1):
            pre_value = actually_injected[i]  # 前一天的数据
            after_value = actually_injected[i+1]  # 后一天的数据
            loss_list = list()
            for j in range(100):
                loss_list.append(self.loss(j, pre_value, after_value))
            lambda_result.append(np.argmin(loss_list))
        return lambda_result
