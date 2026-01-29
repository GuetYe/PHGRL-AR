#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
@Auther:Evans_WP
@Contact:WenPeng19943251019@163.com
@Time:2025/8/15 11:18
@File:smooth_plot_seaborn.py
@Desc:****************
"""
import seaborn
from utils import smooth
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import t

if __name__ == '__main__':
    data_c = pd.read_csv('../matlab_plot/plot_data/SAC_8000_eva_value_data_2025-08-13-20-54-36.csv')
    control_tag = [str(c)[0] for c in data_c['control_tag_list']]
    idx_C = [i for i, tag in enumerate(control_tag) if tag == 'C']
    sample_number = 1500
    window_size = 100
    start_index = 500
    data_index = np.arange(start_index + 1, start_index + sample_number + 1)
    # SAC C类
    ycf = np.array(data_c.loc[idx_C, 'eva_value_list'])
    # 注意：加1100是你自己数据的规律，这里保留
    yc = ycf[data_index + 1100].tolist()
    mov_yc = smooth(yc, window_size)
    xc = np.arange(1, sample_number + 1)
    print(xc)

    plt.figure(figsize=(14,7))
    seaborn.set(style="darkgrid", font_scale=1.5)

    seaborn.lineplot(x=xc,y=yc)

    # # 原始曲线
    # plt.plot(xc, yc, label="SAC-C 原始", color='royalblue')
    # plt.plot(xc, mov_yc, label="SAC-C 滑动平均", color='navy', linestyle='--')
    #
    # # 多项式拟合+置信带
    # degree = 3
    # coefs = np.polyfit(xc, yc, degree)
    # poly = np.poly1d(coefs)
    # yfit = poly(xc)
    # # 置信带估算
    # residuals = yc - yfit
    # dof = len(xc) - (degree + 1)
    # resid_std = np.std(residuals)
    # tval = t.ppf(0.975, dof)
    # conf_interval = tval * resid_std * np.sqrt(1 + (xc - np.mean(xc)) ** 2 / np.sum((xc - np.mean(xc)) ** 2))
    # # 缩放倍数与matlab一致
    # plt.fill_between(xc, yfit - conf_interval, yfit + conf_interval, color=[0.8, 0.8, 1], alpha=0.3,
    #                  label="SAC-C 多项式区间")

    plt.show()


