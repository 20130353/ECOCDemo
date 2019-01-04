# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/8/30 22:06
# file: Ftest_main.py
# description:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import friedmanchisquare
import os

from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def draw_test_pic(avg_rank_data, CD, N, yticks):
    fig, ax = plt.subplots()
    L = list(range(1, len(avg_rank_data) + 1))
    plt.axis([0, 10, 0, len(avg_rank_data) + 1])

    for i in range(len(avg_rank_data)):
        ax.scatter(avg_rank_data[i], L[i], marker='o', c='blue', edgecolors='blue')
        ax.plot([avg_rank_data[i] - CD / 2, avg_rank_data[i] + CD / 2], [L[i], L[i]], '-', color='red', lw=2)

    xminorLocator = MultipleLocator(1)
    ymajorLocator = MultipleLocator(1)

    ax.xaxis.set_major_locator(xminorLocator)
    ax.yaxis.set_major_locator(ymajorLocator)

    plt.yticks(np.arange(len(yticks)), yticks)
    ax.grid()
    # plt.show()
    return fig


def avg_rank(*args):
    # Rank data
    print('source data is\n', np.vstack(args).T)
    rank_data = np.vstack(args) #[18*9]
    rank_data = rank_data.astype(float)
    for i in range(len(rank_data)):
        rank_data[i] = rankdata(rank_data[i])

    print('rank data is \n', rank_data)
    avg_data = np.mean(rank_data, axis=0)
    print('avg rank data is\n', avg_data)
    return avg_data


def get_and_save_stats_p(df,save_path):
    data_F1 = df.loc[:, 'F1']
    data_F2 = df.loc[:, 'F2']
    data_F3 = df.loc[:, 'F3']

    data_N2 = df.loc[:, 'N2']
    data_N3 = df.loc[:, 'N3']
    data_C1 = df.loc[:, 'C1']
    data_CM = df.loc[:, 'CM']

    data_OVO = df.loc[:, 'OVO']
    data_OVA = df.loc[:, 'OVA']
    data_Ordinal = df.loc[:, 'Ordinal']
    data_DECOC = df.loc[:, 'DECOC']
    data_ECOCONE = df.loc[:, 'ECOCONE']
    data_Forest = df.loc[:, 'Forest']

    stat_F, p_F = friedmanchisquare(data_F1, data_F2, data_F3, data_OVO, data_OVA, data_Ordinal, data_DECOC,
                                    data_ECOCONE, data_Forest)
    stat_N, p_N = friedmanchisquare(data_N2, data_N3, data_C1, data_CM, data_OVO, data_OVA, data_Ordinal, data_DECOC,
                                    data_ECOCONE, data_Forest)

    str_F = 'Equal'
    # set size is 9
    bound_F = 2.49
    if stat_F > bound_F:
        str_F = 'Different'

    str_N = 'Equal'
    bound_N = 2.45
    # set size is 10
    if stat_N > bound_N:
        str_N = 'Different'

    print('N:9 alpha:0.05 bound:%.3f F:%.3f P:%.3f res:%s' % (bound_F,stat_F, p_F, str_F))
    print('N:10 alpha:0.05 bound:%.3f F:%.3f P:%.3f res:%s' % (bound_N,stat_N, p_N, str_N))

    f = open(save_path + '_Ftest.txt','w')
    f.write('N:9 alpha:0.05 bound:%.3f F:%.3f P:%.3f res:%s' % (bound_F, stat_F, p_F, str_F))
    f.write('\n')
    f.write('N:10 alpha:0.05 bound:%.3f F:%.3f P:%.3f res:%s' % (bound_N, stat_N, p_N, str_N))

def get_and_save_Ntest_pic(df,save_path):
    K_F = 9
    N_F = 18
    alpha_9 = 3.102
    alpha_10 = 3.164

    CD_F = alpha_9 * np.sqrt(K_F * (K_F + 1) / (6 * N_F))

    K_N = 10
    N_N = 18
    CD_N = alpha_10 * np.sqrt(K_N * (K_N + 1) / (6 * N_N))

    print('CD_F:%0.3f CD_N:%.3f' % (CD_F, CD_N))

    # set size is 9
    ECOC_name_F = ['0', 'F1', 'F2', 'F3', 'OVO', 'OVA', 'Ordinal', 'DECOC', 'ECOCONE', 'Forest']
    # set size is 10
    ECOC_name_N = ['0', 'N2', 'N3', 'C1', 'CM', 'OVO', 'OVA', 'Ordinal', 'DECOC', 'ECOCONE', 'Forest']

    select_F = ['F1', 'F2', 'F3', 'OVO', 'OVA', 'Ordinal', 'DECOC', 'ECOCONE', 'Forest']
    avg_data_F = avg_rank(df.ix[:, select_F])
    pic1 = draw_test_pic(avg_data_F, CD_F, N_F, ['0'] + select_F)

    select_N = ['N2', 'N3', 'C1', 'CM', 'OVO', 'OVA', 'Ordinal', 'DECOC', 'ECOCONE', 'Forest']
    avg_data_F = avg_rank(df.ix[:, select_N])
    pic2 = draw_test_pic(avg_data_F, CD_F, N_F, ['0'] + select_N)

    # pic1.savefig(save_path + '_F.fig')
    pic1.savefig(save_path + '_F.tif',format='tif')
    # pic2.savefig(save_path + '_N.fig')
    pic2.savefig(save_path + '_N.tif',format='tif')

#
# if __name__ == '__main__':
#     # 计算单个feaure 子集的结果
#     path = 'E:/workspace-matlab/ECOC_data/data_trim/论文结果整理-期刊-pair/new_N3_F3/FTest/data/'
#     save_path = 'E:/workspace-matlab/ECOC_data/data_trim/论文结果整理-期刊-pair/new_N3_F3/FTest/'
#     head = [ 'F1', 'F2', 'F3','N2','N3','C1','CM','OVO', 'OVA', 'Ordinal', 'DECOC', 'ECOCONE', 'Forest']
#     for root, dirs, files in os.walk(path):
#         for file in files:
#             if os.path.splitext(file)[1] == '.xlsx':
#                 print(os.path.join(root, file))
#                 df = pd.read_excel(os.path.join(root, file),header=None).ix[:7,[2,3,4,5,6,7,8,9,11,12,14,15,16]]
#                 df.columns = head
#                 get_and_save_stats_p(df,root + os.path.splitext(file)[0])
#                 get_and_save_Ntest_pic(df, root + os.path.splitext(file)[0])

if __name__ == '__main__':
    # 计算单个feaure 子集的结果
    path = 'E:/workspace-matlab/ECOC_data/data_trim/论文结果整理-期刊-pair/new_N3_F3/FTest/many-fs-Ftest/SVM/'
    head = [ 'F1', 'F2', 'F3','N2','N3','C1','CM','OVO', 'OVA', 'Ordinal', 'DECOC', 'ECOCONE', 'Forest']
    whole_df = pd.DataFrame(columns=head)
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.xlsx':
                print(os.path.join(root, file))
                df = pd.read_excel(os.path.join(root, file),header=None).ix[[0,1,2,4,5,6],[2,3,4,5,6,7,8,9,11,12,14,15,16]]
                df.columns = head
                whole_df = pd.concat([whole_df,df],axis=0)

    get_and_save_stats_p(whole_df, root + os.path.splitext(file)[0])
    get_and_save_Ntest_pic(whole_df, root + os.path.splitext(file)[0])