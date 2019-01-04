# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/8/30 16:20
# file: Friedman_testChart.py
# description: draw Friedman chart

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import friedmanchisquare
import sys,os
from  collections import Counter


from pylab import *
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

def draw_test_pic(avg_rank_data, CD,N,yticks,save_path,alpha):
    matplotlib.rcParams.update({'font.size': 8})
    fig, ax = plt.subplots()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(3.3, 3)
    L = list(range(1,len(avg_rank_data)+1))
    plt.axis([0,N+1, 0, len(avg_rank_data)+1])
    for i in range(len(avg_rank_data)):
        if yticks[i] in ['F1','F2','F3','N2','N3','C1','CM','F','N','MDC','high']:
            color = 'red'
        else:
            color = 'blue'
        ax.scatter(avg_rank_data[i], L[i]-0.85, marker='o', c=color, edgecolors=color)
        ax.plot([(avg_rank_data[i] - CD/2),(avg_rank_data[i] + CD/2)], [L[i]-0.85,L[i]-0.85], '-',color=color,lw=2)

        # ax.plot([avg_rank_data[i] - CD/2,avg_rank_data[i] - CD/2], [L[i]-0.865,L[i]-0.835], '-',color=color,lw=2)
        # ax.plot([avg_rank_data[i] + CD/2,avg_rank_data[i] + CD/2], [L[i]-0.865,L[i]-0.835], '-',color=color,lw=2)
    if 'high' in yticks:
        yticks[yticks.index('high')]='ECOCMDC'
    if 'ECOCONE' in yticks:
        yticks[yticks.index('ECOCONE')] = 'ECOCONE'
    if 'MDC' in yticks:
        yticks[yticks.index('MDC')] = 'MDC_AVG'

    plt.xlabel('Mean-Rank',fontsize=9)
    # plt.ylabel('ECOC-Methods')
    plt.xlim(xmin=1)
    plt.xlim(xmax=7)
    plt.xticks([1,2,3,4,5,6,7],['1','2','3','4','5','6','7'])

    # plt.xlim(xmin=3)
    # plt.xlim(xmax=13)
    # plt.xticks([3, 4, 5, 6, 7,8,9,10,11,12,13], ['3', '4', '5', '7', '8', '9', '10','11','12','13'])

    for inx,each in enumerate(yticks):
        yticks[inx] = '  '+each

    plt.yticks(np.arange(len(yticks)),yticks,fontsize=7)

    # ax.yaxis.set_label_position("right")
    # ax.xaxis.set_label_coords(1, -0.05)

    # plt.text(2, 14.5,  'alpha=' + str(round(alpha,2)) + '    CD=' + str(round(CD,2)),fontsize=9)
    ax.grid()
    plt.savefig(save_path)
    plt.show()

def avg_rank(*args):
    # Rank data
    print('source data is\n',np.vstack(args).T)
    rank_data = np.vstack(args)
    rank_data = rank_data.astype(float)
    for i in range(len(rank_data)):

        data = [1-each for each in rank_data[i]]
        # rank_data[i] = rankdata(rank_data[i])
        rank_data[i] = rankdata(data)

    print('rank data is \n',rank_data)
    avg_data = np.mean(rank_data,axis=0)
    print('avg rank data is\n',avg_data)
    return avg_data

def F_test(Tdf,res_path):
    data_F1 = Tdf.loc[:, 'F1']
    data_F2 = Tdf.loc[:, 'F2']
    data_F3 = Tdf.loc[:, 'F3']

    data_N2 = Tdf.loc[:, 'N2']
    data_N3 = Tdf.loc[:, 'N3']
    data_C1 = Tdf.loc[:, 'C1']
    data_CM = Tdf.loc[:, 'CM']

    data_OVO = Tdf.loc[:, 'OVO']
    data_OVA = Tdf.loc[:, 'OVA']
    data_Ordinal = Tdf.loc[:, 'Ordinal']
    data_DECOC = Tdf.loc[:, 'DECOC']
    data_ECOCONE = Tdf.loc[:, 'ECOCONE']
    data_Forest = Tdf.loc[:, 'Forest']

    data_high = Tdf.loc[:,'high']

    stat_F, p_F = friedmanchisquare(data_F1,data_F2,data_F3,data_N2,data_N3,data_C1,data_CM,data_OVO, data_OVA, data_Ordinal, data_DECOC,
                                    data_ECOCONE, data_Forest)
    # stat_N, p_N = friedmanchisquare(data_high,data_OVO, data_OVA, data_Ordinal, data_DECOC,
    #                                 data_ECOCONE, data_Forest)

    str_F = 'Equal'
    if stat_F > 1.82:
        str_F = 'Different'

    print('N:13 alpha:0.05 bound:1.82 F:%.3f P:%.3f res:%s' % (stat_F, p_F, str_F))
    # print('N:10 alpha:0.05 bound:4.06 F:%.3f P:%.3f res:%s' % (stat_N, p_N, str_N))

    filename = res_path + '/' + learner + '.txt'
    f = open(filename, 'a')
    f.write('N:13 alpha:0.05 bound:1.82 F:%.3f P:%.3f res:%s' % (stat_F, p_F, str_F))
    # f.write('N:10 alpha:0.05 bound:4.06 F:%.3f P:%.3f res:%s' % (stat_N, p_N, str_N))
    f.close()


if __name__ == '__main__':
    path = 'E:/workspace-matlab/data/data_fs_10--150_pair_newN3-F3/'
    FS = ['roc', 'ttest', 'wilcoxon']
    Learners = ['svm']
    columns_name = ['F1', 'F2', 'F3', 'N2', 'N3', 'N4', 'L3', 'C1' \
        , 'CM', 'OVO', 'DR', 'OVA', 'Ordinal', 'SR', 'DECOC', 'ECOCONE', 'Forest']
    res_path = 'E:/workspace-matlab/data/data_trim/论文结果整理-期刊-pair/new_N3_F3/FTest/single-fs-fssize-Ftest/'

    ECOC_FN = ['F','N','OVO', 'OVA', 'Ordinal', 'DECOC', 'ECOCONE', 'Forest']
    ECOC_ALL = ['F1', 'F2', 'F3', 'N2', 'N3', 'C1', 'CM', 'OVO', 'OVA', 'Ordinal', 'DECOC', 'ECOCONE',
                'Forest']
    ECOC_MDC = ['MDC', 'OVO', 'OVA', 'Ordinal', 'DECOC', 'ECOCONE','Forest']
    ECOC_high = ['high', 'OVO', 'OVA', 'Ordinal', 'DECOC', 'ECOCONE', 'Forest']

    FN_dict_100 = {'name':'100-150-FN','K':8,'range':range(100,155,5),'N': 66,'alpha':3.031, \
                   'ECOC':ECOC_FN}

    FN_dict_5 = {'name': '5-150-FN', 'K': 8, 'range': range(5, 155, 5), 'N': 180, 'alpha': 3.031, \
                 'ECOC': ECOC_FN}

    all_dict_100 = {'name': '100-150-All', 'K': 13, 'range': range(100, 155, 5), 'N': 66, 'alpha': 3.313, \
                    'ECOC': ECOC_ALL}

    all_dict_5 = {'name': '5-150-All', 'K': 13, 'range': range(5, 155, 5), 'N': 180, 'alpha': 3.313, \
                  'ECOC': ECOC_ALL}

    MDC_dict_100 = {'name': '100-150-MDC', 'K': 7, 'range': range(100, 155, 5), 'N': 66, 'alpha': 2.949, \
                 'ECOC': ECOC_MDC}

    MDC_dict_5 = {'name': '5-150-MDC', 'K': 7, 'range': range(5, 155, 5), 'N': 180, 'alpha': 2.949, \
                  'ECOC': ECOC_MDC}

    high_dict_100 = {'name': '100-150-high', 'K': 7, 'range': range(100, 155, 5), 'N': 66, 'alpha': 2.949, \
                    'ECOC': ECOC_high}

    high_dict_5 = {'name': '5-150-high', 'K': 7, 'range': range(5, 155, 5), 'N': 180, 'alpha': 2.949, \
                  'ECOC': ECOC_high}

    for each_param in [MDC_dict_100]:
        for fs in FS:
            for learner in Learners:
                Tdf = None
                for i in each_param['range']:
                    filepath = path + '/data_fs_' + str(i) + '/' + fs + '/learner_res/' + learner + '_accuracy.csv'
                    df = pd.read_csv(filepath,header=None)
                    if Tdf is None:
                        Tdf = df
                    else:
                        Tdf = pd.concat([Tdf, df], axis=0)

                Tdf.columns = columns_name
                CD = each_param['alpha'] * np.sqrt(each_param['K'] * (each_param['K'] + 1) / (6 * each_param['N']))

                Tdf['F'] = Tdf[['F1', 'F2', 'F3']].mean(1)
                Tdf['N'] = Tdf[['N2', 'N3','C1']].mean(1)
                Tdf['MDC'] = Tdf[['F1','F2','F3','N2', 'N3','C1','CM']].mean(1)

                res =[]
                for inx,row in Tdf[['F1','F2','F3','N2', 'N3','C1','CM']].iterrows():
                    res.append(max(row))
                Tdf['high'] = res

                F_test(Tdf, res_path)

                ECOC = each_param['ECOC']
                avg_data = avg_rank(Tdf[ECOC])

                ECOC_dict = {}
                for inx,each in enumerate(ECOC):
                    ECOC_dict[each] =  avg_data[inx]

                new_ECOC_dict = {}
                for i in range(len(ECOC_dict)):
                    mv = max(ECOC_dict.items(), key=lambda x: x[1])
                    ECOC_dict.pop(mv[0])
                    new_ECOC_dict[mv[0]] = mv[1]

                pic_save_path = res_path + '/'+ each_param['name']+ '_' + fs + '_'+learner +'.tif'
                draw_test_pic(list(new_ECOC_dict.values()), CD, each_param['K'], list(new_ECOC_dict.keys()),pic_save_path,each_param['alpha'])
