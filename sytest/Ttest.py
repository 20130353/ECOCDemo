# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/8/29 12:37
# file: Ttest.py
# description: T test

## Import the packages
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from scipy import stats
import copy

file_path = 'E:/写作论文/写作论文/PR期刊/data/'
columns = ['F1', 'F2', 'F3', 'N2', 'N3', 'C1', 'CM', 'OVO', 'OVA', 'Ordinal', 'DECOC', 'ECOCONE', 'Forest']
data_name = 'SVM_Ttest'

data = pd.read_excel(file_path + data_name + '.xls')
tdata = pd.DataFrame(data=np.zeros([data.shape[0],data.shape[0]]))

pdata = copy.deepcopy(tdata)
for i in range(0,data.shape[0]):
    for j in range(0,data.shape[0]):
        a = data.ix[i]
        b = data.ix[j]
        print('inx:',i,j)
        vatt,varp = stats.levene(a, b)
        is_equal = False
        if(varp > 0.05):
            print('var equal')
            is_equal = True
        else:
            print('var not equal')
        t,p = ttest_ind(a,b,equal_var=is_equal)
        tdata.loc[i,j] = round(t,3)
        pdata.loc[i,j] = round(p,3)

pdata.columns = columns
tdata.columns = columns

tdata.to_csv(file_path+ data_name + '_TT.csv')
pdata.to_csv(file_path+ data_name + '_TP.csv')


# 原假设是：两总体均值相等
