# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 19-1-17
# file: test1
# description:


import numpy as np
import logging
from sklearn.model_selection import train_test_split

from ECOCDemo.Common.Read_Write_tool import read_Microarray_Dataset
from ECOCDemo.Common.Read_Write_tool import read_UCI_Dataset
from ECOCDemo.Common.Read_Write_tool import write_FS_data
from ECOCDemo.FS.DC_Feature_selection import FS_selection
import os


import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import time

from collections import Counter
from ECOCDemo.Common.Read_Write_tool import write_txt
from sklearn.preprocessing import LabelEncoder

def read_data(path):

    with open(path) as f:
        content = f.readlines()

    sample = []
    label = []
    for each in content:
        try:
            t = list(map(float,re.split(r' +',each.strip())))
        except ValueError:
            t = re.split(r' +',each.strip())

        sample.append(t[:-1])
        label.append(t[-1])

    return sample,label


if __name__ == '__main__':

    LOG_FORMAT = "%(message)s"

    logging.basicConfig(filename='FS_Process_UCI.txt', level=logging.DEBUG, format=LOG_FORMAT)

    datanames = ['wine.csv', 'winequality-red.csv', 'winequality-white.csv',
                 'poker-hand-training-true.csv', 'poker-hand-testing.csv','haberman.csv',
                 'wifi_localization.csv','transfusion.csv','thyroid.csv','texture.csv',
                 'sensor_readings_24.csv','sat.csv','processed.cleveland.csv','penbased.csv',
                 'page-blocks.csv','ionosphere.csv','fertility_Diagnosis.csv','contraceptive.csv',
                 'contraceptive-method-choice.csv','column_2C.csv','column_3C.csv','Cardiotocography.csv',
                 'breast-cancer-wisconsin.csv','avila-ts.csv','adult.csv','abalone.csv']

    datanames = ['wifi_localization.csv','abalone.csv']


    fs_name = ['variance_threshold', 'linear_svc', 'tree', 'RandForReg']

    module_path = os.path.dirname(__file__)
    data_folder_path = '/home/smx/Desktop/unbalance_data/1_dis_conti/'
    res_folder_path = '/home/smx/Desktop/unbalance_data/2_continuous/'

    selected_dataname = datanames
    selected_fsname = fs_name


    for i in range(len(selected_fsname)):

        for j in range(len(selected_dataname)):

            path = data_folder_path + selected_dataname[j]

            source_df = pd.read_csv(path,header=None)
            drop_df = source_df.dropna()

            col = [0]

            new_df = None
            for inx in col:

                df = pd.DataFrame(LabelEncoder().fit_transform(drop_df.iloc[:,inx].values))
                if new_df is None:
                    new_df = df
                else:
                    new_df = pd.concat((new_df,df),axis=1)

            drop_df.drop(drop_df.columns[col], axis=1, inplace=True)

            res_df = pd.concat((new_df,drop_df),axis=1)

            res_df.to_csv(res_folder_path + selected_dataname[j],header=None,index=None)



