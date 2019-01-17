# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 19-1-17
# file: test2
# description:

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
                 'transfusion.csv','thyroid.csv','texture.csv',
                 'sensor_readings_24.csv','sat.csv','processed.cleveland.csv','penbased.csv',
                 'page-blocks.csv','ionosphere.csv','fertility_Diagnosis.csv','contraceptive.csv',
                 'contraceptive-method-choice.csv','column_2C.csv','column_3C.csv','Cardiotocography.csv',
                 'breast-cancer-wisconsin.csv','avila-ts.csv','adult.csv','abalone.csv']

    fs_name = ['variance_threshold', 'linear_svc', 'tree', 'RandForReg']

    module_path = os.path.dirname(__file__)
    data_folder_path = '/home/smx/Desktop/unbalance_data/2_continuous/'
    path1 = '/home/smx/Desktop/unbalance_data/3_max_min_FS_data/'
    path2 = '/home/smx/Desktop/unbalance_data/4_train_val_data/'

    selected_dataname = datanames
    selected_fsname = fs_name[1:]

    data_txt_path = path1 + 'data_log.txt'

    for i in range(len(selected_fsname)):

        # 创建结果文件夹
        res_folder_path = path1 + selected_fsname[i] + '/'
        if not os.path.exists(res_folder_path):
            os.makedirs(res_folder_path)

        feature_txt_path = path1 + selected_fsname[i] + '_data_log.txt'

        # 创建结果文件夹
        res_folder_path1 = path2 + selected_fsname[i] + '/'
        if not os.path.exists(res_folder_path1):
            os.makedirs(res_folder_path1)

        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        write_txt(data_txt_path, 'create time is %s' % time_str)
        write_txt(feature_txt_path, 'create time is %s' % time_str)

        for j in range(len(selected_dataname)):

            path = data_folder_path + selected_dataname[j]

            source_df = pd.read_csv(path,header=None)
            drop_df = source_df.dropna()
            feature = drop_df.iloc[:, :-1].values
            label = drop_df.iloc[:, -1].values

            write_txt(data_txt_path,'\n\n*-------------- ' + selected_dataname[j] +'-------------------*')
            write_txt(data_txt_path, 'data len is :' + str(drop_df.size))
            write_txt(data_txt_path, 'label counter is :' + str(Counter(label)))

            # feature = pd.get_dummies(df)

            train_data, test_data, train_label, test_label = train_test_split(feature, label,test_size=0.4)

            scaler = MinMaxScaler().fit(train_data)
            train_data = scaler.transform(train_data)
            test_data = scaler.transform(test_data)

            train_data, train_label, test_data, test_label = \
                FS_selection(train_data, train_label, test_data, test_label, selected_fsname[i])

            train_file_path = res_folder_path + '/' + selected_dataname[j] + '_train.csv'
            write_FS_data(train_file_path, train_data, train_label)

            test_file_path = res_folder_path + '/' + selected_dataname[j] + '_test.csv'
            write_FS_data(test_file_path, test_data, test_label)

            write_txt(feature_txt_path, '\n\n*-------------- ' + selected_dataname[j] + '-------------------*')
            write_txt(feature_txt_path, 'feature len\n' + str(len(train_data[0])))
            write_txt(feature_txt_path, 'train sample len \n' + str(len(train_data)))
            write_txt(feature_txt_path, 'test sample len \n' + str(len(test_data)))

            # =============    保存 train test val                 ===============
            test_data, val_data, test_label, val_label = train_test_split(test_data, test_label, test_size=0.5)

            train_file_path = res_folder_path1 + '/' + selected_dataname[j] + '_train.csv'
            write_FS_data(train_file_path, train_data, train_label)

            test_file_path = res_folder_path1 + '/' + selected_dataname[j] + '_test.csv'
            write_FS_data(test_file_path, test_data, test_label)

            val_file_path = res_folder_path1 + '/' + selected_dataname[j] + '_val.csv'
            write_FS_data(val_file_path, val_data, val_label)

            write_txt(feature_txt_path, '*-------------- ' + selected_dataname[j] + '-------------------*')
            write_txt(feature_txt_path, 'feature len\n' + str(len(train_data[0])))
            write_txt(feature_txt_path, 'train sample len \n' + str(len(train_data)))
            write_txt(feature_txt_path, 'test sample len \n' + str(len(test_data)))
            write_txt(feature_txt_path, 'val sample len \n' + str(len(val_data)))