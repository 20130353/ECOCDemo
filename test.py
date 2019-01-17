# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 19-1-17
# file: test
# description:

# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/3/1 15:03
# file: many_DC_matrix.py
# description: save many DC matrix of many datasets

import numpy as np
import copy
import types
import time
import os
import logging

from ECOCDemo.Common.Evaluation_tool import Evaluation
from ECOCDemo.ECOC.Classifier import ECOC_ONE,OVO_ECOC,OVA_ECOC,DC_ECOC,D_ECOC,Dense_random_ECOC,Sparse_random_ECOC,Self_Adaption_ECOC
from ECOCDemo.FS.DC_Feature_selection import DC_FS, select_data_by_feature_index
from ECOCDemo.Common.Read_Write_tool import read_Microarray_Dataset
from ECOCDemo.Common.Read_Write_tool import write_matrix,read_Microarray_Dataset,write_FS_data



if __name__ == '__main__':

    module_path = os.path.dirname(__file__)



    unbalance_data = ['wine', 'winequality-red', 'winequality-white', 'poker-hand-training-true','thyroid',
                      'sensor_readings_24', 'sat', 'page-blocks','contraceptive',
                      'column_3C', 'Cardiotocography', 'avila-ts', 'abalone']

    fs_name = ['variance_threshold', 'linear_svc', 'tree', 'RandForReg']

    folder_path = module_path + '/UCI/4_train_val_data/'
    save_folder_path = module_path + '/UCI/4_train_val_data/'

    selected_dataname = unbalance_data
    selected_fs_name = fs_name[2:]

    for k in range(len(selected_fs_name)):

        for i in range(len(selected_dataname)):

            path = folder_path + selected_fs_name[k]

            train_path = path + '/' + selected_dataname[i] + '.csv_train.csv'
            test_path = path + '/' + selected_dataname[i] + '.csv_test.csv'
            val_path = path + '/' + selected_dataname[i] + '.csv_val.csv'

            train_data, train_label = read_Microarray_Dataset(train_path)
            test_data, test_label = read_Microarray_Dataset(test_path)
            val_data, val_label = read_Microarray_Dataset(val_path)

            print('Dataset： ' + selected_dataname[i])

            write_FS_data(path + '/' + selected_dataname[i] + '_train.csv', train_data, train_label)
            write_FS_data(path + '/' + selected_dataname[i] + '_test.csv', test_data, test_label)
            write_FS_data(path + '/' + selected_dataname[i] + '_val.csv', val_data, val_label)

            os.remove(path + '/' + selected_dataname[i] + '.csv_train.csv')
            os.remove(path + '/' + selected_dataname[i] + '.csv_test.csv')
            os.remove(path + '/' + selected_dataname[i] + '.csv_val.csv')

