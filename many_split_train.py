# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 18-12-29
# file: test
# description:

from sklearn.model_selection import train_test_split
import os
from ECOCDemo.Common import Read_Write_tool
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ECOCDemo.Common.Read_Write_tool import write_FS_data


if __name__ == '__main__':

    fs_name = ['variance_threshold', 'linear_svc', 'tree', 'RandForReg']

    microarray_dataname = ['Breast', 'Cancers', 'DLBCL', 'GCM', 'Leukemia1', 'Leukemia2', 'Lung1', 'SRBCT']

    # UCI_dataname = ['cleveland', 'dermatology', 'led7digit' \
    #     , 'led24digit', 'letter', 'satimage', 'segment' \
    #     , 'vehicle', 'vowel', 'yeast']

    UCI_dataname = ['car','ecoli','flare','isolet','nursery','penbased','zoo']


    module_path = os.path.dirname(__file__)
    print(module_path)
    data_folder_path = module_path + '/UCI/max_min_FS_data/'

    selected_dataname = UCI_dataname
    selected_fs_name = fs_name

    for k in range(len(selected_fs_name)):

        # 创建结果文件夹
        res_folder_path = module_path + '/UCI/train_val_data_1/' + selected_fs_name[k] + '/'
        if not os.path.exists(res_folder_path):
            os.makedirs(res_folder_path)

        print(res_folder_path)

        for i in range(len(selected_dataname)):
            train_path = data_folder_path + selected_fs_name[k] + '/' + selected_dataname[i] + '_train.csv'
            test_path = data_folder_path + selected_fs_name[k] + '/' + selected_dataname[i] + '_test.csv'
            train_data, train_label = Read_Write_tool.read_Microarray_Dataset(train_path)
            test_data, test_label = Read_Write_tool.read_Microarray_Dataset(test_path)

            scale = MinMaxScaler().fit(train_data)
            train_data = scale.transform(train_data)
            test_data = scale.transform(test_data)

            ntrain_data, nval_data, ntrain_label, nval_label = train_test_split(train_data, train_label)

            write_FS_data(res_folder_path + selected_dataname[i] + '_train.csv', ntrain_data, ntrain_label)

            write_FS_data(res_folder_path + selected_dataname[i] + '_val.csv', nval_data,nval_label)

            write_FS_data(res_folder_path + selected_dataname[i] + '_test.csv', test_data,test_label)