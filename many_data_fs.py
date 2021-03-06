# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/3/20 17:11
# file: many_data_fs.py
# description:

import numpy as np
import logging
from sklearn.model_selection import train_test_split

from ECOCDemo.Common.Read_Write_tool import read_Microarray_Dataset
from ECOCDemo.Common.Read_Write_tool import read_UCI_Dataset
from ECOCDemo.Common.Read_Write_tool import write_FS_data
from ECOCDemo.FS.DC_Feature_selection import FS_selection
import os

from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # current_time log_level function_name user_print_info
    LOG_FORMAT = "%(message)s"

    # set log filepath, log level and info format
    logging.basicConfig(filename='FS_Process_UCI.txt', level=logging.DEBUG, format=LOG_FORMAT)

    microarray_dataname = ['Breast','Cancers','DLBCL','GCM','Leukemia1','Leukemia2'\
                ,'Lung1','SRBCT']

    # UCI_dataname = ['cleveland', 'dermatology', 'led7digit', 'led24digit', 'letter', 'satimage', 'segment',
    #                 'vehicle', 'vowel', 'yeast']

    UCI_dataname = ['car', 'ecoli', 'flare', 'isolet', 'nursery', 'penbased', 'zoo']

    fs_name = ['variance_threshold', 'linear_svc', 'tree', 'RandForReg']

    module_path = os.path.dirname(__file__)
    data_folder_path = module_path + '/UCI/treated_data/'
    res_folder_path =  module_path + '/UCI/max_min_FS_data_1/'

    selected_dataname = UCI_dataname
    selected_fsname = fs_name

    for i in range(len(selected_fsname)):

        fin_folder_path = res_folder_path + selected_fsname[i]

        if not os.path.exists(fin_folder_path):
            os.mkdir(fin_folder_path)

        logging.info("FS method:" + selected_fsname[i])
        for j in range(len(selected_dataname)):

            # train_path = data_folder_path + selected_dataname[j] + '_train.csv'
            # test_path = data_folder_path + selected_dataname[j] + '_test.csv'
            # train_data, train_label = read_UCI_Dataset(train_path)
            # test_data, test_label = read_UCI_Dataset(test_path)

            data_path = data_folder_path + selected_dataname[j] + '.csv'
            data,label = read_UCI_Dataset(data_path)
            train_data,test_data,train_label,test_label = train_test_split(data,label)

            scaler = MinMaxScaler().fit(train_data)
            train_data = scaler.transform(train_data)
            test_data = scaler.transform(test_data)

            train_data, train_label, test_data, test_label = \
                                FS_selection(train_data, train_label, test_data, test_label, selected_fsname[i])

            logging.info(selected_dataname[j] + "\t FS size: " + str(len(train_data[0])))

            train_file_path = fin_folder_path + '/' + selected_dataname[j] +'_train.csv'
            write_FS_data(train_file_path, train_data, train_label)


            test_file_path = fin_folder_path + '/' + selected_dataname[j] + '_test.csv'
            write_FS_data(test_file_path, test_data, test_label)

