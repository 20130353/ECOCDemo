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
from ECOCDemo.Common.Read_Write_tool import write_matrix

#
# def ECOC_Process(train_data, train_label, ECOC_name):
#     dc_option = ECOC_name.split(' ')[1]
#     E = DC_ECOC(dc_option=dc_option)
#     M, index = E.create_matrix(train_data, train_label, dc_option=dc_option)
#     return M
def ECOC_Process(train_data, train_label, ECOC_name):
    E = eval(ECOC_name + '()')
    M, index = E.create_matrix(train_data, train_label)
    return M



if __name__ == '__main__':

    LOG_FORMAT = "%(message)s"

    module_path = os.path.dirname(__file__)

    microarray_dataname = ['Breast', 'Cancers', 'DLBCL', 'GCM', 'Leukemia1', 'Leukemia2' \
        , 'Lung1', 'SRBCT']

    # UCI_dataname = ['cleveland', 'dermatology', 'led7digit', 'led24digit', 'letter', 'satimage', 'segment',
    #                 'vehicle', 'vowel', 'yeast']


    UCI_dataname = ['car', 'ecoli', 'flare', 'isolet', 'nursery', 'penbased', 'zoo']
    unbalance_data = ['wine', 'winequality-red', 'winequality-white', 'poker-hand-training-true', 'haberman', 'thyroid',
                      'sensor_readings_24', 'sat', 'page-blocks', 'ionosphere', 'fertility_Diagnosis', 'contraceptive',
                      'column_2C', 'column_3C', 'Cardiotocography', 'avila-ts', 'abalone']

    fs_name = ['variance_threshold', 'linear_svc', 'tree', 'RandForReg']
    # ecoc_name = ['DC_ECOC F1', 'DC_ECOC F2', 'DC_ECOC F3', 'DC_ECOC N2', 'DC_ECOC N3', 'DC_ECOC Cluster']
    ecoc_name = ['Dense_random_ECOC', 'Sparse_random_ECOC']
    name = ['DR', 'SR']

    folder_path = module_path + '/UCI/4_train_val_data/'
    save_folder_path = module_path + '/UCI/ECOC_matrix_data/train_val/'
    selected_dataname = UCI_dataname
    selected_ecoc_name = ecoc_name
    selected_fs_name = fs_name

    for k in range(len(selected_fs_name)):
        fin_folder_path = folder_path + selected_fs_name[k]
        fin_save_folder_path = save_folder_path + selected_fs_name[k]
        if not os.path.exists(fin_save_folder_path):
            os.mkdir(fin_save_folder_path)

        # set log filepath, log level and info format
        logging.basicConfig(filename=fin_save_folder_path + '/DC_matrix.txt', level=logging.DEBUG,
                            format=LOG_FORMAT)

        for i in range(len(selected_dataname)):

            logging.info('Time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
            logging.info('Dataset： ' + selected_dataname[i])
            train_path = fin_folder_path + '/' + selected_dataname[i] + '_train.csv'
            train_data, train_label = read_Microarray_Dataset(train_path)

            for j in range(len(selected_ecoc_name)):
                logging.info('Random: ' + selected_ecoc_name[j])
                Matrix = ECOC_Process(train_data, train_label, selected_ecoc_name[j])
                dc = name[j]
                save_filepath = fin_save_folder_path + '/' + dc + '_' + selected_dataname[i] + '.xls'
                write_matrix(save_filepath, Matrix)



