# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/7/10 18:14
# file: create_experiment_data.py
# description:

import logging
import time
import copy
import numpy as np
import pandas as pd

from ECOCDemo.Common import Read_Write_tool
from ECOCDemo.ECOC.Classifier import ECOC_ONE, OVO_ECOC, OVA_ECOC, DC_ECOC, D_ECOC, Dense_random_ECOC, \
    Sparse_random_ECOC
from ECOCDemo.ECOC.Classifier import Self_Adaption_ECOC
from ECOCDemo.FS.DC_Feature_selection import *
from ECOCDemo.FS.DC_Feature_selection import DC_FS, select_data_by_feature_index
from ECOCDemo.Common.Evaluation_tool import Evaluation

fs_name = ['variance_threshold', 'linear_svc', 'tree', 'fclassif', 'RandForReg', 'linearsvc_tree']
microarray_dataname = ['Breast', 'Cancers', 'DLBCL', 'GCM', 'Leukemia1', 'Leukemia2' \
    , 'Lung1', 'SRBCT']

other_ECOC = ['OVA_ECOC', 'OVO_ECOC', 'Dense_random_ECOC', 'Sparse_random_ECOC', 'D_ECOC', 'DC_ECOC F1', 'DC_ECOC F2',
              'DC_ECOC F3', 'DC_ECOC N2', 'DC_ECOC N3', 'DC_ECOC Cluster']


def ECOC_Process(train_data, train_label, test_data, test_label, ECOC_name, **param):
    E = None
    if ECOC_name.find('DC_ECOC') >= 0:
        dc_option = ['F1', 'F2', 'F3', 'N2', 'N3', 'N4', 'L3', 'Cluster']
        for i, each in enumerate(dc_option):
            if each in ECOC_name:
                E = eval('DC_ECOC()')
                E.fit(train_data, train_label, dc_option=each)
                break

    elif ECOC_name.find('Self_Adaption_ECOC') >= 0:
        Ternary = ['+', '-', '*', '/', 'and', 'or', 'info', 'DC']
        for i, each in enumerate(Ternary):
            if each in ECOC_name:
                param['ternary_option'] = each
                if each == 'DC':
                    strs = ECOC_name.split('=')
                    param['dc_option'] = strs[1]

                E = eval('Self_Adaption_ECOC()')
                E.fit(train_data, train_label, **param)

    else:
        E = eval(ECOC_name + '()')
        E.fit(train_data, train_label)

    logging.info(ECOC_name + ' Matrix:\n' + str(E.matrix))
    predicted_label = E.predict(test_data)

    evaluation_option = ['simple_acc', 'accuracy', 'sensitivity', 'specifity', 'precision', 'Fscore']
    Eva = Evaluation(test_label, predicted_label)
    res = Eva.evaluation(option=evaluation_option)
    res['cls_acc'] = Eva.evaluate_classifier_accuracy(E.matrix, E.predicted_vector, test_label)
    res['diversity'] = Eva.evaluate_diversity(E.predicted_vector)
    return res


def get_base_M(path, ecoc, dataname):
    dc_option = ['F1', 'F2', 'F3', 'N2', 'N3', 'N4', 'L3', 'Cluster']

    M = None
    for each in dc_option:
        if ecoc.find(each) >= 0:
            filepath = path + str(each) + '_' + str(dataname) + '.xls'
            if M == None:
                M = [Read_Write_tool.read_matirx(filepath)]
            else:
                M.append(Read_Write_tool.read_matirx(filepath))
    return M


def merge_other_SA():
    import os
    module_path = os.path.dirname(__file__)
    res_folder_path = module_path + '/Microarray_res/SAT_DC/20181215/'

    selected_dataname = microarray_dataname
    selected_fs_name = fs_name[4:]

    path1 = module_path + '/Microarray_res/other_ECOC_backup - replaceDRSR10Mean/'
    m = ['accuracy', 'Fscore', 'precision', 'sensitivity', 'simple_acc', 'specifity']

    file_date = '2018-12-17=10-43-50'
    SAT_DC_path_SVM = res_folder_path + file_date + '/'

    for j in range(len(selected_fs_name)):

        if j != 0:
            continue

        for k in range(len(m)):
            fp1 = path1 + 'SVM/' + m[k] + '_' + selected_fs_name[j] + '.xls'
            fp2 = SAT_DC_path_SVM + m[k] + '_' + selected_fs_name[j] + '.xls'

            other_ECOC_df = pd.read_excel(fp1).ix[0:8,
                            ['OVA_ECOC', 'OVO_ECOC', 'Dense_random_ECOC', 'Sparse_random_ECOC', 'D_ECOC', 'DC_ECOC F1',
                             'DC_ECOC F2', 'DC_ECOC N3']]
            SAT_DC_df = pd.read_excel(fp2).ix[0:8, :]

            new_df = pd.concat([SAT_DC_df, other_ECOC_df], axis=1)
            for each in new_df.columns.values:
                mv = np.mean(new_df[each])
                new_df.at[8, each] = mv
            res = new_df.values
            save_filepath = SAT_DC_path_SVM + 'merge_' + m[k] + '_' + selected_fs_name[j] + '.xls'
            row_name = copy.deepcopy(selected_dataname)
            row_name.append('Avg')
            Read_Write_tool.write_file(save_filepath, res, selected_ecoc_name, row_name)


def merge_other_():
    import os
    module_path = os.path.dirname(__file__)
    res_folder_path = module_path + '/Microarray_res/other_ECOC_backup_new/Tree/'
    if not os.path.exists(res_folder_path):
        os.makedirs(res_folder_path)

    selected_fs_name = fs_name[2:5:2]
    selected_ecoc_name = other_ECOC
    selected_dataname = microarray_dataname

    path1 = module_path + '/Microarray_res/other_ECOC_backup - replaceDRSR10Mean/Tree/'
    m = ['accuracy', 'Fscore', 'precision', 'sensitivity', 'simple_acc', 'specifity']

    SAT_DC_path_SVM = module_path + '/Microarray_res/other_ECOC_backup - replace-add/Tree/'


    for j in range(len(selected_fs_name)):

        for k in range(len(m)):
            fp1 = path1 + m[k] + '_' + selected_fs_name[j] + '.xls'
            fp2 = SAT_DC_path_SVM + m[k] + '_' + selected_fs_name[j] + '.xls'

            other_ECOC_df1 = pd.read_excel(fp1).ix[0:8,
                             ['OVA_ECOC', 'OVO_ECOC', 'Dense_random_ECOC', 'Sparse_random_ECOC', 'D_ECOC', 'DC_ECOC F1',
                              'DC_ECOC F2']]
            other_ECOC_df2 = pd.read_excel(fp1).ix[0:8, ['DC_ECOC N3']]
            SAT_DC_df1 = pd.read_excel(fp2).ix[0:8, ['DC_ECOC F3', 'DC_ECOC N2']]
            SAT_DC_df2 = pd.read_excel(fp2).ix[0:8, ['DC_ECOC Cluster']]

            new_df = pd.concat([other_ECOC_df1, SAT_DC_df1, other_ECOC_df2, SAT_DC_df2], axis=1)
            for each in new_df.columns.values:
                mv = np.mean(new_df[each])
                new_df.at[8, each] = mv
            res = new_df.values
            row_name = copy.deepcopy(selected_dataname)
            row_name.append('Avg')
            save_filepath = res_folder_path + m[k] + '_' + selected_fs_name[j] + '.xls'
            Read_Write_tool.write_file(save_filepath, res, selected_ecoc_name, row_name)


def test_single_ECOC():
    ECOC_name = 'Self_Adaption_ECOC F1 F2 DC=F1'
    fs_name = 'RandForReg'
    data_name = 'Breast'
    matrix_folder_path = 'E:/workspace-python/ECOCDemo/Microarray_res/DC_matrix/'

    data_folder_path = 'E:/workspace-python/ECOCDemo/Microarray_data/FS_data/'
    train_path = data_folder_path + fs_name + '/' + data_name + '_train.csv'
    test_path = data_folder_path + fs_name + '/' + data_name + '_test.csv'
    train_data, train_label = Read_Write_tool.read_Microarray_Dataset(train_path)
    test_data, test_label = Read_Write_tool.read_Microarray_Dataset(test_path)

    print('Time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    print('FS: ' + fs_name)
    print('Dataset： ' + data_name)
    print('ECOC: ' + ECOC_name)

    param = {}
    if ECOC_name.find('Self_Adaption_ECOC') >= 0:
        final_folder_matrix_path = matrix_folder_path + fs_name + '/'
        param['base_M'] = get_base_M(final_folder_matrix_path, ECOC_name, data_name)
        print('base_M:\n')
        for i in range(2):
            print(param['base_M'][i])

    res = ECOC_Process(train_data, train_label, test_data, test_label, ECOC_name, **param)

    res_folder_path = 'E:/workspace-python/ECOCDemo/Microarray_res/merge_result/具体过程/'
    txtname = res_folder_path + 'diversity_' + fs_name + '.txt'
    content = 'Data: ' + data_name + '\t' + ' ECOC: ' + ECOC_name \
              + '\t ' + str(res['diversity'])
    Read_Write_tool.write_txt(txtname, content)

    txtname = res_folder_path + 'cls_acc_' + fs_name + '.txt'
    content = 'Data: ' + data_name + '\t' + ' ECOC: ' + ECOC_name \
              + '\t ' + str(res['cls_acc'])
    Read_Write_tool.write_txt(txtname, content)


if __name__ == '__main__':
    # replace_other_DR_SR()
    # merge_other_SA()
    merge_other_()
    # test_single_ECOC()
