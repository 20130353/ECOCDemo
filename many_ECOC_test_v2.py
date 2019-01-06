# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/2/28 10:05
# file: many_ECOC_test.py
# description:

import logging

import time
from ECOCDemo.Common.Evaluation_tool import Evaluation
from ECOCDemo.FS.DC_Feature_selection import *
from ECOCDemo.Common import Read_Write_tool
import os
from ECOCDemo.ECOC.Classifier import DC_ECOC, Self_Adaption_ECOC, OVO_ECOC, OVA_ECOC, Dense_random_ECOC, \
    Sparse_random_ECOC, D_ECOC


def ECOC_Process(dataname, train_data, train_label, test_data, test_label, val_data, val_label, ECOC_name,
                 matrix_folder_path, selected_fs_name):
    if ECOC_name.find('DC_ECOC') >= 0:
        dc_option = ECOC_name.split(' ')[1]
        matrix_path = matrix_folder_path + selected_fs_name + '/'
        base_M = get_base_M(matrix_path, dc_option, dataname)
        E = DC_ECOC(base_M=base_M, dc_option=dc_option)
        E.fit(train_data, train_label)

    elif ECOC_name.find('SAT_ECOC') >= 0:
        dc_option = ECOC_name[8:].strip()
        matrix_path = matrix_folder_path + selected_fs_name + '/'
        base_M = get_base_M(matrix_path, dc_option, dataname)
        E = Self_Adaption_ECOC(dc_option=dc_option, base_M=base_M, create_method='DC')
        E.fit(train_data, train_label, val_data, val_label)

    else:
        E = eval(ECOC_name + '()')
        E.fit(train_data, train_label)

    predicted_label = E.predict(test_data)
    evaluation_option = ['simple_acc', 'accuracy', 'sensitivity', 'specifity', 'precision', 'Fscore']
    Eva = Evaluation(test_label, predicted_label)
    res = Eva.evaluation(option=evaluation_option)

    # res['cls_acc'] = Eva.evaluate_classifier_accuracy(E.matrix, E.predicted_vector, test_label)
    res['diversity'] = Eva.evaluate_diversity(E.predicted_vector)

    return res


def get_base_M(path, string, dataname):
    options = string.split(' ')
    M = None
    all_options = ['F1', 'F2', 'F3', 'N2', 'N3', 'Cluster', 'DR', 'SR']
    for each in options:
        if each in all_options:
            filepath = path + str(each) + '_' + str(dataname) + '.xls'
            if M is None:
                M = Read_Write_tool.read_matirx(filepath)
            else:
                new_M = Read_Write_tool.read_matirx(filepath)
                M = np.hstack((M, new_M))
    return M


def fun(count):

    fs_name = ['variance_threshold', 'linear_svc', 'tree', 'RandForReg']

    microarray_dataname = ['Breast', 'Cancers', 'DLBCL', 'GCM', 'Leukemia1', 'Leukemia2', 'Lung1', 'SRBCT']

    UCI_dataname = ['cleveland', 'dermatology', 'led7digit' \
        , 'led24digit', 'letter', 'satimage', 'segment' \
        , 'vehicle', 'vowel', 'yeast']

    other_ECOC = ['OVA_ECOC', 'OVO_ECOC', 'Dense_random_ECOC', 'Sparse_random_ECOC' \
        , 'D_ECOC', 'DC_ECOC F1', 'DC_ECOC F2', 'DC_ECOC F3', 'DC_ECOC N2', 'DC_ECOC N3', 'DC_ECOC Cluster']

    module_path = os.path.dirname(__file__)
    data_folder_path = module_path + '/UCI/train_val_data/'
    matrix_folder_path = module_path + '/UCI/ECOC_matrix_data/train_val/'

    # 创建结果文件夹
    res_folder_path = module_path + '/UCI/UCI_res/train_val_data/SAT_ECOC/SVM/alalysing'+str(count)+'/'
    if not os.path.exists(res_folder_path):
        os.makedirs(res_folder_path)
    #
    selected_dataname = UCI_dataname[:3]
    ecoc_name = ['SAT_ECOC DR', 'SAT_ECOC SR']
    selected_ecoc_name = ['SAT_ECOC DR', 'SAT_ECOC SR']
    selected_fs_name = fs_name[:3]

    # selected_dataname = UCI_dataname
    # ecoc_name = other_ECOC
    # selected_ecoc_name = other_ECOC  # 为ECOC 算法添加evaluation 操作
    # selected_fs_name = fs_name[2:]

    for k in range(len(selected_fs_name)):

        LOG_FORMAT = "%(message)s"
        logging.basicConfig(filename=res_folder_path + selected_fs_name[k] + '_log.txt', level=logging.DEBUG,
                            format=LOG_FORMAT)

        logging.info('算法备注：')
        logging.info('连续三次没有变化或者变差的时候就停止继续生成新的列，把复杂的类和数量相近的类拼接起来形成列，最后形成的全部的矩阵送入剪枝')

        # save evaluation varibles
        data_acc, data_simacc, data_precision, data_specifity, data_sensitivity, data_cls_acc, data_Fscore = [], [], [], [], [], [], []

        for i in range(len(selected_dataname)):

            train_path = data_folder_path + selected_fs_name[k] + '/' + selected_dataname[i] + '_train.csv'
            test_path = data_folder_path + selected_fs_name[k] + '/' + selected_dataname[i] + '_test.csv'
            val_path = data_folder_path + selected_fs_name[k] + '/' + selected_dataname[i] + '_val.csv'

            # 一行是一个样本
            train_data, train_label = Read_Write_tool.read_Microarray_Dataset(train_path)
            test_data, test_label = Read_Write_tool.read_Microarray_Dataset(test_path)
            val_data, val_label = Read_Write_tool.read_Microarray_Dataset(val_path)

            acc, simacc, precision, specifity, sensitivity, cls_acc, Fscore = [], [], [], [], [], [], []

            for j in range(len(selected_ecoc_name)):

                logging.info('\n\nSVM classifier')
                logging.info('Time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                logging.info('FS: ' + selected_fs_name[k])
                logging.info('Dataset： ' + selected_dataname[i])
                logging.info('ECOC: ' + selected_ecoc_name[j])
                logging.info('Using KNN-Decoding')

                print('\n\nSVM classifier')
                print('Time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                print('FS: ' + selected_fs_name[k])
                print('Dataset： ' + selected_dataname[i])
                print('ECOC: ' + selected_ecoc_name[j])
                print('Using KNN-Decoding')

                res = ECOC_Process(selected_dataname[i], train_data, train_label, test_data, test_label, val_data,
                                   val_label,selected_ecoc_name[j], matrix_folder_path, selected_fs_name[k])

                if 'simple_acc' in res:
                    simacc.append(res['simple_acc'])
                if 'accuracy' in res:
                    acc.append(res['accuracy'])
                if 'sensitivity' in res:
                    sensitivity.append(res['sensitivity'])
                if 'specifity' in res:
                    specifity.append(res['specifity'])
                if 'precision' in res:
                    precision.append(res['precision'])
                if 'Fscore' in res:
                    Fscore.append(res['Fscore'])

                if 'diversity' in res:
                    txtname = res_folder_path + 'diversity_' + selected_fs_name[k] + '.txt'
                    content = 'Data: ' + selected_dataname[i] + '\t' + ' ECOC: ' + selected_ecoc_name[j] \
                              + '\t ' + str(res['diversity'])

                if 'cls_acc' in res:
                    txtname = res_folder_path + 'cls_acc_' + selected_fs_name[k] + '.txt'
                    content = 'Data: ' + selected_dataname[i] + '\t' + ' ECOC: ' + selected_ecoc_name[j] \
                              + '\t ' + str(res['cls_acc'])
                    Read_Write_tool.write_txt(txtname, content)

            data_simacc.append(simacc)
            data_acc.append(acc)
            data_precision.append(precision)
            data_specifity.append(specifity)
            data_Fscore.append(Fscore)
            data_sensitivity.append(sensitivity)

        row_name = copy.deepcopy(selected_dataname)
        row_name.append('Avg')
        if np.all(data_simacc):
            save_filepath = res_folder_path + 'simple_acc_' + selected_fs_name[k] + '.xls'
            data_simacc.append(np.mean(data_simacc, axis=0))
            Read_Write_tool.write_file(save_filepath, data_simacc, selected_ecoc_name, row_name)

        if np.all(data_acc):
            save_filepath = res_folder_path + 'accuracy_' + selected_fs_name[k] + '.xls'
            data_acc.append(np.mean(data_acc, axis=0))
            Read_Write_tool.write_file(save_filepath, data_acc, selected_ecoc_name, row_name)

        if np.all(data_sensitivity):
            save_filepath = res_folder_path + 'sensitivity_' + selected_fs_name[k] + '.xls'
            data_sensitivity.append(np.mean(data_sensitivity, axis=0))
            Read_Write_tool.write_file(save_filepath, data_sensitivity, selected_ecoc_name, row_name)

        if np.all(data_specifity):
            save_filepath = res_folder_path + 'specifity_' + selected_fs_name[k] + '.xls'
            data_specifity.append(np.mean(data_specifity, axis=0))
            Read_Write_tool.write_file(save_filepath, data_specifity, selected_ecoc_name, row_name)

        if np.all(data_precision):
            save_filepath = res_folder_path + 'precision_' + selected_fs_name[k] + '.xls'
            data_precision.append(np.mean(data_precision, axis=0))
            Read_Write_tool.write_file(save_filepath, data_precision, selected_ecoc_name, row_name)

        if np.all(data_Fscore):
            save_filepath = res_folder_path + 'Fscore_' + selected_fs_name[k] + '.xls'
            data_Fscore.append(np.mean(data_Fscore, axis=0))
            Read_Write_tool.write_file(save_filepath, data_Fscore, selected_ecoc_name, row_name)


if __name__ == '__main__':
    for each in range(25):
        fun(each)