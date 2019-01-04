# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 2018/2/28 10:05
# file: many_ECOC_test.py
# description:

import time
from ECOCDemo.Common.Evaluation_tool import Evaluation
from ECOCDemo.FS.DC_Feature_selection import *
from ECOCDemo.Common import Read_Write_tool
import os
from ECOCDemo.ECOC.Classifier import DC_ECOC, Self_Adaption_ECOC,OVO_ECOC,OVA_ECOC,Dense_random_ECOC,Sparse_random_ECOC,D_ECOC


def ECOC_Process(dataname, train_data, train_label, test_data, test_label, ECOC_name, matrix_folder_path):
    if ECOC_name.find('DC_ECOC') >= 0:
        dc_option = ECOC_name.split(' ')[1]
        matrix_path = matrix_folder_path + selected_fs_name[k] + '/'
        base_M = get_base_M(matrix_path, dc_option, dataname)
        E = DC_ECOC(base_M=base_M, dc_option=dc_option)
        E.fit(train_data, train_label)

    elif ECOC_name.find('SAT_ECOC') >= 0:
        dc_options = ECOC_name[8:]
        matrix_path = matrix_folder_path + selected_fs_name[k] + '/'
        base_M = get_base_M(matrix_path, dc_options, dataname)
        evaluation_option = ECOC_name.split('evaluation=')[1]
        E = Self_Adaption_ECOC(base_M=base_M, create_method='DC', evaluation_option=evaluation_option)
        E.fit(train_data, train_label)

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
    all_options = ['F1', 'F2', 'F3', 'N2', 'N3', 'Cluster']
    for each in options:
        if each in all_options:
            filepath = path + str(each) + '_' + str(dataname) + '.xls'
            if M is None:
                M = Read_Write_tool.read_matirx(filepath)
            else:
                new_M = Read_Write_tool.read_matirx(filepath)
                M = np.hstack((M, new_M))
    return M


def get_conbination(ecoc):
    res = []
    for i in range(len(ecoc)):
        for j in range(i + 1, len(ecoc)):
            for k in range(j + 1, len(ecoc)):
                str = 'SAT_ECOC ' + ecoc[i] + ' ' + ecoc[j] + ' ' + ecoc[k]
                res.append(str)
    return res


def get_Ternary(ecoc, operations):
    new_ecoc = []
    for op in operations:
        for e in ecoc:
            new_e = e + ' evaluation=' + op
            new_ecoc.append(new_e)
    return new_ecoc


if __name__ == '__main__':

    fs_name = ['variance_threshold', 'linear_svc', 'tree', 'fclassif', 'RandForReg']

    microarray_dataname = ['Breast', 'Cancers', 'DLBCL', 'GCM', 'Leukemia1', 'Leukemia2', 'Lung1', 'SRBCT']

    UCI_dataname = ['car', 'cleveland', 'dermatology','led7digit' \
        , 'led24digit', 'letter', 'satimage', 'segment' \
        , 'vehicle', 'vowel', 'yeast']

    other_ECOC = [ 'OVA_ECOC','OVO_ECOC','Dense_random_ECOC','Sparse_random_ECOC'\
                ,'D_ECOC','DC_ECOC F1','DC_ECOC F2','DC_ECOC F3','DC_ECOC N2','DC_ECOC N3','DC_ECOC Cluster']

    module_path = os.path.dirname(__file__)
    data_folder_path = module_path + '/UCI/FS_data/'
    matrix_folder_path = module_path + '/UCI/DC_matrix/'

    # 创建结果文件夹
    res_folder_path = module_path + '/UCI/UCI_res/SAT_ECOC/SVM/'
    if not os.path.exists(res_folder_path):
        os.makedirs(res_folder_path)
    #
    selected_dataname = UCI_dataname[1:]
    ecoc_name = get_conbination(['F1','F2','F3','N2','N3','Cluster'])
    selected_ecoc_name = get_Ternary(ecoc_name,['F1'])  # 为ECOC 算法添加evaluation 操作
    selected_fs_name = fs_name[2:]


    # selected_dataname = UCI_dataname
    # ecoc_name = other_ECOC
    # selected_ecoc_name = other_ECOC  # 为ECOC 算法添加evaluation 操作
    # selected_fs_name = fs_name[2:]

    for k in range(len(selected_fs_name)):

        if k != 0:
            continue

        LOG_FORMAT = "%(message)s"
        logging.basicConfig(filename=res_folder_path + selected_fs_name[k] + '_log.txt', level=logging.DEBUG,
                            format=LOG_FORMAT)

        # save evaluation varibles
        data_acc, data_simacc, data_precision, data_specifity, data_sensitivity, data_cls_acc, data_Fscore = [], [], [], [], [], [], []

        for i in range(len(selected_dataname)):

            train_path = data_folder_path + selected_fs_name[k] + '/' + selected_dataname[i] + '_train.csv'
            test_path = data_folder_path + selected_fs_name[k] + '/' + selected_dataname[i] + '_test.csv'
            train_data, train_label = Read_Write_tool.read_Microarray_Dataset(train_path)
            test_data, test_label = Read_Write_tool.read_Microarray_Dataset(test_path)

            acc, simacc, precision, specifity, sensitivity, cls_acc, Fscore = [], [], [], [], [], [], []

            for j in range(len(selected_ecoc_name)):

                # if selected_ecoc_name[j].find('SAT_ECOC F1 F2 C1 evaluation=F1') == -1:
                #     continue

                logging.info('\n\nSVM classifier')
                logging.info('Time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                logging.info('FS: ' + selected_fs_name[k])
                logging.info('Dataset： ' + selected_dataname[i])
                logging.info('ECOC: ' + selected_ecoc_name[j])
                logging.info('Not Using KNN-Decoding')

                print('\n\nSVM classifier')
                print('Time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                print('FS: ' + selected_fs_name[k])
                print('Dataset： ' + selected_dataname[i])
                print('ECOC: ' + selected_ecoc_name[j])
                print('Not Using KNN-Decoding')

                res = ECOC_Process(selected_dataname[i], train_data, train_label, test_data, test_label,
                                   selected_ecoc_name[j], matrix_folder_path)

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
