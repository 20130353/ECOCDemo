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
        E = Self_Adaption_ECOC(dc_option='F1', base_M=base_M, create_method='DC')
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

    unbalance_data = ['wine', 'winequality-red', 'winequality-white','thyroid',
                      'sensor_readings_24', 'sat', 'page-blocks', 'contraceptive',
                      'column_3C', 'Cardiotocography', 'avila-ts', 'abalone', 'car', 'cleveland',
                      'dermatology', 'flare','nursery', 'satimage', 'segment', 'yeast']
    # thyroid 时长1个小时，sensor_readings_24 20分钟



    other_ECOC = ['OVA_ECOC', 'OVO_ECOC', 'Dense_random_ECOC', 'Sparse_random_ECOC' \
        , 'D_ECOC', 'DC_ECOC F1', 'DC_ECOC F2', 'DC_ECOC F3', 'DC_ECOC N2', 'DC_ECOC N3', 'DC_ECOC Cluster']

    module_path = os.path.dirname(__file__)
    data_folder_path = module_path + '/UCI/4_train_val_data/'
    matrix_folder_path = module_path + '/UCI/ECOC_matrix_data/train_val/'

    # 创建结果文件夹
    res_folder_path = module_path + '/UCI/UCI_res/train_val_data/SAT_ECOC/SVM/alalysing' + str(count) + '/'
    if not os.path.exists(res_folder_path):
        os.makedirs(res_folder_path)
    #
    selected_dataname = unbalance_data
    ecoc_name = ['SAT_ECOC DR', 'SAT_ECOC SR']
    selected_ecoc_name = ['SAT_ECOC DR', 'SAT_ECOC SR']
    selected_fs_name = fs_name[:3]

    for k in range(len(selected_fs_name)):

        if k != 0:
            continue

        LOG_FORMAT = "%(message)s"
        logging.basicConfig(filename=res_folder_path + selected_fs_name[k] + '_log.txt', level=logging.DEBUG,
                            format=LOG_FORMAT)

        logging.info('算法备注：')
        logging.info('1. 使用三进制生成的新的列没有经过去重复、去相反的判断')
        logging.info('2. 对unbalance的列使用近似KNN的算法调整')
        logging.info('3. 过程打印了非常多的log，用来分析算法效果')
        logging.info('4. 要添加修改的列数量站所有列数量的比例')
        logging.info('5. 要添加修改后的列的结果对比')
        logging.info('6. 只要簇中包含小类样本就会收录')
        logging.info('7. 修改决定每个簇是否要保留的条件：当簇中包含小类样本个数超过一半的时候就留下')
        logging.info('8. 使用Birch来聚类')
        logging.info('9. 添加小类样本限制')
        logging.info('10. 防止小类样本完全霸占整个类')
        logging.info('11. 生成过程中添加了去重复去掉相反的无用处理')
        logging.info('12. 生成matrix的过程添加了至少要生产logN个column的条件限制')
        logging.info('13. 将大类样本下采样')

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
                                   val_label, selected_ecoc_name[j], matrix_folder_path, selected_fs_name[k])

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

    import warnings

    warnings.filterwarnings('ignore')

    for each in range(28,29):
        fun(each)

    # bandwidth=3.063224
