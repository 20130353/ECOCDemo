"""
There are some common ECOC classifiers in this model, which shown as below
1.OVA ECOC
2.OVO ECOC
3.Dense random ECOC
4.Sparse random ECOC
5.D ECOC
6.AGG ECOC
7.CL_ECOC
8.ECOC_ONE

There are all defined as class, which inherit __BaseECOC

edited by Feng Kaijie
2017/11/30
"""
import random
from abc import ABCMeta
from itertools import combinations
import logging
import copy
import time
import math
from collections import Counter
from sklearn.cluster import AffinityPropagation, MeanShift, estimate_bandwidth, DBSCAN, Birch

from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.special import comb
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
# import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import neighbors

from ECOCDemo.ECOC import Criterion
from ECOCDemo.ECOC import Matrix_tool as MT
from ECOCDemo.ECOC.Distance import *
from ECOCDemo.ECOC.SFFS import sffs
from ECOCDemo.ECOC import Greedy_Search
from ECOCDemo.Common.Evaluation_tool import Evaluation
from ECOCDemo.ECOC.Greedy_Search import get_DC_value

__all__ = ['Self_Adaption_ECOC', 'OVA_ECOC', 'OVO_ECOC', 'Dense_random_ECOC', 'Sparse_random_ECOC', 'D_ECOC',
           'AGG_ECOC', 'CL_ECOC', 'DC_ECOC']


# inner class should be marked as __class
class __BaseECOC(object):
    """
    the base class for all to inherit
    """
    __metaclass__ = ABCMeta

    def __init__(self, distance_measure=euclidean_distance, base_estimator=svm.SVC):
        """
        :param distance_measure: a callable object to define the way to calculate the distance between predicted vector
                                    and true vector
        :param base_estimator: a class with fit and predict method, which define the base classifier for ECOC
        """
        self.estimator = base_estimator
        self.predictors = []
        self.matrix = None
        self.index = None
        self.distance_measure = distance_measure
        self.predicted_vector = []
        self.train_data = []
        self.train_label = []

    def create_matrix(self, data, label):

        """
        a method to create coding matrix for ECOC
        :param data: the data used in ecoc
        :param label: the corresponding label to data
        :return: coding matrix
        """
        raise AttributeError('create_matrix is not defined!')

    def fit(self, data, label, **estimator_param):
        """
        a method to train base estimator based on given data and label
        :param data: data used to train base estimator
        :param label: label corresponding to the data
        :param estimator_param: some param used by base estimator
        :return: None
        """
        self.train_data = data
        self.train_label = label
        self.predictors = []
        self.matrix, self.index = self.create_matrix(data, label)

        for i in range(self.matrix.shape[1]):
            dat, cla = MT.get_data_from_col(data, label, self.matrix[:, i], self.index)
            estimator = self.estimator(**estimator_param).fit(dat, cla)
            self.predictors.append(estimator)

    def predict(self, data):
        """
        a method used to predict label for give data
        :param data: data to predict
        :return: predicted label
        """
        res = []
        if len(self.predictors) == 0:
            logging.debug('The Model has not been fitted!')
        if len(data.shape) == 1:
            data = np.reshape(data, [1, -1])
        for i in data:
            # find k neighbors from train data
            knn_model = neighbors.KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3).fit(self.train_data,
                                                                                                 self.train_label)
            knn_pre_label = knn_model.predict([i])
            predicted_vector = self._use_predictors(i)  # one row
            index = {l: i for i, l in enumerate(np.unique(self.train_label))}
            knn_pre_index = index[knn_pre_label[0]]
            # make it 0 when the knn predicted class is 0
            try:
                column_len = len(self.matrix[0])
            except TypeError:
                if self.matrix[knn_pre_index] == 0:
                    predicted_vector[0] = 0
            else:
                for j in range(column_len):
                    if self.matrix[knn_pre_index][j] == 0:
                        predicted_vector[j] = 0

            self.predicted_vector.append(list(predicted_vector))
            value = MT.closet_vector(predicted_vector, self.matrix, self.distance_measure)
            res.append(MT.get_key(self.index, value))
        return np.array(res)

    def validate(self, data, label, k=3, **estimator_params):
        """
        using cross validate method to validate model
        :param data: data used in validate
        :param label: ;label corresponding to data
        :param k: k-fold validate
        :param estimator_params: params used by base estimators
        :return: accuracy
        """
        acc_list = []
        kf = KFold(n_splits=k, shuffle=True)
        original_predictors = copy.deepcopy(self.predictors)
        for train_index, test_index in kf.split(data):
            data_train, data_test = data[train_index], data[test_index]
            label_train, label_test = label[train_index], label[test_index]
            self.fit(data_train, label_train, **estimator_params)
            label_predicted = self.predict(data_test)
            acc_list.append(accuracy_score(label_test, label_predicted))
        self.predictors = copy.deepcopy(original_predictors)
        return np.mean(acc_list)

    def _use_predictors(self, data):
        """
        :param data: data to predict
        :return: predicted vector
        """
        res = []
        for i in self.predictors:
            res.append(i.predict(np.array([data]))[0])
        return np.array(res)


class OVA_ECOC(__BaseECOC):
    """
    ONE-VERSUS-ONE ECOC
    """

    def create_matrix(self, data, label):
        index = {l: i for i, l in enumerate(np.unique(label))}
        matrix = np.eye(len(index)) * 2 - 1
        return matrix, index


class OVO_ECOC(__BaseECOC):
    """
    ONE-VERSUS-ONE ECOC
    """

    def create_matrix(self, data, label):
        index = {l: i for i, l in enumerate(np.unique(label))}
        groups = combinations(range(len(index)), 2)
        matrix_row = len(index)
        matrix_col = np.int(comb(len(index), 2))
        col_count = 0
        matrix = np.zeros((matrix_row, matrix_col))
        for group in groups:
            class_1_index = group[0]
            class_2_index = group[1]
            matrix[class_1_index, col_count] = 1
            matrix[class_2_index, col_count] = -1
            col_count += 1
        return matrix, index


class Dense_random_ECOC(__BaseECOC):
    """
    Dense random ECOC
    """

    def create_matrix(self, data, label):
        while True:
            index = {l: i for i, l in enumerate(np.unique(label))}
            matrix_row = len(index)
            if matrix_row > 3:
                matrix_col = np.int(np.floor(10 * np.log10(matrix_row)))
            else:
                matrix_col = matrix_row
            matrix = np.random.random((matrix_row, matrix_col))
            class_1_index = matrix > 0.5
            class_2_index = matrix < 0.5
            matrix[class_1_index] = 1
            matrix[class_2_index] = -1
            if (not MT.exist_same_col(matrix)) and (not MT.exist_same_row(matrix)) and MT.exist_two_class(matrix):
                return matrix, index


class Sparse_random_ECOC(__BaseECOC):
    """
    Sparse random ECOC
    """

    def create_matrix(self, data, label):
        while True:
            index = {l: i for i, l in enumerate(np.unique(label))}
            matrix_row = len(index)
            if matrix_row > 3:
                matrix_col = np.int(np.floor(15 * np.log10(matrix_row)))
            else:
                matrix_col = np.int(np.floor(10 * np.log10(matrix_row)))
            matrix = np.random.random((matrix_row, matrix_col))
            class_0_index = np.logical_and(0.25 <= matrix, matrix < 0.75)
            class_1_index = matrix >= 0.75
            class_2_index = matrix < 0.25
            matrix[class_0_index] = 0
            matrix[class_1_index] = 1
            matrix[class_2_index] = -1
            if (not MT.exist_same_col(matrix)) and (not MT.exist_same_row(matrix)) and MT.exist_two_class(matrix):
                return matrix, index


class D_ECOC(__BaseECOC):
    """
    Discriminant ECOC
    """

    def create_matrix(self, data, label):
        index = {l: i for i, l in enumerate(np.unique(label))}
        matrix = None
        labels_to_divide = [np.unique(label)]
        while len(labels_to_divide) > 0:
            label_set = labels_to_divide.pop(0)
            datas, labels = MT.get_data_subset(data, label, label_set)
            class_1_variety_result, class_2_variety_result = sffs(datas, labels)
            new_col = np.zeros((len(index), 1))
            for i in class_1_variety_result:
                new_col[index[i]] = 1
            for i in class_2_variety_result:
                new_col[index[i]] = -1
            if matrix is None:
                matrix = copy.copy(new_col)
            else:
                matrix = np.hstack((matrix, new_col))
            if len(class_1_variety_result) > 1:
                labels_to_divide.append(class_1_variety_result)
            if len(class_2_variety_result) > 1:
                labels_to_divide.append(class_2_variety_result)
        return matrix, index


class AGG_ECOC(__BaseECOC):
    """
    Agglomerative ECOC
    """

    def create_matrix(self, data, label):
        index = {l: i for i, l in enumerate(np.unique(label))}
        matrix = None
        labels_to_agg = np.unique(label)
        labels_to_agg_list = [[x] for x in labels_to_agg]
        label_dict = {labels_to_agg[value]: value for value in range(labels_to_agg.shape[0])}
        num_of_length = len(labels_to_agg_list)
        class_1_variety = []
        class_2_variety = []
        while len(labels_to_agg_list) > 1:
            score_result = np.inf
            for i in range(0, len(labels_to_agg_list) - 1):
                for j in range(i + 1, len(labels_to_agg_list)):
                    class_1_data, class_1_label = MT.get_data_subset(data, label, labels_to_agg_list[i])
                    class_2_data, class_2_label = MT.get_data_subset(data, label, labels_to_agg_list[j])
                    score = Criterion.agg_score(class_1_data, class_1_label, class_2_data, class_2_label,
                                                score=Criterion.max_distance_score)
                    if score < score_result:
                        score_result = score
                        class_1_variety = labels_to_agg_list[i]
                        class_2_variety = labels_to_agg_list[j]
            new_col = np.zeros((num_of_length, 1))
            for i in class_1_variety:
                new_col[label_dict[i]] = 1
            for i in class_2_variety:
                new_col[label_dict[i]] = -1
            if matrix is None:
                matrix = new_col
            else:
                matrix = np.hstack((matrix, new_col))
            new_class = class_1_variety + class_2_variety
            labels_to_agg_list.remove(class_1_variety)
            labels_to_agg_list.remove(class_2_variety)
            labels_to_agg_list.insert(0, new_class)
        return matrix, index


class CL_ECOC(__BaseECOC):
    """
    Centroid loss ECOC, which use regressors as base estimators
    """

    def __init__(self, distance_measure=euclidean_distance, base_estimator=svm.SVR):
        super(CL_ECOC, self).__init__(distance_measure, base_estimator)

    def create_matrix(self, data, label):
        index = {l: i for i, l in enumerate(np.unique(label))}
        matrix = None
        labels_to_divide = [np.unique(label)]
        while len(labels_to_divide) > 0:
            label_set = labels_to_divide.pop(0)
            datas, labels = MT.get_data_subset(data, label, label_set)
            class_1_variety_result, class_2_variety_result = sffs(datas, labels,
                                                                  score=Criterion.max_center_distance_score)
            class_1_data_result, class_1_label_result = MT.get_data_subset(data, label, class_1_variety_result)
            class_2_data_result, class_2_label_result = MT.get_data_subset(data, label, class_2_variety_result)
            class_1_center_result = np.average(class_1_data_result, axis=0)
            class_2_center_result = np.average(class_2_data_result, axis=0)
            belong_to_class_1 = [
                euclidean_distance(x, class_1_center_result) <= euclidean_distance(x, class_2_center_result)
                for x in class_1_data_result]
            belong_to_class_2 = [
                MT.euclidean_distance(x, class_2_center_result) <= MT.euclidean_distance(x, class_1_center_result)
                for x in class_2_data_result]
            class_1_true_num = {k: 0 for k in class_1_variety_result}
            class_2_true_num = {k: 0 for k in class_2_variety_result}
            for y in class_1_label_result[belong_to_class_1]:
                class_1_true_num[y] += 1
            for y in class_2_label_result[belong_to_class_2]:
                class_2_true_num[y] += 1
            class_1_label_count = {k: list(class_1_label_result).count(k) for k in class_1_variety_result}
            class_2_label_count = {k: list(class_2_label_result).count(k) for k in class_2_variety_result}
            class_1_ratio = {k: class_1_true_num[k] / class_1_label_count[k] for k in class_1_variety_result}
            class_2_ratio = {k: -class_2_true_num[k] / class_2_label_count[k] for k in class_2_variety_result}
            new_col = np.zeros((len(index), 1))
            for i in class_1_ratio:
                new_col[index[i]] = class_1_ratio[i]
            for i in class_2_ratio:
                new_col[index[i]] = class_2_ratio[i]
            if matrix is None:
                matrix = copy.copy(new_col)
            else:
                matrix = np.hstack((matrix, new_col))
            if len(class_1_variety_result) > 1:
                labels_to_divide.append(class_1_variety_result)
            if len(class_2_variety_result) > 1:
                labels_to_divide.append(class_2_variety_result)
        return matrix, index


class ECOC_ONE(__BaseECOC):
    """
    ECOC-ONE:Optimal node embedded ECOC
    """

    def __init__(self, distance_measure=euclidean_distance, base_estimator=svm.SVC, iter_num=10, **param):
        self.train_data = None
        self.validate_data = None
        self.train_label = None
        self.validation_y = None
        self.estimator = base_estimator
        self.matrix = None
        self.index = None
        self.predictors = None
        self.predictor_weights = None
        self.predicted_vector = []
        self.iter_num = iter_num
        self.param = param
        self.distance_measure = distance_measure

    def create_matrix(self, train_data, train_label, validate_data, validate_label, estimator, **param):
        index = {l: i for i, l in enumerate(np.unique(train_label))}
        matrix = None
        predictors = []
        predictor_weights = []
        labels_to_divide = [np.unique(train_label)]
        while len(labels_to_divide) > 0:
            label_set = labels_to_divide.pop(0)
            label_count = len(label_set)
            groups = combinations(range(label_count), np.int(np.ceil(label_count / 2)))
            score_result = 0
            est_result = None
            for group in groups:
                class_1_variety = np.array([label_set[i] for i in group])
                class_2_variety = np.array([l for l in label_set if l not in class_1_variety])
                class_1_data, class_1_label = MT.get_data_subset(train_data, train_label, class_1_variety)
                class_2_data, class_2_label = MT.get_data_subset(train_data, train_label, class_2_variety)
                class_1_cla = np.ones(len(class_1_data))
                class_2_cla = -np.ones(len(class_2_data))
                train_d = np.vstack((class_1_data, class_2_data))
                train_c = np.hstack((class_1_cla, class_2_cla))
                est = estimator(**param).fit(train_d, train_c)
                class_1_data, class_1_label = MT.get_data_subset(validate_data, validate_label, class_1_variety)
                class_2_data, class_2_label = MT.get_data_subset(validate_data, validate_label, class_2_variety)
                class_1_cla = np.ones(len(class_1_data))
                class_2_cla = -np.ones(len(class_2_data))
                validation_d = np.array([])
                validation_c = np.array([])
                try:
                    validation_d = np.vstack((class_1_data, class_2_data))
                    validation_c = np.hstack((class_1_cla, class_2_cla))
                except Exception:
                    if len(class_1_data) > 0:
                        validation_d = class_1_data
                        validation_c = class_1_cla
                    elif len(class_2_data) > 0:
                        validation_d = class_2_data
                        validation_c = class_2_cla
                if validation_d.shape[0] > 0 and validation_d.shape[1] > 0:
                    score = est.score(validation_d, validation_c)
                else:
                    score = 0.8
                if score >= score_result:
                    score_result = score
                    est_result = est
                    class_1_variety_result = class_1_variety
                    class_2_variety_result = class_2_variety
            new_col = np.zeros((len(index), 1))
            for i in class_1_variety_result:
                new_col[index[i]] = 1
            for i in class_2_variety_result:
                new_col[index[i]] = -1
            if matrix is None:
                matrix = copy.copy(new_col)
            else:
                matrix = np.hstack((matrix, new_col))
            predictors.append(est_result)
            predictor_weights.append(MT.estimate_weight(1 - score_result))
            if len(class_1_variety_result) > 1:
                labels_to_divide.append(class_1_variety_result)
            if len(class_2_variety_result) > 1:
                labels_to_divide.append(class_2_variety_result)
        return matrix, index, predictors, predictor_weights

    def fit(self, data, label):
        self.train_data, self.validate_data, self.train_label, self.validation_y = train_test_split(data, label,
                                                                                                    test_size=0.25)
        self.matrix, self.index, self.predictors, self.predictor_weights = \
            self.create_matrix(self.train_data, self.train_label, self.validate_data, self.validation_y, self.estimator,
                               **self.param)
        feature_subset = MT.get_subset_feature_from_matrix(self.matrix, self.index)
        for i in range(self.iter_num):
            y_pred = self.predict(self.validate_data)
            y_true = self.validation_y
            confusion_matrix = MT.create_confusion_matrix(y_true, y_pred, self.index)
            while True:
                max_index = np.argmax(confusion_matrix)
                max_index_y = np.floor(max_index / confusion_matrix.shape[1])
                max_index_x = max_index % confusion_matrix.shape[1]
                label_y = MT.get_key(self.index, max_index_y)
                label_x = MT.get_key(self.index, max_index_x)
                score_result = 0
                col_result = None
                est_result = None
                est_weight_result = None
                feature_subset_m = None
                feature_subset_n = None
                for m in range(len(feature_subset) - 1):
                    for n in range(m + 1, len(feature_subset)):
                        if ((label_y in feature_subset[m] and label_x in feature_subset[n])
                            or (label_y in feature_subset[n] and label_x in feature_subset[m])) \
                                and (set(feature_subset[m]).intersection(set(feature_subset[n])) == set()):
                            col = MT.create_col_from_partition(feature_subset[m], feature_subset[n], self.index)
                            if not MT.have_same_col(col, self.matrix):
                                train_data, train_cla = MT.get_data_from_col(self.train_data, self.train_label, col,
                                                                             self.index)
                                est = self.estimator(**self.param).fit(train_data, train_cla)
                                validation_data, validation_cla = MT.get_data_from_col(self.validate_data,
                                                                                       self.validation_y, col,
                                                                                       self.index)
                                if validation_data is None:
                                    score = 0.8
                                else:
                                    score = est.score(validation_data, validation_cla)
                                if score >= score_result:
                                    score_result = score
                                    col_result = col
                                    est_result = est
                                    est_weight_result = MT.estimate_weight(1 - score_result)
                                    feature_subset_m = m
                                    feature_subset_n = n
                if col_result is None:
                    confusion_matrix[np.int(max_index_y), np.int(max_index_x)] = 0
                    if np.sum(confusion_matrix) == 0:
                        break
                else:
                    break
            try:
                self.matrix = np.hstack((self.matrix, col_result))
                self.predictors.append(est_result)
                self.predictor_weights.append(est_weight_result)
                feature_subset.append(feature_subset[feature_subset_m] + feature_subset[feature_subset_n])
            except (TypeError, ValueError):
                pass

    def predict(self, data):
        res = []
        if len(self.predictors) == 0:
            logging.debug('The Model has not been fitted!')
        if len(data.shape) == 1:
            data = np.reshape(data, [1, -1])

        for i in data:
            predict_res = self._use_predictors(i)

            if self.predicted_vector == []:
                self.predicted_vector = copy.deepcopy(predict_res)
            else:
                self.predicted_vector = np.row_stack((self.predicted_vector, predict_res))

            value = MT.closet_vector(predict_res, self.matrix, y_euclidean_distance, np.array(self.predictor_weights))
            res.append(MT.get_key(self.index, value))

        vector = []
        for i in range(self.matrix.shape[1]):
            vector.append(list(self.predicted_vector[:, i]))
        self.predicted_vector = copy.deepcopy(vector)

        return np.array(res)


class DC_ECOC(__BaseECOC):
    """
    DC ECOC
    code by sunmengxin
    """

    def __init__(self, dc_option='F1', base_M=None, distance_measure=euclidean_distance, base_estimator=svm.SVC):
        """
        :param:dc_option,used for select dc measure to split two groups
                default is 'F1'

        :param base_M: pre_defined matrix


        :param distance_measure: a callable object to define the way to calculate the distance between predicted vector
                                    and true vector

        :param base_estimator: a class with fit and predict method, which define the base classifier for ECOC

        """

        super(DC_ECOC, self).__init__(distance_measure, base_estimator)
        self.matrix = base_M
        self.dc_option = dc_option

    def create_matrix(self, data, label, dc_option):
        labels_to_divide = [np.unique(label)]
        index = {l: i for i, l in enumerate(np.unique(label))}

        matrix = None
        while len(labels_to_divide) > 0:
            label_set = labels_to_divide.pop(0)

            # get correspoding label and data from whole data and label
            datas, labels = MT.get_data_subset(data, label, label_set)

            # DC search
            class_1, class_2 = Greedy_Search.greedy_search(datas, labels, dc_option=dc_option)
            new_col = np.zeros((len(index), 1))
            for i in class_1:
                new_col[index[i]] = 1
            for i in class_2:
                new_col[index[i]] = -1
            if matrix is None:
                matrix = copy.copy(new_col)
            else:
                matrix = np.hstack((matrix, new_col))
            if len(class_1) > 1:
                labels_to_divide.append(class_1)
            if len(class_2) > 1:
                labels_to_divide.append(class_2)
        return matrix, index

    def fit(self, data, label, **estimator_param):
        """
        a method to train base estimator based on given data and label
        :param data: data used to train base estimator
        :param label: label corresponding to the data
        :param estimator_param: some param used by base estimator
        :return: None
        """
        self.train_data = data
        self.train_label = label
        if self.matrix is not None:
            self.index = {l: i for i, l in enumerate(np.unique(label))}
        else:
            self.matrix, self.index = self.create_matrix(data, label, self.dc_option)
        for i in range(self.matrix.shape[1]):
            dat, cla = MT.get_data_from_col(data, label, self.matrix[:, i], self.index)
            estimator = self.estimator(**estimator_param).fit(dat, cla)
            self.predictors.append(estimator)


class Temp_Class(__BaseECOC):

    def __init__(self, distance_measure=euclidean_distance, base_estimator=svm.SVC):
        """
        :param distance_measure: a callable object to define the way to calculate the distance between predicted vector
                                    and true vector
        :param base_estimator: a class with fit and predict method, which define the base classifier for ECOC
        """
        super(Temp_Class, self).__init__(distance_measure, base_estimator)
        self._matrix = None

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix = matrix

    def fit(self, data, label, **estimator_param):
        """
        a method to train base estimator based on given data and label
        :param data: data used to train base estimator
        :param label: label corresponding to the data
        :param estimator_param: some param used by base estimator
        :return: None
        """
        self.predictors = []
        self.train_data = data
        self.train_label = label
        self.index = {l: i for i, l in enumerate(np.unique(self.train_label))}

        try:
            column_len = self._matrix.shape[1]
        except IndexError:
            dat, cla = MT.get_data_from_col(data, label, self._matrix, self.index)
            estimator = self.estimator(**estimator_param).fit(dat, cla)
            self.predictors.append(estimator)
        else:
            for i in range(column_len):
                dat, cla = MT.get_data_from_col(data, label, self._matrix[:, i], self.index)
                estimator = self.estimator(**estimator_param).fit(dat, cla)
                self.predictors.append(estimator)


class Self_Adaption_ECOC(__BaseECOC):

    def __init__(self, dc_option, base_M, create_method, distance_measure=euclidean_distance, base_estimator=svm.SVC):

        super(Self_Adaption_ECOC, self).__init__(distance_measure, base_estimator)
        self.dc_option = dc_option
        self.base_M = base_M
        self.create_method = create_method

    def check_pos_neg_len(self, new_column, data, label):

        pos_len, neg_len = 0, 0
        for i in range(len(new_column)):
            d = data[label == MT.get_key(self.index, i)]
            l = label[label == MT.get_key(self.index, i)]
            if new_column[i] == 1:
                pos_len += len(d)
            elif new_column[i] == -1:
                neg_len += len(d)
        return float(pos_len) / neg_len

    def find_most_simi_class(self, small_cls, large_cls, new_column):
        '''

        :param small_cls:  small class index
        :param large_cls:  large class index
        :param new_column:
        :return:
        '''
        large_cls_DC = {}.fromkeys(large_cls)
        for each_cls in large_cls:
            large_cls_DC[each_cls] = get_DC_value(self.train_data, self.train_label,
                                                  [MT.get_key(self.index, e) for e in small_cls],
                                                  [MT.get_key(self.index, each_cls)],
                                                  self.dc_option)

        large_cls_DC = sorted(large_cls_DC.items(), key=lambda x: x[1])  # 复杂度从小到大
        stop = False
        for each_cls in large_cls_DC:
            new_column[each_cls[0]] = -1
            _, _, _, pos_data, neg_data, extra_data, _, _, _ = self.get_pos_neg(new_column, self.train_data,
                                                                                self.train_label)
            try:
                pos_len = len(pos_data)
                neg_len = len(neg_data)
            except IndexError:  # 数量为０
                new_column[each_cls[0]] = 1  # 返回上一步
                stop = True
            else:
                # 数量接近２倍
                if (float(pos_len) / float(neg_len) <= 2) or (float(neg_len) / float(pos_len) <= 2):
                    stop = True
            if stop is True:
                return new_column
        return new_column

    def adjust_unbalance(self, small_cls, large_cls, extra_cls, small_data, large_data, extra_data, small_label,
                         large_label, extra_label, new_column):

        logging.info('*======adjust_unbalance=======*')

        logging.info('small class sample %s' %str(Counter(small_label)))
        logging.info('large class sample %s' %str(Counter(large_label)))

        # 用small class 和 extra class的数据拟合
        if extra_data is None or len(extra_data) == 0:
            logging.info('no extra class data')
            logging.info('before change column:\t' + str([each[0] for each in new_column]))
            new_column = self.find_most_simi_class(small_cls, large_cls, new_column)
            logging.info('after change column:\t' + str([each[0] for each in new_column]))
            return new_column

        else:
            AP_train_label = np.concatenate((small_label, extra_label), axis=0)
            AP_train_data = np.concatenate((small_data, extra_data), axis=0)

        # AP = AffinityPropagation().fit(AP_train_data)  # AP 算法拟合数据
        # bandwidth = estimate_bandwidth(AP_train_data, quantile=0.05) Mean shitf 算法拟合数据
        # AP = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(AP_train_data)

        # AP = DBSCAN(eps=3, min_samples=2).fit(AP_train_data)
        AP = Birch().fit(AP_train_data)
        # 找到所有合适的簇，同时选中簇包含的样本
        train_members = [False for _ in range(len(AP_train_data))]
        cluster_label = AP.labels_
        n_cluster = len(np.unique(cluster_label))

        logging.info('***---------- cluster label counter-------------***')
        small_cls_sample_dict = {}
        small_cls_sample_len_dict = {}
        for k in range(n_cluster):
            class_members = cluster_label == k
            counter = Counter(AP_train_label[class_members])  # 找到一个簇中样本的label的个数
            logging.info('cluster label %s %s' % (str(k),str(counter)))
            small_cls_sample_len = sum([counter[MT.get_key(self.index, inx)] for inx in small_cls])
            small_cls_sample_len_dict[k] = round(float(small_cls_sample_len) / sum(class_members), 2)  # 计算每个簇中小类样本的比例
            small_cls_sample_dict[k] = class_members

        small_cls_sample_len_dict = sorted(small_cls_sample_len_dict.items(), key=lambda x: x[1])  # 排序之后的数据类型是ｔｕｐｌｅ
        small_cls_sample_len_dict.reverse()
        for key, value in small_cls_sample_len_dict:  # 如果小类样本不够大类样本的一般的话就继续添加
            if sum(train_members) == 0 or float(len(large_data)) / sum(train_members) >= 2:
                train_members = list(train_members) and list(small_cls_sample_dict[key])
            else:
                break

            # *===================  logging   ==============================
        logging.info('new_column')
        logging.info(new_column)
        logging.info(
            'small_class len :%d,large_class len:%d, small_class sample:%d,large_class sample len:%d'
            % (len(small_cls), len(large_cls), len(small_data), len(large_data)))

        if extra_cls is not None and extra_data is not None:
            logging.info('extra_class: %d,extra_sample len: %d' % (len(extra_cls), len(extra_data)))

        logging.info('cluster len:%d' % (n_cluster))
        logging.info('cluster contain small sample len:' + str(small_cls_sample_len_dict))
        # *===================  logging   ==============================

        if sum(train_members) == 0:
            logging.info('no one train sample is selected')
            return new_column

        sel_train_data = AP_train_data[train_members]
        sel_train_label = AP_train_label[train_members]
        sel_cls = np.unique(sel_train_label)

        if len(large_data)/float(len(sel_train_data)) >=3:
            logging.info('large data sampling')
            np.random.shuffle(large_data)
            large_data = large_data[:len(sel_train_data)*2]

        # 构造正负类的样本标签,预测验证集的数据，判断
        pos_label = [1 for _ in range(len(large_data))]
        neg_label = [-1 for _ in range(len(sel_train_data))]

        model = self.estimator().fit(np.concatenate((large_data, sel_train_data), axis=0),
                                     np.concatenate((pos_label, neg_label), axis=0))

        logging.info('before change column:\t' + str([each[0] for each in new_column]))
        logging.info('pos len:%d vs. neg len:%d' % (len(pos_label), len(neg_label)))
        logging.info('selected class:%s' % str(sel_cls))

        small_cls_label = [MT.get_key(self.index, each) for each in small_cls]
        logging.info('small_cls_label %s' % str(small_cls_label))
        temp_label = model.predict(self.val_data)
        for k in sel_cls:
            pos_neg_ratio = self.check_pos_neg_len(new_column, self.train_data, self.train_label)
            if pos_neg_ratio <= 2 or (pos_neg_ratio >= 0.2 and pos_neg_ratio <= 1):  # 如果正负类样本个数差不多就停止
                break
            if k not in small_cls_label:  # 只能修改不是小类的样本
                class_members = self.val_label == k
                try:
                    majority_label = list(Counter(temp_label[class_members])).pop()
                except IndexError:
                    pass
                else:
                    logging.info('class label %s' % str(k))
                    logging.info('majority label %s' % majority_label)
                    if majority_label == new_column[small_cls[0]]:  # 只有将类别改成和小类一样的话，才允许修改
                        new_column[self.index[k]] = majority_label  # 这边会修改原先可能为+1，,1，0的值

        logging.info('after change column:\t' + str([each[0] for each in new_column]))
        return new_column

    def get_pos_neg(self, new_column, train_data, train_label):
        pos_cls, neg_cls, extra_cls, pos_data, neg_data, extra_data, pos_label, neg_label, extra_label = [], [], [], None, None, None, None, None, None
        for i in range(len(new_column)):
            d = train_data[train_label == MT.get_key(self.index, i)]
            l = train_label[train_label == MT.get_key(self.index, i)]
            if new_column[i] == 1:
                pos_cls.append(i)
                if pos_data is None:
                    pos_data = copy.deepcopy(d)
                    pos_label = copy.deepcopy(l)
                else:
                    pos_data = np.concatenate((pos_data, d), axis=0)
                    pos_label = np.concatenate((pos_label, l), axis=0)
            elif new_column[i] == -1:
                neg_cls.append(i)
                if neg_data is None:
                    neg_data = copy.deepcopy(d)
                    neg_label = copy.deepcopy(l)
                else:
                    neg_data = np.concatenate((neg_data, d), axis=0)
                    neg_label = np.concatenate((neg_label, l), axis=0)
            else:
                extra_cls.append(i)
                if extra_data is None:
                    extra_data = copy.deepcopy(d)
                    extra_label = copy.deepcopy(l)
                else:
                    extra_data = np.concatenate((extra_data, d), axis=0)
                    extra_label = np.concatenate((extra_label, l), axis=0)

        return pos_cls, neg_cls, extra_cls, pos_data, neg_data, extra_data, pos_label, neg_label, extra_label

    def temp_log(self, matrix, train_data, train_label, test_data, test_label, **kwargs):

        temp_class = Temp_Class()
        temp_class.matrix = matrix
        temp_class.fit(train_data, train_label)
        pred_label = temp_class.predict(test_data)
        cfus_matrix = confusion_matrix(test_label, pred_label)
        Eva = Evaluation(test_label, pred_label)
        classifier_acc = Eva.evaluate_classifier_accuracy(matrix, temp_class.predicted_vector,
                                                          test_label, self.index)

        # print the info of one column
        try:
            col_len = matrix.shape[1]
        except IndexError:

            pos_neg_r_train = self.check_pos_neg_len(matrix, train_data, train_label)
            pos_neg_r_test = self.check_pos_neg_len(matrix, test_data, test_label)
            counter_train = Counter(train_label)
            counter_test = Counter(test_label)
            pred_label_clsfer = [each[0] for each in temp_class.predicted_vector]
            counter_pred = Counter(pred_label_clsfer)
            clsfer_acc_i = classifier_acc[0]

            logging.info('------------------------------')
            logging.info('matrix')
            logging.info(matrix)
            logging.info('column %d info:' % 0)
            logging.info('pos_neg_r_train : %f' % round(pos_neg_r_train, 2))
            logging.info('pos_neg_r_test : %f' % round(pos_neg_r_test, 2))
            logging.info('count train %s' % str(counter_train))
            logging.info('count test %s' % str(counter_test))
            logging.info('count pred %s' % str(counter_pred))
            logging.info('classifier acc is %f' % round(clsfer_acc_i, 2))

        else:
            logging.info('matrix')
            logging.info(matrix)
            for i in range(col_len):
                pos_neg_r_train = self.check_pos_neg_len(matrix[:, i], train_data, train_label)
                pos_neg_r_test = self.check_pos_neg_len(matrix[:, i], test_data, test_label)
                counter_train = Counter(train_label)
                counter_test = Counter(test_label)
                pred_label_clsfer = [each[i] for each in temp_class.predicted_vector]
                counter_pred = Counter(pred_label_clsfer)
                clsfer_acc_i = classifier_acc[i]

                logging.info('------------------------------')
                logging.info('column %d info:' % i)
                logging.info('pos_neg_r_train : %f' % round(pos_neg_r_train, 2))
                logging.info('pos_neg_r_test : %f' % round(pos_neg_r_test, 2))
                logging.info('count train %s' % str(counter_train))
                logging.info('count test %s' % str(counter_test))
                logging.info('count pred %s' % str(counter_pred))
                logging.info('classifier acc is %f' % round(clsfer_acc_i, 2))

        logging.info('confusion matrix')
        logging.info(cfus_matrix)

        if kwargs['row_HD'] is True:
            row_HD = Eva.row_HD(matrix)
            logging.info('row HD')
            logging.info(row_HD)

        if kwargs['col_HD'] is True:
            col_HD = Eva.row_HD(matrix)
            logging.info('col HD')
            logging.info(col_HD)

        return

    def cut_columns(self, matrix):

        data_len = len(self.val_label)
        try:
            column_len = matrix.shape[1]
        except IndexError:
            return matrix
        else:

            if column_len <= math.log(len(self.index), 2):
                return matrix


            #  计算列的准确率
            temp_class = Temp_Class()
            temp_class.matrix = matrix
            temp_class.fit(self.train_data, self.train_label)
            _ = temp_class.predict(self.val_data)
            clsfer_pred_label = temp_class.predicted_vector
            clsfer_acc_ratio = []
            for i in range(matrix.shape[1]):
                i_column = matrix[:, i]
                column_true_label = []
                for j in range(data_len):
                    if i_column[self.index[self.val_label[j]]] == 1:
                        column_true_label.append(1)
                    elif i_column[self.index[self.val_label[j]]] == -1:
                        column_true_label.append(-1)
                    else:
                        column_true_label.append(0)
                pre_label = [row[i] for row in clsfer_pred_label]
                clsfer_acc_ratio.append(
                    sum(list(map(lambda a, b: 1 if a == b else 0, pre_label, column_true_label))) / float(data_len))

            # 找到准确率较小的分类器的位置
            try:
                weak_clsfer_inx = [inx for inx, each in enumerate(clsfer_acc_ratio) if each < np.mean(clsfer_acc_ratio)]
            except IndexError:
                # 每个分类器的准确率都很高，weak_clsfer的数量为0
                return matrix
            else:
                while len(weak_clsfer_inx) > 0:
                    inx = weak_clsfer_inx.pop(0)
                    i_column = matrix[:, inx]
                    left_matrix = matrix[:, :inx]
                    right_matrix = matrix[:, inx + 1:]
                    if left_matrix.size == 0:
                        sub_matrix = right_matrix
                    elif right_matrix.size == 0:
                        sub_matrix = left_matrix
                    else:
                        if left_matrix.shape[0] == 1:
                            left_matrix = [[each] for each in left_matrix]
                        if right_matrix.shape[0] == 1:
                            right_matrix = [[each] for each in right_matrix]
                        sub_matrix = np.hstack([left_matrix, right_matrix])

                    temp_class = Temp_Class()
                    temp_class.matrix = matrix
                    temp_class.fit(self.train_data, self.train_label)
                    whole_pre_label = temp_class.predict(self.val_data)
                    whole_wrong_ratio = sum(
                        [1 for inx in range(data_len) if whole_pre_label[inx] != self.val_label[inx]]) / float(data_len)

                    temp_class.matrix = sub_matrix
                    temp_class.fit(self.train_data, self.train_label)
                    sub_pre_label = temp_class.predict(self.val_data)
                    sub_wrong_inx = [inx for inx in range(len(sub_pre_label)) if
                                     sub_pre_label[inx] != self.val_label[inx]]

                    temp_class.matrix = i_column
                    temp_class.fit(self.train_data, self.train_label)
                    column_pre_label = temp_class.predict(self.val_data)
                    column_wrong_inx = [inx for inx in range(len(column_pre_label)) if
                                        column_pre_label[inx] != self.val_label[inx]]
                    try:
                        column_ctrbuton = len(set(sub_wrong_inx) - set(column_wrong_inx)) / len(set(sub_wrong_inx))
                    except ZeroDivisionError:
                        column_ctrbuton = len(set(sub_wrong_inx) - set(column_wrong_inx))

                    if column_ctrbuton < 0.01 or len(sub_wrong_inx) / float(data_len) <= whole_wrong_ratio:
                        matrix = sub_matrix
                        weak_clsfer_inx = [each - 1 for each in weak_clsfer_inx]
        return matrix

    def select_column(self, matrix_pool, cplx_class, total_cplx_class_num, select_column):
        prob = 0.5
        matrix_pool = MT.shuffle_matrix(matrix_pool)
        select_cloumn_j = None


        # 如果没有复杂类的话，就随机抽取一列
        cplx_num = sum([1 if each[1] == None else 0 for each in cplx_class.items()])
        if cplx_num == len(cplx_class):
            while True:
                res = [random.randint(-1,1) for _ in matrix_pool[:,0]]
                if sum(res) == len(matrix_pool[:,0]) or sum(res) == 0:
                    continue
                else:
                    return res



        while True:
            stop = False
            for i in range(matrix_pool.shape[1]):
                i_column = matrix_pool[:, i]
                contain_num = 0
                for j in range(len(i_column)):
                    if i_column[j] != 0 and cplx_class[j] == True:
                        contain_num += 1
                if contain_num > total_cplx_class_num * prob:
                    if select_column is not None:
                        if (i_column == select_column).all() == False:
                            select_cloumn_j = i_column
                            stop = True
                    else:
                        select_cloumn_j = i_column
                        stop = True
                if stop is True:
                    break
            if select_cloumn_j is not None:
                break
            else:
                prob -= 0.05
        return select_cloumn_j

    def create_matrix(self):

        logging.info('**-----------  init matrix info --------------**')
        self.temp_log(self.base_M, self.train_data, self.train_label, self.val_data, self.val_label,
                      row_HD=True, col_HD=True)

        matrix_pool = copy.deepcopy(self.base_M)
        rand_column = random.randint(0, matrix_pool.shape[1] - 1)
        res_matrix = self.base_M[:, rand_column]

        count = 0
        change_count = 0
        while True:
            # =======    算法开始１．：计算借助confusion matrix计算复杂类      ============================================================
            logging.info('*---------------------iter %d------------------------------*' % count)
            logging.info('**-----------  current matrix info --------------**')
            self.temp_log(res_matrix, self.train_data, self.train_label, self.val_data, self.val_label,
                          row_HD=True, col_HD=True)

            count += 1
            temp_class = Temp_Class()
            temp_class.matrix = res_matrix
            temp_class.fit(self.train_data, self.train_label)
            pred_label = temp_class.predict(self.val_data)
            cfus_matrix = confusion_matrix(self.val_label, pred_label)

            cplx_class = {}.fromkeys(self.rows.keys())
            total_cplx_class_num = 0
            class_acc = [(cfus_matrix[i][i]) / sum(cfus_matrix[i, :]) for i in range(len(cfus_matrix))]

            average_class_acc = np.mean(class_acc)
            threhold = min(max(0.9*average_class_acc,0),1)
            for i in range(len(class_acc)):
                if class_acc[i] <= threhold:
                    cplx_class[i] = True
                    total_cplx_class_num += 1

            logging.info('average_class_acc %f' % round(average_class_acc, 2))
            logging.info('cplx_class_threhold %f' % round(threhold, 2))
            logging.info('cplx_class %s' % str(cplx_class))

            # =======    算法停止２．： 满足三个条件之一即可停止      ============================================================
            # 如果达到log2(N)列，并且没有复杂类存在
            try:
                column_len = res_matrix.shape[1]
            except IndexError:
                column_len = 1
            if column_len >= math.log(len(self.index), 2) and total_cplx_class_num == 0:
                # 如果每个类的分类情况都实现50%的正确率就结束
                logging.info('total_cplx_class_num == 0 break')
                break

            # 如果达到10log2(N)列, 如果每个类的分类正确率大于0.8
            elif column_len >= math.log(len(self.index), 2) and average_class_acc >= 0.8:
                logging.info('average_class_acc >= 0.8')
                break

            # 如果达到10log2(N)列
            elif column_len >= 10 * math.log(len(self.index), 2):
                logging.info('column_len >= 10*math.log(len(self.index))')
                break

            # =======    算法继续３．：挑选包含复杂类的两列     ============================================================

            select_cloumn_i = self.select_column(matrix_pool, cplx_class, total_cplx_class_num, None)

            if select_cloumn_i is None:
                raise ValueError('ERROR: SAT_ECOC select_column i is None')
            else:
                logging.info('select_i_column %s' % str([each for each in select_cloumn_i]))

            select_cloumn_j = self.select_column(matrix_pool, cplx_class, total_cplx_class_num, select_cloumn_j)
            if select_cloumn_j is None:
                raise ValueError('ERROR: SAT_ECOC select_column j is None')
            else:
                logging.info('select_j_column %s' % str([each for each in select_cloumn_i]))

            # ========   算法继续４．：生成新的一列    ============================
            try:
                most_cplx_class_inx = random.choice([inx for inx, key in cplx_class.items() if key != None])
                logging.info('most_cplx_inx is %d' % most_cplx_class_inx)
            except IndexError:
                most_cplx_class_inx = random.randint(0,len(cplx_class.items()))
                logging.info('random cplx class index is %d' %most_cplx_class_inx)

            new_column = MT.left_right_create_parent(select_cloumn_i, select_cloumn_j, self.train_data,
                                                     self.train_label, self.create_method, self.dc_option, res_matrix,
                                                     most_cplx_class_inx)

            if new_column is None:
                logging.info('new column is None')
                continue

            # ================  算法继续５．：解决unbalance问题        ======================================
            if new_column is not None:
                pos_cls, neg_cls, extra_cls, pos_data, neg_data, extra_data, pos_label, neg_label, extra_label = self.get_pos_neg(
                    new_column, self.train_data, self.train_label)

                new_column_backup = copy.deepcopy(new_column)

                if len(pos_data) / float(len(neg_data)) >= 3:  # 小类，大类，额外类
                    new_column = self.adjust_unbalance(neg_cls, pos_cls, extra_cls, neg_data, pos_data, extra_data,
                                                       neg_label, pos_label, extra_label, new_column)
                elif (len(neg_data) / float(len(pos_data)) >= 3):
                    new_column = self.adjust_unbalance(pos_cls, neg_cls, extra_cls, pos_data, neg_data, extra_data,
                                                       pos_label, neg_label, extra_label, new_column)
                if (new_column != new_column_backup).any():
                    change_count += 1
                    pos_column = [[1] for _ in new_column]
                    neg_column = [[-1] for _ in new_column]
                    if (new_column == pos_column).all() or (new_column == neg_column).all():
                        logging.info('changed new column only has one class')
                        new_column = None
                    else:
                        logging.info('** ---------- before changing, the new column performance ---------**')
                        self.temp_log(new_column_backup, self.train_data, self.train_label, self.val_data,
                                      self.val_label,
                                      row_HD=False, col_HD=False)

                        logging.info('** ---------- after changing, the new column performance ---------**')
                        self.temp_log(new_column, self.train_data, self.train_label, self.val_data, self.val_label,
                                      row_HD=False, col_HD=False)
                else:
                    logging.info('change no bit!')

            if new_column is not None:
                try:
                    res_matrix = np.hstack([res_matrix, new_column])
                except ValueError:
                    res_matrix = np.hstack([[[each] for each in res_matrix], new_column])

                matrix_pool = np.hstack([matrix_pool, new_column])

        logging.info('change ratio is:\t' + str(float(change_count) / float(count)))

        # ================ 算法继续６． 剪枝        ======================================

        final_matrix = self.cut_columns(res_matrix)
        logging.info('** ---------- before cutting, the new column performance -----------**')
        self.temp_log(res_matrix, self.train_data, self.train_label, self.val_data, self.val_label,
                      row_HD=True, col_HD=True)

        logging.info('** ---------- after cutting, the new column performance ------------**')
        self.temp_log(final_matrix, self.train_data, self.train_label, self.val_data, self.val_label,
                      row_HD=True, col_HD=True)
        return final_matrix

    def fit(self, train_data, train_label, val_data, val_label, **param):
        """
        a method to train base estimator based on given data and label
        :param data: data used to train base estimator
        :param label: label corresponding to the data
        :param estimator_param: some param used by matrix and base estimator
        :return: None
        """
        self.predictors = []
        self.train_data = train_data
        self.train_label = train_label
        self.val_data = val_data
        self.val_label = val_label
        self.index = {l: i for i, l in enumerate(np.unique(self.train_label))}
        self.rows = {i: l for i, l in enumerate(np.unique(self.train_label))}

        self.matrix = self.create_matrix()

        try:
            column_len = self.matrix.shape[1]
        except IndexError:
            dat, cla = MT.get_data_from_col(self.train_data, self.train_label, self.matrix, self.index)
            if 'estimator_param' in param:
                estimator = self.estimator(**param['estimator_param']).fit(dat, cla)
            else:
                estimator = self.estimator().fit(dat, cla)
            self.predictors.append(estimator)
        else:
            for i in range(column_len):
                dat, cla = MT.get_data_from_col(self.train_data, self.train_label, self.matrix[:, i], self.index)
                if 'estimator_param' in param:
                    estimator = self.estimator(**param['estimator_param']).fit(dat, cla)
                else:
                    estimator = self.estimator().fit(dat, cla)
                self.predictors.append(estimator)
