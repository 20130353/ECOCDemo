# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 19-1-3
# file: test
# description:

import numpy as np

col = np.array([1, 1, 1, 1, -1])
matrix = np.array([[1, 1, 1, -1], [1, -1, 1, 1], [1, 1, 1, 1], [1, -1, 1, 1], [-1, -1, -1, -1]])


def same(col,matrix):
    col = col.reshape((1, -1))[0]
    try:
        column_len = matrix.shape[1]
    except IndexError:
        for each in zip(col, matrix):
            if each[0] != each[1]:
                return False
        return True
    else:
        for i in range(column_len):
            i_column = matrix[:, i]
            for each in zip(col, i_column):
                if each[0] != each[1]:
                    return False
        return True

if __name__ == '__main__':
    print(same(col,matrix))
