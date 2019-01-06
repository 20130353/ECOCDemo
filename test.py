# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 19-1-3
# file: test
# description:

import logging
from sklearn.model_selection import train_test_split

from ECOCDemo.Common.Read_Write_tool import read_UCI_Dataset
from ECOCDemo.Common.Read_Write_tool import write_FS_data
from ECOCDemo.FS.DC_Feature_selection import FS_selection
import os

from sklearn.preprocessing import MinMaxScaler
from collections import Counter

cplx_class = {'a': True, 'b': False, 'c': True}


close_class_map = {}.fromkeys([key for key, value in cplx_class.items() if value is True])
each_cls_data_len = Counter(['a', 'b', 'a', 'b', 'c'])
for key, value in close_class_map.items():
    cls_len = each_cls_data_len[key]
    colse_cls, close_class_gap = None, 0xffffff
    for each in each_cls_data_len:
        if key != each and abs(each_cls_data_len[each] - cls_len) < close_class_gap:
            colse_cls = each
            close_class_gap = abs(each_cls_data_len[each] - cls_len)
    close_class_map[key] = colse_cls
print(close_class_map)

cplx_class = {'1':'a','2':'b','3':'c'}
print(list(cplx_class.values()).index('c'))
