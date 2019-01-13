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
from collections import Counter
import numpy as np
from ECOCDemo.ECOC import Matrix_tool as MT
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs

dict = {'a':1,'b':3,'c':4}
sorted_ = sorted(dict.items(),key=lambda x:x[1])
sorted_.reverse()
print(sorted_)
for each in sorted_:
    print(each[0],each[1])