import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
              1,2,3,4
              5,6,,8
              0,1,2,'''

df = pd.read_csv(StringIO(csv_data),header=None).T
print('csv_data df:')
print(df)

#统计为空的数目
print('Na number:' %df.isnull().sum())

#丢弃空的
print('drop na:')
df = df.dropna()
print(df)
#
# from sklearn.preprocessing import Imputer
# # axis=0 列  axis = 1 行
# imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
# imr.fit(df) # fit 构建得到数据
# imputed_data = imr.transform(df.values) #transform 使用均值填充数据的缺失值
# print('transformed data:')
# print(imputed_data)

df.columns = ['F1','F2','F3','label']

# 1, 利用LabelEncoder类快速编码,但此时对color并不适合,
# # 看起来,好像是有大小的
# from sklearn.preprocessing import LabelEncoder
# class_le = LabelEncoder()
# df['F1'] = class_le.fit_transform(df['F1'].values)
# print('label Encoder')
# print(df)

#2, 映射字典将类标转换为整数
# import numpy as np
# class_mapping = {label: idx for idx, label in enumerate(np.unique(df['F1']))}
# df['F1'] = df['F1'].map(class_mapping)
# print('mapping')
# print(df)


#3,处理1不适用的
#利用创建一个新的虚拟特征
from sklearn.preprocessing import OneHotEncoder
pf = pd.get_dummies(df[['F1']])
df = pd.concat([df, pf], axis=1)
df.drop(['F1'], axis=1, inplace=True)
print(df)
