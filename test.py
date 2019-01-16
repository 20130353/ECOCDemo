
import numpy as np
#
list = [[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
# slice = np.random.permutation(len(list))  #从list中随机获取5个元素，作为一个片断返回
# print(list(slice[:2]))


import numpy as np
print ("排序列表：", list)
np.random.shuffle(list)
print ("随机排序列表：", list)