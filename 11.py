# -*- coding: utf-8 -*-
# author: sunmengxin
# time: 19-1-9
# file: 11
# description:

from collections import Counter
d =['a','a','b','d','d','c']
counter = Counter(d)
print(counter)
print(list(counter).pop())
print(dict(counter))
print(counter.items() )