# -*- coding: utf-8 -*-
# @Time    : 2020/11/8 3:17
# @Author  : Zeqi@@
# @FileName: numpy_numba.py
# @Software: PyCharm

import timeit
# timeit是Python标准库内置的小工具，可以快速测试小段代码的性能
import numpy as np
from numba import jit


nobs = 10000

def proc_numpy(x,y,z):

   x = x*2 - ( y * 55 )      # these 4 lines represent use cases
   y = x + y*2               # where the processing time is mostly
   z = x + y + 99            # a function of, say, 50 to 200 lines
   z = z * ( z - .88 )       # of fairly simple numerical operations

   return z

@jit
def proc_numba(xx,yy,zz):
   for j in range(nobs):     # as pointed out by Llopis, this for loop
      x, y = xx[j], yy[j]    # is not needed here.  it is here by
                             # accident because in the original benchmarks
      x = x*2 - ( y * 55 )   # I was doing data creation inside the function
      y = x + y*2            # instead of passing it in as an array
      z = x + y + 99         # in any case, this redundant code seems to
      z = z * ( z - .88 )    # have something to do with the code running
                             # faster.  without the redundant code, the
      zz[j] = z              # numba and numpy functions are exactly the same.
   return zz

x = np.random.randn(nobs)
y = np.random.randn(nobs)
z = np.zeros(nobs)
res_numpy = proc_numpy(x,y,z)

z = np.zeros(nobs)
res_numba = proc_numba(x,y,z)

print(np.all( res_numpy == res_numba))
print(timeit.timeit(stmt= 'proc_numpy', setup='from __main__ import proc_numpy'))
print(timeit.timeit(stmt= 'proc_numba', setup='from __main__ import proc_numba'))

# timeit.timeit(stmt, setup,timer, number)
# 参数说明：
# stmt: statement的缩写，你要测试的代码或者语句，纯文本，默认值是 "pass"
# setup: 在运行stmt前的配置语句，纯文本，默认值也是 "pass"
# timer: 计时器，一般忽略这个参数
# number: stmt执行的次数，默认是1000000，一百万