# -*- coding: utf-8 -*-
# @Time    : 2020/11/8 22:44
# @Author  : Zeqi@@
# @FileName: kernelfunction_add.py
# @Software: PyCharm

from numba import cuda
import numpy as np
import math
from time import time

@cuda.jit
def gpu_add(a, b, result, n):
    """
    :param a:  parameter1 加数1
    :param b:  parameter2 加数2
    :param result: 结果 = 加数1 + 加数2
    :param n: 有效线程数
    :return:
    """

    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n:
        result[idx] = a[idx] + b[idx]

def main():
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x
    print('x', x)
    print('y', y)

    gpu_result = np.zeros(n)
    cpu_result = np.zeros(n)

    # 每个block调用1024个threads
    threads_per_block = 1024
    # 决定block数量 ~ 19532
    blocks_per_grid = math.ceil(n / threads_per_block)
    start = time()
    gpu_add[blocks_per_grid, threads_per_block](x, y, gpu_result, n)
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))
    start = time()
    cpu_result = np.add(x, y)
    print("cpu vector add time " + str(time() - start))

    if (np.array_equal(cpu_result, gpu_result)):
        print("result correct")

if __name__ == "__main__":
    main()

    """
    gpu vector add time 0.6893036365509033
    cpu vector add time 0.026129722595214844
    
    numpy 速度更快
    """
