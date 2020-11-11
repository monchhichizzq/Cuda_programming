# -*- coding: utf-8 -*-
# @Time    : 2020/11/8 22:16
# @Author  : Zeqi@@
# @FileName: numba_test2.py
# @Software: PyCharm

import numpy as np
from numba import cuda


def cpu_print(N):
    for i in range(0, N):
        print(i)

@cuda.jit
def gpu_print(N):
    # 8线程同时进行
    print(cuda.threadIdx.x, cuda.blockIdx.x, cuda.blockDim.x)
    # print('thread: {0}, block: {1}, block_dim: {2}'.format(cuda.threadIdx.x, cuda.blockIdx.x, cuda.blockDim.x))
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if (idx < N): #如果N设为5， 依旧启动8线程，但只有5线程进行计算
        print(idx)

def main():
    print("gpu print:")
    gpu_print[2, 4](5)
    cuda.synchronize()
    print("cpu print:")
    cpu_print(8)

if __name__ == "__main__":
    main()