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
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx < n :
        result[idx] = a[idx] + b[idx]

def main():
    n = 20000000
    x = np.arange(n).astype(np.int32)
    y = 2 * x

    # 拷贝数据到设备端
    x_device = cuda.to_device(x)
    y_device = cuda.to_device(y)

    # 在显卡设备上初始化一块用于存放GPU计算结果的空间
    # 这很好解释了为什么gpu计算显存时需要考虑特征图大小！！！
    gpu_result = cuda.device_array(n)
    cpu_result = np.empty(n)
    gpu_result_cpu = np.empty(n)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(n / threads_per_block)

    start = time()
    gpu_add[blocks_per_grid, threads_per_block](x, y,  gpu_result_cpu, n)
    cuda.synchronize()
    print("gpu（non-optmi） vector add time " + str(time() - start))

    start = time()
    gpu_add[blocks_per_grid, threads_per_block](x_device, y_device, gpu_result, n) #gpu 需要启动时间，越算越快
    cuda.synchronize()
    print("gpu vector add time " + str(time() - start))

    start = time()
    cpu_result = np.add(x, y) #cpu 不需要启动时间
    print("cpu vector add time " + str(time() - start))

    if (np.array_equal(cpu_result, gpu_result.copy_to_host())):
        print("result correct!")

if __name__ == "__main__":
    main()