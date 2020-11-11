# -*- coding: utf-8 -*-
# @Time    : 2020/11/8 3:37
# @Author  : Zeqi@@
# @FileName: numba_test.py
# @Software: PyCharm

from numba import cuda
print(cuda.gpus) # CUDA_VISIBLE_DEVICES='5' python example.py


"""
1. 使用from numba import cuda引入cuda库
2. 在GPU函数上添加@cuda.jit装饰符，表示该函数是一个在GPU设备上运行的函数，GPU函数又被称为核函数。
    主函数调用GPU核函数时，需要添加如[1, 2]这样的执行配置，这个配置是在告知GPU以多大的并行粒度同时进行计算。
    gpu_print[1, 2]()表示同时开启2个线程并行地执行gpu_print函数，函数将被并行地执行2次。
    下文会深入探讨如何设置执行配置。
3. GPU核函数的启动方式是异步的：启动GPU函数后，CPU不会等待GPU函数执行完毕才执行下一行代码。
    必要时，需要调用cuda.synchronize()，告知CPU等待GPU执行完核函数后，再进行CPU端后续计算。
    这个过程被称为同步，也就是GPU执行流程图中的红线部分。如果不调用cuda.synchronize()函数，执行结果也将改变，"print by cpu.将先被打印。
    虽然GPU函数在前，但是程序并没有等待GPU函数执行完，而是继续执行后面的cpu_print函数，
    由于CPU调用GPU有一定的延迟，反而后面的cpu_print先被执行，因此cpu_print的结果先被打印了出来。
"""

def cpu_print():
    print("print by cpu.")

# 在GPU函数上添加@cuda.jit装饰符
# 表示该函数是一个在GPU设备上运行的函数，GPU函数又被称为核函数。
@cuda.jit
def gpu_print():
    # GPU核函数
    print("print by gpu.")

def main():
    gpu_print[1, 2]()   # 同时开启2个线程并行地执行gpu_print函数，函数将被并行地执行2次
    cuda.synchronize() # 告知CPU等待GPU执行完核函数后，再进行CPU端后续计算
    cpu_print()

def main_2():
    gpu_print[2,4]()    # 把前面的程序改为并行执行8次：可以用2个block，每个block中有4个thread
    cuda.synchronize()  # 告知CPU等待GPU执行完核函数后，再进行CPU端后续计算
    cpu_print()




if __name__ == "__main__":
    main_2()