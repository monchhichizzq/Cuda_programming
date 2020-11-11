# Cuda programming

- [Introduction](#introduction)
    - [Why is cuda programming](#why-is-cuda-programming)



## Introduction

### Why we need cuda programming?

Python是当前最流行的编程语言，被广泛应用在深度学习、金融建模、科学和工程计算上。
作为一门解释型语言，它运行速度慢也常常被用户诟病。
著名Python发行商Anaconda公司开发的Numba库为程序员提供了Python版CPU和GPU编程工具，
速度比原生Python快数十倍甚至更多。使用Numba进行GPU编程，你可以享受：

1. Python简单易用的语法；
2. 极快的开发速度；
3. 成倍的硬件加速。


### Why python programming is slow?

Python is slow, as it is interpreted at run-time and not compiled to native code. 
People have tried to get a compiler for Python for quite sometime now but Python being a dynamic langauage, 
most of the efforts have been far from successful.

### Possible solution
If we cannot compile the entire python code, compile parts of it without many changes your existing code.

    