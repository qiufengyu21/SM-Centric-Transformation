# SM-Centric-Transformation
A Clang tool that transform any CUDA code to the SM-Centric form

# Paper
[ICS'15] "Enabling and Exploiting Flexible Task Assignment on GPU through SM-Centric Program Transformations", Bo Wu, Guoyang Chen, Dong Li, Xipeng Shen, Jeffrey Vetter, ACM International Conference on Supercomputing, Newport Beach, CA, 2015. (25% acceptance rate) [PDF](https://people.engr.ncsu.edu/xshen5/Publications/ics15.pdf)

# Example Transformation
## Matrix Addition: 

[Original](https://github.com/qiufengyu21/SM-Centric-Transformation/blob/master/Example/MA/matrixAdd_org.cu) 

[SMC form](https://github.com/qiufengyu21/SM-Centric-Transformation/blob/master/Example/MA/matrixAdd_smc.cu)

## Matrix Multiplication:

[Original](https://github.com/qiufengyu21/SM-Centric-Transformation/blob/master/Example/MM/matrixMul_org.cu) 

[SMC form](https://github.com/qiufengyu21/SM-Centric-Transformation/blob/master/Example/MM/matrixMul_smc.cu)
