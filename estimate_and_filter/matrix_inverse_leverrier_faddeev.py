# - * - coding: UTF-8   #
"""
@author: lintan
@file_name:matrix_inverse_leverrier_faddeev.py
@time:2023/09/30 19:18
@IDE:PyCharm
@copyright: https://linctanny.github.io/
"""
import numpy as np


def Feverrier_Faddeev(mat):
    """
    可逆矩阵求逆的有限迭代算法
    :param mat: 需要求逆的矩阵
    :return:
    """
    if np.linalg.det(mat) == 0:
        print('The matrix is not invertible, please change another one')
    else:
        m, n = mat.shape
        B = [np.eye(n), mat]
        a = [0, np.sum(np.diagonal(mat))]
        I=np.eye(n)
        for i in range(2, n+1):
            B_temp = mat@(B[i - 1] - a[i - 1]*I)  # @这个符号相当于Matlab中的用于矩阵乘法的*
            a_temp = (1 / i) * np.sum(np.diagonal(B_temp))  #计算a的更新数字
            B.append(B_temp)   # 对矩阵B进行更新
            a.append(a_temp)   # 对a值进行更新
        return (B[n - 1] - a[n - 1]*I) / a[n], B, a


if __name__ == '__main__':
    mat = np.array([[2, -1,1], [-1, 2,1],[1,-1,2]])
    mat_inv, B, a = Feverrier_Faddeev(mat)
    print(mat_inv)
