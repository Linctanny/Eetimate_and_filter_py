# - * - coding: UTF-8   #
"""
@author: lintan
@file_name:moore_penrose_inverse.py
@time:2023/10/14 20:22
@IDE:PyCharm
@copyright: https://linctanny.github.io/
"""
# 导入相关的包
import numpy as np
import scipy.linalg as sl  # scipy中提供了求广义逆的函数pinv(求实矩阵)与pinvh(求复矩阵)


def column_penrose_inverse(column_vector):
    """
    返回一个列向量的广义逆
    :param column_vector: 待计算的列向量
    :return: 返回一个行向量
    """
    return np.conjugate(column_vector.T) / (float(np.conjugate(column_vector.T) @ column_vector))


def row_penrose_inverse(row_vector):
    """
    返回一个行向量的广义逆
    :param row_vector:
    :return:
    """
    return np.conjugate(row_vector.T) / (float(row_vector @ np.conjugate(row_vector.T)))


# Greville分块算法，列向量进行更新，下面会实现行向量进行更新的一个算法
def Greville_inverse_column(matrix):
    """
    使用Greville分块算法进行求解矩阵的MOORE_PENROSE广义逆
    """
    m, n = matrix.shape
    # 产生一个列表来存储每次迭代的矩阵前几列
    matrix_temp = []
    # 产生一个列表存储广义逆的迭代序列
    matrix_inverse_temp = []
    if n == 1:
        print('该矩阵的列数为1,直接输出一个行向量')
        print(column_penrose_inverse(matrix))
    else:
        print('矩阵列数大于1，进行迭代求解')
        matrix_temp.append(matrix[:, 0:1])  # 初始值矩阵的设置
        matrix_inverse_temp.append(column_penrose_inverse(matrix_temp[0]))  # 迭代逆矩阵的初始值设置
        print('初始化完毕，开始进行迭代')
        for i in range(1, n):
            temp_matrix = matrix[:, :i + 1]
            d_temp = matrix_inverse_temp[i - 1] @ temp_matrix[:, -1:]
            c_temp = temp_matrix[:, -1:] - matrix_temp[i - 1] @ d_temp
            if np.linalg.norm(c_temp) > 1e-10:  # 之后比较零向量直接用范数，不要去用分量是否相等，因为python自带的精度会影响，没有完全等于0的量出现
                k_temp = column_penrose_inverse(c_temp)
            else:
                k_temp = np.conjugate(d_temp.T) @ matrix_inverse_temp[i - 1] / (
                        1 + float(np.conjugate(d_temp.T) @ d_temp))
            matrix_temp.append(temp_matrix)
            # An_inverse_temp = np.vstack((matrix_inverse_temp[i - 1] - d_temp @ k_temp, k_temp))  #矩阵的堆叠
            An_inverse_temp = np.vstack((matrix_inverse_temp[i - 1] - np.dot(d_temp, k_temp), k_temp))
            matrix_inverse_temp.append(An_inverse_temp)
        return matrix_inverse_temp[-1]  # 返回所需要的


def Greville_inverse_row(matrix):
    """
    把Greville方法按照行更新来执行，为之后最小二乘算法的递推形式做准备
    :param matrix: 需要计算的矩阵
    :return: 返回代入矩阵matrix的广义逆矩阵(加号逆)(满足moore-penrose的四个方程)
    """
    m, n = matrix.shape
    # 产生一个列表来存储每次迭代的矩阵前几列
    matrix_temp = []
    # 产生一个列表存储广义逆的迭代序列
    matrix_inverse_temp = []
    if m == 1:
        print('该矩阵的行数为1,直接输出一个列向量')
        print(row_penrose_inverse(matrix))
    else:
        print('矩阵行数大于1，进行迭代求解')
        matrix_temp.append(np.conjugate(matrix[:, 0:1]).T)  # 初始值矩阵的设置
        matrix_inverse_temp.append(row_penrose_inverse(matrix_temp[0]))  # 迭代逆矩阵的初始值设置
        print('初始化完毕，开始进行迭代')
        for i in range(1, n):
            temp_matrix = np.conjugate(matrix[:, :i + 1].T)
            d_temp = np.conjugate(matrix_inverse_temp[i - 1].T) @ np.conjugate(temp_matrix[-1:, :].T)
            c_temp = np.conjugate(temp_matrix[-1:, :].T) - np.conjugate(matrix_temp[i - 1].T) @ d_temp
            if np.linalg.norm(c_temp) > 1e-10:  # 之后比较零向量直接用范数，不要去用分量是否相等，因为python自带的精度会影响，没有完全等于0的量出现
                k_temp = row_penrose_inverse(np.conjugate(c_temp.T))
            else:
                k_temp = matrix_inverse_temp[i - 1] @ d_temp / (1 + float(np.conjugate(d_temp.T) @ d_temp))
            matrix_temp.append(temp_matrix)
            # An_inverse_temp = np.vstack((matrix_inverse_temp[i - 1] - d_temp @ k_temp, k_temp))  #矩阵的堆叠
            An_inverse_temp = np.hstack((matrix_inverse_temp[i - 1] - np.dot(k_temp, np.conjugate(d_temp.T)), k_temp))
            matrix_inverse_temp.append(An_inverse_temp)
        return np.conj(matrix_inverse_temp[-1].T)  # 返回所需要的


def enhanced_gre_inverse(matrix):
    """
    改进Greville求广义逆算法
    :param matrix: 代求矩阵
    :return: 返回matrix的广义逆
    """
    m, n = matrix.shape  # 返回代求矩阵的形状
    P_0 = np.zeros((m, m))  # P_list的初始值零矩阵
    Q_0 = np.eye(m)  # Q_list的初始值设置，m阶单位阵，后面也会经常使用
    # 定义存储数据的列表，P,Q,K的更新计算A的共轭转置的广义逆
    P_list = [P_0]
    Q_list = [Q_0]
    K_list = []
    A_PEN = []  # A的广义逆的存储
    # 下面利用公式开始迭代计算k_temp，P_temp，Q_temp
    for i in range(n):
        a_temp = matrix[:, i:i + 1]
        # Q*a_n+1=0的情形
        if np.linalg.norm(Q_list[i] @ a_temp) < 1e-10:
            k_temp = (P_list[i] @ a_temp) / (1 + np.conj(a_temp.T) @ P_list[i] @ a_temp)
            P_temp = (Q_0 - k_temp @ np.conj(a_temp.T)) @ P_list[i]
            Q_temp = Q_list[i]
        else:
            #  Q*a_n+1！=0的情形
            k_temp = (Q_list[i] @ a_temp) / (np.conj(a_temp.T) @ Q_list[i] @ a_temp)
            P_temp = (Q_0 - k_temp @ np.conj(a_temp.T)) @ P_list[i] @ np.conj(
                (Q_0 - k_temp @ np.conj(a_temp.T)).T) + k_temp @ np.conj(k_temp.T)
            Q_temp = (Q_0 - k_temp @ np.conj(a_temp.T)) @ Q_list[i]
        # 往K_list，P_list，Q_list中输入更新后的K,P,Q值
        K_list.append(k_temp)
        P_list.append(P_temp)
        Q_list.append(Q_temp)
        # 下面开始计算每一步的广义逆
        if i == 0:
            A_PEN_temp = k_temp
            A_PEN.append(A_PEN_temp)
        else:
            m1, n1 = A_PEN[i - 1].shape  # 返回上一步的广义逆的形状，以构造D,来进行每一步的广义逆的更新计算
            D = np.vstack(
                (np.hstack((A_PEN[i - 1], np.zeros((m1, 1)))), np.hstack((np.zeros((1, n1)), [[1]]))))  # D矩阵的构造
            A_PEN_temp = np.hstack((Q_0 - K_list[i] @ np.conj(a_temp.T), K_list[i])) @ D  # 广义逆的结果更新
            A_PEN.append(A_PEN_temp)  # 传入数据列表
    return np.conj(A_PEN[-1].T)  # 返回结果


if __name__ == '__main__':
    A = np.array(np.random.rand(40, 500))  # 输入待求解矩阵
    # print(A@A.T)
    # A = np.array([[1,2,3],[3,4,5],[7,8,9]])
    # A=np.array([[100,2],[3,10]])
    A_penrose_inverse = sl.pinv(A)  # numpy中自带的求解广义逆的函数
    B = Greville_inverse_column(A)  # 书上列更新的Greville递推算法
    C = Greville_inverse_row(A)  # 书上行更新的Greville递推算法
    D = enhanced_gre_inverse(A)  # 基于行更新的改进Greville算法
