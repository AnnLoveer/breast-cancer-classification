import numpy as np
from math import pow, floor


def init_vk(matrix: np.ndarray):
    """
    get the initial cluster centroid values
    :param matrix: image matrix
    :return: a matrix with the centroid values and the location
    Note: vk[0, 0],vk[1, 0] are the centroid values for each cluster
          (vk[0, 1], vk[0, 2]) is the coordinate for vk[0, 0] on matrix
          (vk[1, 1], vk[1, 2]) is the coordinate for vk[1, 0] on matrix

    """
    mat_flatten = matrix.flatten()
    mat_flatten = np.sort(mat_flatten)
    m = mat_flatten.shape[0]
    vk = np.zeros([2, 3], dtype=int)
    point1 = mat_flatten[round(m/3)]
    point2 = mat_flatten[round(m*2/3)]
    vk[0, 0] = point1
    vk[1, 0] = point2
    point1_loc_x = np.where(matrix == point1)[0][0]
    point1_loc_y = np.where(matrix == point1)[1][0]
    point2_loc_x = np.where(matrix == point2)[0][0]
    point2_loc_y = np.where(matrix == point2)[1][0]
    vk[0, 1] = point1_loc_x
    vk[0, 2] = point1_loc_y
    vk[1, 1] = point2_loc_x
    vk[1, 2] = point2_loc_y
    return vk


def get_r_neighborhood(point: tuple, r: int):
    """
    find the r-neighborhood matrix index list at point
    :param point: regional point on matrix
    :param r: number of neighborhood of point
    :return: the r-neighborhood matrix index list at point
    """
    ex_num = int((r-1)/2)
    out = np.zeros([r*r, 2])
    index = 0
    for i in range(r):
        for j in range(r):
            out[index, 0] = point[0] - ex_num + i
            out[index, 1] = point[1] - ex_num + j
            index += 1
    out_ = []
    for i in range(out.shape[0]):
        out_.append(list(out[i, :]))
    for i in range(len(out_)):
        for j in range(len(out_[1])):
            out_[i][j] = int(out_[i][j])
    return out_


def init_r(point1: tuple, point2: tuple):
    """
    calculate the number of overlap point at point1 and point2 in matrix in order to
    find the Neighborhood value for maximum overlap rate
    :param point1: the first point
    :param point2: the second point
    :return: find R
    """
    mat1_3 = get_r_neighborhood(point1, r=3)
    mat1_5 = get_r_neighborhood(point1, r=5)
    mat1_7 = get_r_neighborhood(point1, r=7)
    mat2_3 = get_r_neighborhood(point2, r=3)
    mat2_5 = get_r_neighborhood(point2, r=5)
    mat2_7 = get_r_neighborhood(point2, r=7)
    out_3 = len([l for l in mat1_3 if l in mat2_3])
    out_5 = len([l for l in mat1_5 if l in mat2_5])
    out_7 = len([l for l in mat1_7 if l in mat2_7])
    maxim = max(out_3, out_5, out_7)
    if out_3 == maxim:
        out = 3
    elif out_5 == maxim:
        out = 5
    else:
        out = 7
    return out


def object_func(matrix: np.ndarray, vk: np.ndarray, m: int, alpha: float):
    r = init_r((vk[0, 1], vk[0, 2]), (vk[1, 1], vk[1, 2]))
    u = membership_func(matrix, vk, r, m, alpha)
    m_row, m_col, cls = u.shape
    j_e = 0
    j_h = 0
    for i in range(m_row):
        for j in range(m_col):
            for k in range(cls):
                j_e += u[i, j, k] * pow(np.linalg.norm(matrix[i, j] - vk[k, 0]), 2)
                j_h += u[i, j, k] * pow(hausdorff_dist(matrix, np.array(get_r_neighborhood((vk[k, 1], vk[k, 2]), init_r((i, j), (vk[k, 1], vk[k, 2]))))), 2)
    return j_e + j_h


def hausdorff_dist(A: np.ndarray, B: np.ndarray):
    """
    calculate the hausdorff distance between the values of matrix A and the values of matrix B
    :param A: matrix A
    :param B: matrix B
    :return: value of hausdorff distance
    """
    assert A.size > 0
    assert B.size > 0
    row_a, col_a = A.shape
    row_b, col_b = B.shape
    inf_A = np.zeros([row_b, col_b])
    inf_B = np.zeros([row_a, col_a])
    inf_AB = np.zeros([row_a, col_a])
    inf_BA = np.zeros([row_b, col_b])
    for i in range(row_a):
        for j in range(col_a):
            for k in range(row_b):
                for l in range(col_b):
                    inf_A[k, l] = np.abs(B[k, l] - A[i, j])
            inf_AB[i, j] = np.min(inf_A)
    h_AB = np.max(inf_AB)
    for k in range(row_b):
        for l in range(col_b):
            for i in range(row_a):
                for j in range(col_a):
                    inf_B[i, j] = np.abs(A[i, j] - B[k, l])
            inf_BA[k, l] = np.min(inf_B)
    h_BA = np.max(inf_BA)
    return max(h_AB, h_BA)


def membership_func(matrix: np.ndarray, centroid_matrix: np.ndarray, r: int, m: int, alpha: float):
    """
    get and update the member function by this function
    :param matrix: the R-neighborhood original matrix
    :param centroid_matrix: the cluster centroid value and coordination
    :param m: membership ratio
    :param alpha: weight ratio
    :return: the membership matrix
    """
    m_row, m_col = matrix.shape
    cluster_num = centroid_matrix.shape[0]
    u = np.zeros([m_row, m_col, cluster_num], dtype=float)
    dividend = np.zeros([m_row, m_col, cluster_num], dtype=float)
    divisor = np.zeros([m_row, m_col], dtype=float)
    for i in range(m_row):
        for j in range(m_col):
            for l in range(cluster_num):
                vk_mat = get_r_neighborhood((centroid_matrix[l, 1], centroid_matrix[l, 2]), r)
                vk_mat = np.array(vk_mat)
                dividend[i, j, l] = pow(pow(np.linalg.norm(matrix[i, j] - centroid_matrix[l, 0], ord=2), 2)
                               + alpha * pow(hausdorff_dist(matrix, vk_mat), 2),
                               -1/(m-1))
                divisor[i, j] += dividend[i, j, l]
    for i in range(m_row):
        for j in range(m_col):
            for l in range(cluster_num):
                u[i, j, l] = dividend[i, j, l]/divisor[i, j]
    return u


def update_vk(matrix: np.ndarray, u: np.ndarray, m: int):
    m_row, m_col = matrix.shape
    u_cen = u.shape[2]
    vk = np.zeros([u_cen, 3])
    dividend = 0
    divisor = 0
    for c in range(u_cen):
        for i in range(m_row):
            for j in range(m_col):
                dividend += u[i, j, c] * matrix[i, j]
                divisor += u[i, j, c]
        vk[c, 0] = floor(dividend/divisor)
    point1_loc_x = np.where(matrix == vk[0, 0])[0][0]
    point1_loc_y = np.where(matrix == vk[0, 0])[1][0]
    point2_loc_x = np.where(matrix == vk[1, 0])[0][0]
    point2_loc_y = np.where(matrix == vk[1, 0])[1][0]
    vk[0, 1] = point1_loc_x
    vk[0, 2] = point1_loc_y
    vk[1, 1] = point2_loc_x
    vk[1, 2] = point2_loc_y
    return vk


def converge_if(threshold: float, object_val: float, update_obj_val: float):
    if abs(update_obj_val - object_val) <= threshold:
        return 1
    else:
        return 0
