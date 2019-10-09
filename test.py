import numpy as np
import cv2 as cv


def init_vk(matrix: np.ndarray):
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


img = r'F:\BreastCancer\TrainSetOrigional(536)\0001.png'
a = cv.imread(img, 0)
vk = init_vk(a)
p1 = (vk[0, 1], vk[0, 2])
p2 = (vk[1, 1], vk[1, 2])
out = init_r(p1, p2)



