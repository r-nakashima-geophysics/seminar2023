"""
Chapra & Canale (2015) Example 30.2

1 次元の熱伝導方程式を simple 陰解法で解く.

"""

import math
from time import perf_counter
from typing import Final

import numpy as np
from scipy.linalg import solve_banded

# ========== パラメータ ==========
# 境界条件
T_LEFT: Final[float] = 100
T_RIGHT: Final[float] = 50

# 熱伝導率
KAPPA: Final[float] = 0.835

# 時間ステップ
DELTA_TIME: Final[float] = 0.1
END_TIME: Final[float] = 10

# 格子点間隔
DELTA_X: Final[float] = 2

# 境界の座標
LEFT: Final[float] = 0
RIGHT: Final[float] = 10

# ========== パラメータ [ここまで] ==========

LAMBDA: Final[float] = KAPPA * DELTA_TIME / (DELTA_X ** 2)

NUM_TIME: Final[int] = 1 + int(END_TIME/DELTA_TIME)
NUM_X: Final[int] = 1 + int((RIGHT-LEFT)/DELTA_X)

LIN_TIME: Final[np.ndarray] = np.linspace(0, END_TIME, NUM_TIME)
LIN_X: Final[np.ndarray] = np.linspace(LEFT, RIGHT, NUM_X)

SIZE_MAT: Final[int] = NUM_X - 2


def make_mat() -> np.ndarray:
    """
    Ax=b の行列 A をつくる

    Returns
    -----
    mat : ndarray
        Ax=b の行列 A

    """

    mat: np.ndarray = np.zeros((SIZE_MAT, SIZE_MAT))

    for i_diag in range(SIZE_MAT):
        mat[i_diag, i_diag] = 1 + 2 * LAMBDA
    #
    for i_diag in range(SIZE_MAT-1):
        mat[i_diag, i_diag+1] = -LAMBDA
        mat[i_diag+1, i_diag] = -LAMBDA
    #

    return mat
#


def make_banded(mat: np.ndarray) -> np.ndarray:
    """
    Ax=b の行列 A を the diagonal banded form に変換する

    Parameters
    -----
    mat : ndarray
        Ax=b の行列 A

    Returns
    -----
    mat_banded : ndarray
        Ax=b の行列 A の the diagonal banded form

    """

    mat_banded: np.ndarray = np.ndarray((3, SIZE_MAT))

    mat_banded[0, 0] = 0
    mat_banded[1, 0] = mat[0, 0]
    mat_banded[2, 0] = mat[1, 0]
    for i_diag in range(1, SIZE_MAT-1):
        mat_banded[0, i_diag] = mat[i_diag-1, i_diag]
        mat_banded[1, i_diag] = mat[i_diag, i_diag]
        mat_banded[2, i_diag] = mat[i_diag+1, i_diag]
    #
    mat_banded[0, -1] = mat[-2, -1]
    mat_banded[1, -1] = mat[-1, -1]
    mat_banded[2, -1] = 0

    return mat_banded
#


def make_vec(temp: np.ndarray) -> np.ndarray:
    """
    Ax=b のベクトル b をつくる

    Parameters
    -----
    temp : ndarray
        ある時刻での格子点の温度を保存しておく配列

    Returns
    -----
    vec : np.ndarray
        Ax=b のベクトル b

    """

    vec: np.ndarray = temp

    vec[0] += LAMBDA * T_LEFT
    vec[-1] += LAMBDA * T_RIGHT

    return vec
#


def integrate_time(mat_banded: np.ndarray) -> np.ndarray:
    """
    simple 陰解法で熱伝導方程式を時間積分する

    Parameters
    -----
    mat_banded : ndarray
        Ax=b の行列 A の the diagonal banded form

    Returns
    -----
    temp_all : ndarray
        全ての時刻での格子点の温度を保存しておく配列

    """

    temp: np.ndarray = np.zeros_like(LIN_X)
    temp[0] = T_LEFT
    temp[-1] = T_RIGHT

    temp_all: np.ndarray = np.full((NUM_TIME, NUM_X), math.nan)

    vec: np.ndarray
    for i_time in range(NUM_TIME):
        temp_all[i_time, :] = temp
        vec = make_vec(temp[1:-1])
        temp[1:-1] = solve_banded((1, 1), mat_banded, vec)
    #

    return temp_all
#


if __name__ == '__main__':
    TIME_INIT: float = perf_counter()

    matrix: np.ndarray = make_mat()
    matrix_banded: np.ndarray = make_banded(matrix)

    temperature: np.ndarray = integrate_time(matrix_banded)

    np.set_printoptions(precision=7, suppress=True)
    print(temperature)

    TIME_ELAPSED: float = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.5f} s')
#
