"""
Chapra & Canale (2015) Example 30.5

2 次元の熱伝導方程式を ADI 法で解く.
Dirichlet 境界条件.

Notes
-----
コマンドライン引数以外のパラメータは以下に記述されている.

"""

import copy
import math
from time import perf_counter
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from scipy.linalg import solve_banded

# ========== パラメータ ==========
# 境界条件
T_TOP: Final[float] = 100
T_RIGHT: Final[float] = 50
T_BOTTOM: Final[float] = 0
T_LEFT: Final[float] = 75

# 一様な初期条件
T_INIT: Final[float] = 0

# 熱伝導率
KAPPA: Final[float] = 0.835

# 時間ステップ
DELTA_TIME: Final[float] = 0.25
END_TIME: Final[float] = 300

# 格子点間隔
DELTA_X: Final[float] = 1
DELTA_Y: Final[float] = 1

# 境界の座標
TOP: Final[float] = 40
RIGHT: Final[float] = 40
BOTTOM: Final[float] = 0
LEFT: Final[float] = 0

# ========== パラメータ [ここまで] ==========

LAMBDA_X: Final[float] \
    = 2 * KAPPA * DELTA_TIME / (DELTA_X**2)
LAMBDA_Y: Final[float] \
    = 2 * KAPPA * DELTA_TIME / (DELTA_Y**2)

NUM_TIME: Final[int] = 1 + int(END_TIME/DELTA_TIME)
LIN_TIME: Final[np.ndarray] = np.linspace(0, END_TIME, NUM_TIME)

NUM_X: Final[int] = 1 + int((RIGHT-LEFT)/DELTA_X)
NUM_Y: Final[int] = 1 + int((TOP-BOTTOM)/DELTA_Y)

SIZE_MAT_X: Final[int] = NUM_X - 2
SIZE_MAT_Y: Final[int] = NUM_Y - 2

LIN_X: Final[np.ndarray] = np.linspace(LEFT, RIGHT, NUM_X)
LIN_Y: Final[np.ndarray] = np.linspace(BOTTOM, TOP, NUM_Y)

GRID_X: Final[np.ndarray]
GRID_Y: Final[np.ndarray]
GRID_X, GRID_Y = np.meshgrid(LIN_X, LIN_Y)


def make_animation(fig_bundle: tuple, contf):
    """
    アニメーションを作成

    Parameters
    -----
    fig_bundle : tuple
        figure 関連の tuple
    contf :
        コンター

    Returns
    -----
    ani :
        アニメーション

    """

    figure, axis, frames = fig_bundle

    axis.set_xlim(LEFT, RIGHT)
    axis.set_ylim(BOTTOM, TOP)

    axis.set_xlabel(r'$x$', fontsize=18)
    axis.set_ylabel(r'$y$', fontsize=18)

    axis.tick_params(labelsize=16)
    axis.minorticks_on()

    axis.set_aspect('equal')

    figure.tight_layout()

    figure.subplots_adjust(right=0.91, wspace=0.25)
    axpos = axis.get_position()
    cbar_ax = figure.add_axes([0.81, axpos.y0, 0.01, axpos.height])

    cbar = figure.colorbar(contf, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label=r'$T [\mathrm{^\circ C}]$', size=18)

    ani = animation.ArtistAnimation(figure, frames, interval=25)

    return ani
#


def init_temp() -> np.ndarray:
    """
    格子点の温度を保存しておく配列を作成する

    Returns
    -----
    temp : ndarray
        格子点の温度を保存しておく配列

    """

    temp: np.ndarray = np.full((NUM_Y, NUM_X), math.nan)

    t_init: float = 0

    for i_x in range(NUM_X):
        for i_y in range(NUM_Y):

            if (1 <= i_x <= SIZE_MAT_X) and (1 <= i_y <= SIZE_MAT_Y):
                temp[i_y, i_x] = t_init
            #

            if (1 <= i_x <= SIZE_MAT_X) and (i_y == NUM_Y - 1):
                temp[i_y, i_x] = T_TOP
            elif (i_x == NUM_X - 1) and (1 <= i_y <= SIZE_MAT_Y):
                temp[i_y, i_x] = T_RIGHT
            elif (1 <= i_x <= SIZE_MAT_X) and (i_y == 0):
                temp[i_y, i_x] = T_BOTTOM
            elif (i_x == 0) and (1 <= i_y <= SIZE_MAT_Y):
                temp[i_y, i_x] = T_LEFT
            #
        #
    #

    return temp
#


def make_mat(str_xy: str) -> np.ndarray:
    """
    Ax=b の行列 A をつくる

    Parameters
    -----
    str_xy : str
        'x' or 'y'

    Returns
    -----
    mat : ndarray
        Ax=b の行列 A

    """

    size_mat: int = int()
    lambda_implicit: float = float()
    if str_xy == 'x':
        size_mat = SIZE_MAT_X
        lambda_implicit = LAMBDA_X
    elif str_xy == 'y':
        size_mat = SIZE_MAT_Y
        lambda_implicit = LAMBDA_Y
    #

    mat: np.ndarray = np.zeros((size_mat, size_mat))

    for i_diag in range(size_mat):
        mat[i_diag, i_diag] = 2 * (1+lambda_implicit)
    #
    for i_diag in range(size_mat-1):
        mat[i_diag, i_diag+1] = -lambda_implicit
        mat[i_diag+1, i_diag] = -lambda_implicit
    #

    return mat
#


def make_banded(mat: np.ndarray, str_xy: str) -> np.ndarray:
    """
    Ax=b の行列 A を the diagonal banded form に変換する

    Parameters
    -----
    mat : ndarray
        Ax=b の行列 A
    str_xy : str
        'x' or 'y'

    Returns
    -----
    mat_banded : ndarray
        Ax=b の行列 A の the diagonal banded form

    """

    size_mat: int = int()
    if str_xy == 'x':
        size_mat = SIZE_MAT_X
    elif str_xy == 'y':
        size_mat = SIZE_MAT_Y
    #

    mat_banded: np.ndarray = np.ndarray((3, size_mat))

    mat_banded[0, 0] = 0
    mat_banded[1, 0] = mat[0, 0]
    mat_banded[2, 0] = mat[1, 0]
    for i_diag in range(1, size_mat-1):
        mat_banded[0, i_diag] = mat[i_diag-1, i_diag]
        mat_banded[1, i_diag] = mat[i_diag, i_diag]
        mat_banded[2, i_diag] = mat[i_diag+1, i_diag]
    #
    mat_banded[0, -1] = mat[-2, -1]
    mat_banded[1, -1] = mat[-1, -1]
    mat_banded[2, -1] = 0

    return mat_banded
#


def main_loop(temp: np.ndarray,
              mat_banded_x: np.ndarray, mat_banded_y: np.ndarray,
              fig_bundle: tuple):
    """
    ADI 法で熱伝導方程式を時間積分し, コンターを作成

    Parameters
    -----
    temp : ndarray
        格子点の温度 (初期値) を保存しておく配列
    mat_banded_x : ndarray
        Ax=b の行列 A の the diagonal banded form
    mat_banded_y : ndarray
        Ax=b の行列 A の the diagonal banded form
    fig_bundle : tuple
        figure 関連の tuple

    Returns
    -----
    fig_bundle : tuple
        figure 関連の tuple
    contf :
        コンター

    """

    axis: plt.Axes = fig_bundle[1]
    frames: list = fig_bundle[2]

    time: float
    contf = []
    for i_time in range(NUM_TIME):
        time = LIN_TIME[i_time]

        print('t = ', time)
        print(temp)

        temp = wrapper_adi(temp, mat_banded_x, 'y')
        if i_time == 0:
            print('t =', DELTA_TIME * 0.5)
            print(temp)
        #
        temp = wrapper_adi(temp, mat_banded_y, 'x')

        contf = make_contourf(temp, axis)

        text = axis.text(14, 42, f"$t = {time:.2f}$", size=18)

        frames.append(contf.collections + [text])
    #

    return fig_bundle, contf
#


def wrapper_adi(temp: np.ndarray,
                mat_banded: np.ndarray, str_xy: str) -> np.ndarray:
    """
    ADI 法の時間 0.5 ステップ分

    Parameters
    -----
    temp : ndarray
        格子点の温度 (初期値) を保存しておく配列
    mat_banded_x : ndarray
        Ax=b の行列 A の the diagonal banded form
    str_xy : str
        'x' or 'y'

    Returns
    -----
    temp_new : ndarray
        格子点の温度 (収束した値) を保存しておく配列

    """

    num_sol: int = int()
    num_grid: int = int()
    if str_xy == 'x':
        num_sol = SIZE_MAT_X
        num_grid = NUM_Y
    elif str_xy == 'y':
        num_sol = SIZE_MAT_Y
        num_grid = NUM_X
    #
    temp_new = copy.copy(temp)
    temp_explicit: np.ndarray \
        = np.array([np.full(num_sol+2, math.nan), ] * 3)

    for i_grid in range(1, num_grid-1):
        if str_xy == 'x':
            temp_explicit[0] = temp[i_grid-1, :]
            temp_explicit[1] = temp[i_grid, :]
            temp_explicit[2] = temp[i_grid+1, :]
        elif str_xy == 'y':
            temp_explicit[0] = temp[:, i_grid-1]
            temp_explicit[1] = temp[:, i_grid]
            temp_explicit[2] = temp[:, i_grid+1]
        #

        vec = make_vec(temp_explicit, num_sol, str_xy)

        if str_xy == 'x':
            temp_new[i_grid, 1:-1] \
                = solve_banded((1, 1), mat_banded, vec)
        elif str_xy == 'y':
            temp_new[1:-1, i_grid] \
                = solve_banded((1, 1), mat_banded, vec)
        #
    #

    return temp_new
#


def make_vec(temp: np.ndarray, size_vec: int, str_xy: str) \
        -> np.ndarray:
    """
    Ax=b のベクトル b をつくる

    Parameters
    -----
    temp : ndarray
        ある時刻での格子点の温度を保存しておく配列
    size_vec : int
         Ax=b のベクトル b のサイズ
    str_xy : str
        'x' or 'y'

    Returns
    -----
    vec : ndarray
        Ax=b のベクトル b

    """

    lambda_explicit: float = float()
    lambda_implicit: float = float()
    if str_xy == 'x':
        lambda_explicit = LAMBDA_Y
        lambda_implicit = LAMBDA_X
    elif str_xy == 'y':
        lambda_explicit = LAMBDA_X
        lambda_implicit = LAMBDA_Y
    #

    vec: np.ndarray = np.zeros(size_vec)

    for i_col in range(size_vec):
        vec[i_col] = lambda_explicit*temp[0][i_col+1] \
            + 2*(1-lambda_explicit)*temp[1][i_col+1] \
            + lambda_explicit*temp[2][i_col+1]
    #

    vec[0] += lambda_implicit * temp[1][0]
    vec[-1] += lambda_implicit * temp[1][-1]

    return vec
#


def make_contourf(temp: np.ndarray, axis):
    """
    温度分布のコンターを作る

    Parameters
    -----
    temp : ndarray
        格子点の温度を保存しておく配列
    axis :
        Axes オブジェクト

    Return
    -----
    contf :
        コンター

    """

    t_max: float = range_temp()[0]
    t_min: float = range_temp()[1]

    t_step: float
    if t_max-t_min < 100:
        t_step = 5
    else:
        t_step = 10
    #
    levels: np.ndarray = np.arange(t_min, t_max+1, t_step)

    contf = axis.contourf(GRID_X, GRID_Y, temp,
                          vmin=t_min, vmax=t_max,
                          levels=levels, cmap='inferno')

    return contf
#


def range_temp() -> tuple[float, float]:
    """
    とりうる温度の範囲を決める

    Returns
    -----
    t_max : float
        境界条件の温度の最大値
    t_min : float
        境界条件の温度の最小値

    """

    t_bc: list[float] = [T_TOP, T_RIGHT, T_BOTTOM, T_LEFT]

    t_max: float = max(t_bc)
    t_min: float = min(t_bc)

    return t_max, t_min
#


if __name__ == '__main__':
    TIME_INIT: float = perf_counter()

    plt.rcParams['text.usetex'] = True
    np.set_printoptions(precision=7, suppress=True)

    temperature: np.ndarray
    temperature = init_temp()

    matrix_x: np.ndarray = make_mat('x')
    matrix_y: np.ndarray = make_mat('y')
    matrix_banded_x: np.ndarray = make_banded(matrix_x, 'x')
    matrix_banded_y: np.ndarray = make_banded(matrix_y, 'y')

    fig, ax = plt.subplots()
    frame_list: list = []
    bundle: tuple = (fig, ax, frame_list)

    bundle, contour = main_loop(
        temperature, matrix_banded_x, matrix_banded_y, bundle)
    anim = make_animation(bundle, contour)

    anim.save('ani.mp4', writer="ffmpeg", dpi=300)

    TIME_ELAPSED: float = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.5f} s')

    plt.show()
#
