"""
Chapra & Canale (2015) Example 29.3

2 次元のラプラス方程式 (加熱された板の温度分布) を Gauss-Seidel 法で解く.
Dirichlet 境界条件 or Neumann 境界条件, 熱源入り.

T_{i,j} = ( T_{i+1,j}/(\\delta x)^2 + T_{i-1,j}/(\\delta x)^2
    + T_{i,j+1}/(\\delta y)^2 + T_{i,j-1}/(\\delta y)^2
    + f_{i,j} )
    / ( 2/(\\delta x)^2 + 2/(\\delta y)^2 )

Notes
-----
コマンドライン引数以外のパラメータは以下に記述されている.

"""

import logging
import math
from time import perf_counter
from typing import Final

import matplotlib.pyplot as plt
import numpy as np
from numba import njit

# ========== パラメータ ==========
# 境界条件
# D: Dirichlet 境界条件 (T)
# N: Neumann 境界条件 (dT/dx or dT/dy)
T_TOP: Final[tuple[float, str]] = (100, 'D')
T_RIGHT: Final[tuple[float, str]] = (50, 'D')
T_BOTTOM: Final[tuple[float, str]] = (0, 'N')
T_LEFT: Final[tuple[float, str]] = (75, 'D')

# 加速パラメータ
LAMBDA: Final[float] = 1.5

# 停止基準
CRITERION: Final[float] = 0.0001

# 最大反復回数
MAX_ITER: Final[float] = 2000

# 格子点間隔
DELTA_X: Final[float] = 0.01
DELTA_Y: Final[float] = 0.01

# 境界の座標
TOP: Final[float] = 1
RIGHT: Final[float] = 1
BOTTOM: Final[float] = 0
LEFT: Final[float] = 0


@njit
def heat_source(pos_x: float, pos_y: float) -> float:
    """
    熱源を指定する関数
    """

    heat: float

    heat = 0

    # center_x: float = (RIGHT-LEFT)/2
    # center_y: float = (TOP-BOTTOM)/2
    # sigma_x: float = (RIGHT-LEFT)/10
    # sigma_y: float = (TOP-BOTTOM)/10
    # norm: float = 1 / 2*math.pi*sigma_x*sigma_y
    # heat = 100000 * math.exp(
    #     - (pos_x-center_x)**2/(2*(sigma_x**2))
    #     - (pos_y-center_y)**2/(2*(sigma_y**2))
    # ) * norm

    return heat
#
# ========== パラメータ [ここまで] ==========


NUM_GRID_X: Final[int] = int((RIGHT-LEFT)/DELTA_X) - 1
NUM_GRID_Y: Final[int] = int((TOP-BOTTOM)/DELTA_Y) - 1

NUM_COL: Final[int] = NUM_GRID_X + 2
NUM_ROW: Final[int] = NUM_GRID_Y + 2

LIN_X: Final[np.ndarray] = np.linspace(LEFT, RIGHT, NUM_COL)
LIN_Y: Final[np.ndarray] = np.linspace(BOTTOM, TOP, NUM_ROW)

GRID_X: Final[np.ndarray]
GRID_Y: Final[np.ndarray]
GRID_X, GRID_Y = np.meshgrid(LIN_X, LIN_Y)


def plot_temp(temp: np.ndarray):
    """
    温度分布を図示する

    Parameters
    -----
    temp : ndarray
        格子点の温度を保存しておく配列

    Return
    -----
    fig :
        figure オブジェクト

    """

    fig = plt.figure()
    axis = fig.add_subplot(111)

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

    axis.set_xlim(LEFT, RIGHT)
    axis.set_ylim(BOTTOM, TOP)

    axis.set_xlabel(r'$x$', fontsize=18)
    axis.set_ylabel(r'$y$', fontsize=18)

    axis.tick_params(labelsize=16)
    axis.minorticks_on()

    axis.set_aspect('equal')

    fig.tight_layout()

    fig.subplots_adjust(right=0.91, wspace=0.25)
    axpos = axis.get_position()
    cbar_ax = fig.add_axes([0.81, axpos.y0, 0.01, axpos.height])

    cbar = fig.colorbar(contf, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label=r'$T [\mathrm{^\circ C}]$', size=18)

    return fig
#


def init_temp() -> tuple[np.ndarray, np.ndarray]:
    """
    格子点の温度を保存しておく配列を作成する

    Returns
    -----
    temp : ndarray
        格子点の温度を保存しておく配列
    grid : ndarray
        計算する格子点の位置を保存しておく配列

    """

    temp: np.ndarray = np.full((NUM_ROW, NUM_COL), math.nan)
    grid: np.ndarray = np.full((NUM_ROW, NUM_COL), False)

    t_init: float = range_temp()[2]

    for i_x in range(NUM_COL):
        for i_y in range(NUM_ROW):

            if (1 <= i_x <= NUM_GRID_X) and (1 <= i_y <= NUM_GRID_Y):
                temp[i_y, i_x] = t_init
                grid[i_y, i_x] = True
            #

            if (1 <= i_x <= NUM_GRID_X) and (i_y == NUM_ROW - 1):
                if T_TOP[1] == 'D':
                    temp[i_y, i_x] = T_TOP[0]
                elif T_TOP[1] == 'N':
                    temp[i_y, i_x] = t_init
                    temp[i_y, 0] = T_LEFT[0]
                    temp[i_y, -1] = T_RIGHT[0]
                    grid[i_y, i_x] = True
                else:
                    logger.error('境界条件の設定が間違っています.')
            elif (i_x == NUM_COL - 1) and (1 <= i_y <= NUM_GRID_Y):
                if T_RIGHT[1] == 'D':
                    temp[i_y, i_x] = T_RIGHT[0]
                elif T_RIGHT[1] == 'N':
                    temp[i_y, i_x] = t_init
                    temp[i_y, -1] = T_TOP[0]
                    temp[i_y, 0] = T_BOTTOM[0]
                    grid[i_y, i_x] = True
                else:
                    logger.error('境界条件の設定が間違っています.')
            elif (1 <= i_x <= NUM_GRID_X) and (i_y == 0):
                if T_BOTTOM[1] == 'D':
                    temp[i_y, i_x] = T_BOTTOM[0]
                elif T_BOTTOM[1] == 'N':
                    temp[i_y, i_x] = t_init
                    temp[i_y, 0] = T_LEFT[0]
                    temp[i_y, -1] = T_RIGHT[0]
                    grid[i_y, i_x] = True
                else:
                    logger.error('境界条件の設定が間違っています.')
            elif (i_x == 0) and (1 <= i_y <= NUM_GRID_Y):
                if T_LEFT[1] == 'D':
                    temp[i_y, i_x] = T_LEFT[0]
                elif T_LEFT[1] == 'N':
                    temp[i_y, i_x] = t_init
                    temp[i_y, -1] = T_TOP[0]
                    temp[i_y, 0] = T_BOTTOM[0]
                    grid[i_y, i_x] = True
                else:
                    logger.error('境界条件の設定が間違っています.')
            #
        #
    #

    return temp, grid
#


def range_temp() -> tuple[float, float, float]:
    """
    とりうる温度の範囲を決める

    Returns
    -----
    t_max : float
        境界条件の温度の最大値
    t_min : float
        境界条件の温度の最小値
    t_mean : float
        境界条件の温度の平均値

    """

    t_bc: list = [
        bc[0] for bc in (T_TOP, T_RIGHT, T_BOTTOM, T_LEFT) if bc[1] == 'D']

    t_max: float = max(t_bc)
    t_min: float = min(t_bc)

    t_mean: float = sum(t_bc) / len(t_bc)

    return t_max, t_min, t_mean
#


def main_loop(temp: np.ndarray, grid: np.ndarray):
    """
    Gauss-Seidel 法の反復を行う

    Parameters
    -----
    temp : ndarray
        格子点の温度 (初期値) を保存しておく配列
    grid : ndarray
        計算する格子点の位置を保存しておく配列

    Returns
    -----
    temp : ndarray
        格子点の温度 (収束した値) を保存しておく配列

    """

    for i_itr in range(MAX_ITER):
        temp, stop, max_error = wrapper_gauss_seidel(temp, grid)

        print('====================')
        print(f'反復回数: {i_itr+1}')
        print(f'最大誤差: {max_error*100:.4f} %')

        if stop:
            break
        #
    #

    return temp
#


def wrapper_gauss_seidel(temp: np.ndarray, grid: np.ndarray) \
        -> tuple[np.ndarray, bool, float]:
    """
    Gauss-Seidel 法の反復 1 回分の計算を行う

    Parameters
    -----
    temp : ndarray
        格子点の温度を保存しておく配列
    grid : ndarray
        計算する格子点の位置を保存しておく配列

    Returns
    -----
    temp : ndarray
        格子点の温度を保存しておく配列
    stop : bool
        停止基準を満たすかどうかのブール値
    max_error : float
        誤差の最大値

    """

    t_old: float
    t_up: float
    t_right: float
    t_down: float
    t_left: float
    t_new: float

    stop: bool = True
    max_error: float = 0

    for i_y in range(NUM_ROW):
        for i_x in range(NUM_COL):

            t_old = temp[i_y, i_x]

            if grid[i_y, i_x]:
                t_up = value_temp(temp, i_x, i_y+1)
                t_right = value_temp(temp, i_x+1, i_y)
                t_down = value_temp(temp, i_x, i_y-1)
                t_left = value_temp(temp, i_x-1, i_y)

                t_new = gauss_seidel(t_up, t_right, t_down, t_left) \
                    + value_src(i_x, i_y)

                temp[i_y, i_x] = overrelaxation(t_old, t_new)

                stop, max_error = check_stop(t_old, t_new, stop, max_error)
            #
        #
    #

    return temp, stop, max_error
#


@njit
def value_temp(temp: np.ndarray, i_x: int, i_y: int) -> float:
    """
    Neumann 境界条件のときに Gauss-Seidel 法の計算に使う格子点が, 境界外の点を含むかどうかをチェックする

    Parameters
    -----
    temp : ndarray
        格子点の温度を保存しておく配列
    i_x : int
        確認を行う格子点の番号 (x)
    i_y : int
        確認を行う格子点の番号 (y)

    Returns
    -----
    t_value : ndarray
        確認を行なった格子点での温度

    """

    t_value: float

    if (i_y == NUM_ROW) and (T_TOP[1] == 'N'):
        t_value = temp[i_y-2, i_x] + 2*DELTA_Y*T_TOP[0]
    elif (i_x == NUM_COL) and (T_RIGHT[1] == 'N'):
        t_value = temp[i_y, i_x-2] + 2*DELTA_X*T_RIGHT[0]
    elif (i_y == -1) and (T_BOTTOM[1] == 'N'):
        t_value = temp[i_y+2, i_x] - 2*DELTA_Y*T_BOTTOM[0]
    elif (i_x == -1) and (T_LEFT[1] == 'N'):
        t_value = temp[i_y, i_x+2] - 2*DELTA_X*T_LEFT[0]
    else:
        t_value = temp[i_y, i_x]
    #

    return t_value
#


@njit
def gauss_seidel(
        t_up: float, t_right: float, t_down: float, t_left: float) \
        -> float:
    """
    上下左右の格子点の温度を使って, 真ん中の格子点の温度の候補値を求める

    Parameters
    -----
    t_up : float
        上の格子点の温度
    t_right : float
        右の格子点の温度
    t_down : float
        下の格子点の温度
    t_left : float
        左の格子点の温度

    Returns
    -----
    t_center : float
        真ん中の格子点の温度

    """

    dx2: float = DELTA_X ** 2
    dy2: float = DELTA_X ** 2

    t_center: float \
        = (t_up/dy2+t_right/dx2+t_down/dy2+t_left/dx2) / (2/dx2+2/dy2)

    return t_center
#


@njit
def value_src(i_x: int, i_y: int) -> float:
    """
    格子点の熱源を計算する関数

    Parameters
    -----
    i_x : int
        格子点の番号 (x)
    i_y : int
        格子点の番号 (y)

    Return
    -----
        f_value : float

    """

    pos_x: float = LEFT + i_x * DELTA_X
    pos_y: float = BOTTOM + i_y * DELTA_Y

    dx2: float = DELTA_X ** 2
    dy2: float = DELTA_X ** 2

    f_value: float = heat_source(pos_x, pos_y) / (2/dx2+2/dy2)

    return f_value
#


@njit
def overrelaxation(t_old: float, t_new: float) -> float:
    """
    加速緩和を行う

    Parameters
    -----
    t_old : float
        反復の前のステップでの温度
    t_new : float
        反復の現在のステップでの温度

    Returns
    -----
    t_new_sor : float
        加速緩和後の温度

    """

    t_new_sor: float = LAMBDA*t_new + (1-LAMBDA)*t_old

    return t_new_sor
#


def check_stop(t_old: float, t_new: float,
               stop: bool, max_error: float) -> tuple[bool, float]:
    """
    停止基準をチェックする

    Parameters
    -----
    t_old : float
        反復の前のステップでの温度
    t_new : float
        反復の現在のステップでの温度
    stop : bool
        停止基準を満たすかどうかのブール値
    max_error : float
        誤差の最大値

    Returns
    -----
    stop : bool
        停止基準を満たすかどうかのブール値
    max_error : float
        誤差の最大値

    """

    error: float = math.fabs((t_new-t_old)/t_new)

    if error > max_error:
        max_error = error
    #

    if error > CRITERION:
        stop = False
    #

    return stop, max_error
#


if __name__ == '__main__':
    TIME_INIT: float = perf_counter()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    plt.rcParams['text.usetex'] = True

    temperature: np.ndarray
    grid_points: np.ndarray
    temperature, grid_points = init_temp()
    temperature = main_loop(temperature, grid_points)

    figure = plot_temp(temperature)

    TIME_ELAPSED: float = perf_counter() - TIME_INIT
    print(f'{__name__}: {TIME_ELAPSED:.5f} s')

    figure.savefig("laplace2d.png", dpi=300)
    plt.show()
#
