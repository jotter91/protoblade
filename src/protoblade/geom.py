import pathlib
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from typing import Tuple

cartesian_type = np.dtype([("x", np.double), ("y", np.double),("z",np.double)])
polar_type = np.dtype([("r", np.double), ("theta", np.double),("z",np.double)])


def load_curves_from_fpd(fname: str or pathlib.Path)->NDArray:
    """
    Load a series of curves from a formatted point data (fpd) file
    """
    with open(fname, 'r') as f:
        n_pts, n_curve = list(map(int, f.readline().split()))
        pts = np.array([tuple(map(float, line.split())) for line in f.readlines()],dtype=cartesian_type)
    return pts

def convert_to_polar(pts:NDArray)->NDArray:
    """
    Create arrays for radius , Theta and Z from cartesian array
    :param pts: array of cartesian points dtype cartesian
    :return:
    """
    out  = np.empty(shape=(pts['z'].shape[0],), dtype=polar_type)

    out['z'] = pts['z']
    out['r'] =(pts['x']**2.0 + pts['y']**2)**0.5
    out['theta'] = np.arctan(pts['y']/pts['z'])

    return out

def calculate_curve_length(x:NDArray,y:NDArray)->NDArray:
    """
    Create an array of curve length based on the two arrays x,y
    """

    s = np.zeros(x.shape[0])
    for i in range(1, x.shape[0]):
        dx = x[i] - x[i - 1]
        dy = y[i] - y[i - 1]
        s[i] =  s[i-1] + np.sqrt(dx * dx + dy * dy)
    return s

def reinterpolate_curve(x:NDArray,y:NDArray,s:NDArray,base:int=1.5)->Tuple[NDArray,NDArray,NDArray]:
    """
    Reinterpolate the arrays x,y and s based on a power law to base
    :param x: array for axial co-ordinate
    :param y: array for y co-ordinate
    :param s: array for curve length
    :param base: base for power law
    :return: (x_new,y_new,s_new)
    """
    # resample
    N = x.shape[0]
    fx = interp1d(s, x)
    fy = interp1d(s, y)

    del_s = max(s) - min(s)
    base = 1.5

    start = 0.0 ** (1.0 / base)
    mid = 0.5 ** (1.0 / base)
    s_new = np.zeros(N)
    for i in range(1, N + 1):
        s_new[i - 1] = (start + (i - 1) * (mid - start) / (N - 1)) ** base

    s_2 = np.flipud(s_new)

    s_3 = 1 - s_2
    s_4 = np.delete(s_3, 0)

    s_new = np.concatenate((s_new, s_4)) * del_s

    x_new = np.zeros(len(s_new))
    y_new = np.zeros(len(s_new))
    for i in range(len(s_new)):
        x_new[i] = fx(s_new[i])
        y_new[i] = fy(s_new[i])

    return x, y,s_new


def calculate_curvature()->NDArray:
    return