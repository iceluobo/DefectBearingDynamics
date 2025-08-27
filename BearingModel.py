import numpy as np
from numpy import sin, cos, pi
from numba import njit


@njit
def ode_system(t, y,
               Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
               dC=0, phi_do=0, phi_oc=0,
               dCi=0, phi_di=0, phi_oci=0,
               Hbomax=0, Hbimax=0, phi_b=0, wr=0, alpha_b=0, j=1):
    xi, yi, xo, yo, dxi, dyi, dxo, dyo = y
    phi_0 = wc * t
    phi = phi_0 + np.arange(Nb) / Nb * 2 * pi
    phi = np.mod(phi, 2 * pi)
    # 外滚道故障位移激励控制
    Hdo = np.zeros_like(phi)
    if dC != 0 and phi_do != 0:
        Hdo = Hdo_calculate(phi, dC, phi_do, phi_oc)
    # 内滚道故障位移激励控制
    Hdi = np.zeros_like(phi)
    if dCi != 0 and phi_di != 0:
        phi_id = wi * t + phi_oci
        Hdi = Hdi_calculate(phi, dCi, phi_di, phi_id)
    # 滚子故障位移激励控制
    Hdr = np.zeros_like(phi)
    if Hbomax != 0 and Hbimax != 0 and wr != 0 and phi_b != 0:
        beta = wr * t + 2 * pi * (j - 1) / Nb + alpha_b
        beta = np.mod(beta, 2 * pi)
        Hdr = Hdr_calculate(beta, Hbomax, Hbimax, phi_b, j, Nb)
    # 非线性(故障)激励计算
    delta = delta_calculate(xi, yi, xo, yo, phi, c, Hdo, Hdi, Hdr)
    fx, fy = contact_force(phi, delta, Ki, Ko)
    ddxi = (-fx) / mi
    ddyi = (Fr - fy) / mi
    ddxo = (fx - Kbh * xo - Cbh * dxo) / mo
    ddyo = (fy - Kbh * yo - Cbh * dyo) / mo
    return [dxi, dyi, dxo, dyo, ddxi, ddyi, ddxo, ddyo]


@njit
def contact_force(phi, delta, Ki, Ko):
    fx = Ki * np.sum(delta ** (10 / 9) * np.cos(phi))
    fy = Ki * np.sum(delta ** (10 / 9) * np.sin(phi))
    return fx, fy


@njit
def delta_calculate(xi, yi, xo, yo, phi, c, Hdo, Hdi, Hdr):
    delta = (xi - xo) * cos(phi) + (yi - yo) * sin(phi) - c / 2 - Hdo - Hdi - Hdr
    delta = np.maximum(delta, 0)
    return delta


@njit
def Hdo_calculate(phi, dC, phi_do, phi_oc=0):
    Hdo = np.zeros_like(phi)
    mask = (np.abs(phi - phi_oc) <= phi_do)
    Hdo[mask] = dC * np.cos((phi[mask] - phi_oc) * np.pi / (2 * phi_do))
    return Hdo


@njit
def Hdi_calculate(phi, dCi, phi_di, phi_id):
    Hdi = np.zeros_like(phi)
    phi_i = np.mod(phi_id, 2 * pi)  # 内圈故障中心角度位置
    mask = (np.abs(phi - phi_i) <= phi_di)
    Hdi[mask] = dCi * np.cos((phi[mask] - phi_i) * np.pi / (2 * phi_di))
    return Hdi


@njit
def Hdr_calculate(beta, Hbomax, Hbimax, phi_b, j, Nb):
    Hdr = np.zeros(Nb)
    if np.abs(beta) <= phi_b:
        Hdr[j - 1] = Hbomax * 0.5 * (1 + np.cos(pi * (beta - phi_b / 2) / phi_b * 2))
    elif np.abs(beta - pi) <= phi_b:
        Hdr[j - 1] = Hbimax * 0.5 * (1 + np.cos(np.pi * (beta - np.pi) / phi_b))
    return Hdr


@njit
def y_2_acc(t, y,
            Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
            dC=0, phi_do=0, phi_oc=0,
            dCi=0, phi_di=0, phi_oci=0,
            Hbomax=0, Hbimax=0, phi_b=0, wr=0, alpha_b=0, j=1):
    acc = np.zeros((4, y.shape[1]))
    for i in range(y.shape[1]):
        acc[:, i] = ode_system(t[i], y[:, i],
                               Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
                               dC, phi_do, phi_oc,
                               dCi, phi_di, phi_oci,
                               Hbomax, Hbimax, phi_b, wr, alpha_b, j)[4:8]
    dy = np.vstack((y, acc))
    return dy
