import numpy as np
from numpy import sin, cos, pi
from sympy import zeros


def ode_system(t, y,
               Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
               dC=0, phi_do=0, phi_oc=0,
               dCi=0, phi_di=0, phi_oci=0):
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

    # 非线性(故障)激励计算
    delta = delta_calculate(xi, yi, xo, yo, phi, c, Hdo, Hdi)
    fx, fy = contact_force(phi, delta, Ki, Ko)
    ddxi = (-fx) / mi
    ddyi = (Fr - fy) / mi
    ddxo = (fx - Kbh * xo - Cbh * dxo) / mo
    ddyo = (fy - Kbh * yo - Cbh * dyo) / mo
    return [dxi, dyi, dxo, dyo, ddxi, ddyi, ddxo, ddyo]


def contact_force(phi, delta, Ki, Ko):
    fx = Ki * np.sum(delta ** (10 / 9) @ np.cos(phi))
    fy = Ki * np.sum(delta ** (10 / 9) @ np.sin(phi))
    return fx, fy


def delta_calculate(xi, yi, xo, yo, phi, c, Hdo, Hdi):
    delta = (xi - xo) * cos(phi) + (yi - yo) * sin(phi) - c / 2 - Hdo - Hdi
    delta = np.maximum(delta, 0)
    return delta


def Hdo_calculate(phi, dC, phi_do, phi_oc=0):
    # 向量操作优化
    Hdo = np.zeros_like(phi)
    mask = (np.abs(phi - phi_oc) <= phi_do)
    # mask = (cos(phi - phi_oc)) <= cos(phi_do)
    Hdo[mask] = dC * np.cos((phi[mask] - phi_oc) * np.pi / (2 * phi_do))
    return Hdo


def Hdi_calculate(phi, dCi, phi_di, phi_id):
    # 向量操作优化
    Hdi = np.zeros_like(phi)
    phi_i = np.mod(phi_id, 2 * pi)  # 内圈故障中心角度位置
    mask = (np.abs(phi - phi_i) <= phi_di)
    Hdi[mask] = dCi * np.cos((phi[mask] - phi_i) * np.pi / (2 * phi_di))
    return Hdi


def y_2_acc(t, y,
            Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
            dC=0, phi_do=0, phi_oc=0,
            dCi=0, phi_di=0, phi_oci=0):
    acc = np.zeros((4, y.shape[1]))
    for i in range(y.shape[1]):
        acc[:, i] = ode_system(t[i], y[:, i],
                               Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
                               dC, phi_do, phi_oc,
                               dCi, phi_di, phi_oci)[4:8]
    dy = np.vstack((y, acc))
    return dy
