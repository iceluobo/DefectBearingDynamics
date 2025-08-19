import numpy as np
from numpy import sin, cos, pi


def ode_system(t, y, Fr, Nb, mi, mo, wc, Ki, Ko, Kbh, Cbh, c, dC=0, phi_Do=0, phi_oc=0):
    xi, yi, xo, yo, dxi, dyi, dxo, dyo = y
    phi_0 = wc * t
    phi = phi_0 + np.arange(Nb) / Nb * 2 * pi
    phi = np.mod(phi, 2 * pi)
    # 外滚道故障位移激励控制
    Hdo = np.zeros_like(phi)
    if dC != 0 and phi_Do != 0:
        Hdo = Hdo_calculate(phi, dC, phi_Do, phi_oc)
    # 非线性(故障)激励计算
    delta = delta_calculate(xi, yi, xo, yo, phi, c, Hdo)
    fx, fy = contact_force(phi, delta, Ki, Ko)
    ddxi = (-fx - xi) / mi
    ddyi = (Fr - fy - yi) / mi
    ddxo = (fx - Kbh * xo - Cbh * dxo) / mo
    ddyo = (fy - Kbh * yo - Cbh * dyo) / mo
    return [dxi, dyi, dxo, dyo, ddxi, ddyi, ddxo, ddyo]


def contact_force(phi, delta, Ki, Ko):
    fx = Ki * np.sum(delta ** (10 / 9) * np.cos(phi))
    fy = Ko * np.sum(delta ** (10 / 9) * np.sin(phi))
    return fx, fy


def delta_calculate(xi, yi, xo, yo, phi, c, Hdo):
    delta = (xi - xo) * cos(phi) + (yi - yo) * sin(phi) - c - Hdo
    delta = np.maximum(delta, 0)
    # delta_0 = (xi - xo) * cos(phi) + (yi - yo) * sin(phi) - c
    # delta_0 = np.maximum(delta_0, 0)
    # print(delta_0,'\n',Hdo,'\n')
    return delta


def Hdo_calculate(phi, dC, phi_do, phi_oc=0):
    # 向量操作优化
    Hdo = np.zeros_like(phi)
    mask = (np.abs(phi - phi_oc) <= phi_do)
    Hdo[mask] = dC * np.cos((phi[mask] - phi_oc) * np.pi / (2 * phi_do))
    return Hdo

# 原版
# def Hdo_calculate(dC, phi, phi_do, phi_oc):
#     phi = np.atleast_1d(phi)
#     Hdo = np.zeros_like(phi)
#     for i in range(len(phi)):
#         if phi[i] < phi_oc and phi[i] >= phi_oc - phi_do:
#             Hdo[i] = dC * cos((phi[i] - phi_oc) * np.pi / 2 / phi_do)
#         elif phi[i] == phi_oc:
#             Hdo[i] = dC
#         elif phi[i] >= phi_oc and phi[i] <= phi_oc + phi_do:
#             Hdo[i] = dC * cos((phi[i] - phi_oc) * np.pi / 2 / phi_do)
#         else:
#             Hdo[i] = 0
#     return Hdo
