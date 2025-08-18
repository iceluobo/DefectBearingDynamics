import numpy as np
from numpy import sin, cos, pi


def ode_system(t, y, Fr, Nb, mi, mo, wc, Ki, Ko, Kbh, Cbh, c):
    xi, yi, xo, yo, dxi, dyi, dxo, dyo = y
    phi_0 = wc * t
    phi = phi_0 + np.arange(Nb) / Nb * 2 * pi
    delta = delta_calculate(xi, yi, xo, yo, phi, c)
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


def delta_calculate(xi, yi, xo, yo, phi, c):
    delta = (xi - xo) * cos(phi) + (yi - yo) * sin(phi) - c
    delta = np.maximum(delta, 0)
    return delta
