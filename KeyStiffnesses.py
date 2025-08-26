import matplotlib.pyplot as plt
import numpy as np
from math import pi
from scipy.optimize import curve_fit
import matplotlib


def cal_Ki_Ko(R1, R2, Rr, E, v, l, Q):
    bi = 1.59 * (Q / l * (R1 * Rr / (R1 + Rr)) * (1 - v ** 2) / E) ** (1 / 2)
    di = 2 * Q * (1 - v ** 2) / (pi * l * E) * (np.log(4 * R1 * R2 / bi ** 2) + 0.814)

    bo = 1.59 * (Q / l * (Rr * R2 / (R2 - Rr)) * (1 - v ** 2) / E) ** (1 / 2)
    do = 2 * Q * (1 - v ** 2) / (pi * l * E) * (1 - np.log(bo))

    return di, do


def Fd(x, K):
    return K * (x ** (10 / 9))


def calculate(R1, R2, Rr, E, v, l, Q, filename):
    matplotlib.use('TkAgg')
    di, do = [], []
    for i in range(len(Q)):
        if Q[i] == 0:
            di.append(0)
            do.append(0)
            continue
        D = cal_Ki_Ko(R1, R2, Rr, E, v, l, Q[i])
        di.append(D[0])
        do.append(D[1])
    plt.figure(figsize=(4, 3))
    plt.rcParams['font.size'] = 9  # 设置全局字体大小为9磅
    plt.rcParams['font.family'] = 'Times New Roman'  # 设置全局字体为新罗马
    plt.plot(di, Q, 'g--', label='theary_i')
    plt.plot(do, Q, 'b--', label='theary_o')

    const_i = curve_fit(Fd, di, Q, p0=[1e10, ])
    const_o = curve_fit(Fd, do, Q, p0=[1e10, ])
    print(const_i)
    print(const_o)

    delta = np.linspace(0, 1.2e-5, 100)
    F_i = Fd(delta, const_i[0][0])
    F_o = Fd(delta, const_o[0][0])

    plt.plot(delta, F_i, 'r-.', label='poly_i:k=' + str(const_i[0][0]))
    plt.plot(delta, F_o, 'k-.', label='ploy_o:k=' + str(const_o[0][0]))
    plt.legend()
    plt.xlabel(r'$\delta$ /m')
    plt.ylabel(r'$F$ /N')
    plt.tight_layout()
    plt.savefig(filename + '.pdf', dpi=300)
    return const_i, const_o


def save_K(Ki, Ko, filename):
    with open(f"{filename}.txt", "w") as f:
        f.write(f"{int(Ki[0].item())}\n")
        f.write(f"{int(Ko[0].item())}\n")


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    # R1 = 15.75e-3
    # R2 = 23.25e-3
    # Rr = 3.75e-3
    # E = 2.1e11
    # v = 0.3
    # l = 9e-3
    # Q = np.linspace(0, 3000, 3001)
    # filename = 'NUP205'
    # Ki, Ko = calculate(R1, R2, Rr, E, v, l, Q, filename)
    # save_K(Ki, Ko, filename)

    Rr = 11e-3 / 2
    Rm = 60.5e-3 / 2
    R1 = Rm - Rr
    R2 = Rm + Rr

    E = 2.1e11
    v = 0.3
    l = 11e-3
    Q = np.linspace(0, 3000, 3001)
    filename = 'NJ208'
    Ki, Ko = calculate(R1, R2, Rr, E, v, l, Q, filename)
    save_K(Ki, Ko, filename)
