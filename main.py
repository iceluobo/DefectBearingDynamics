import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
import pandas
from scipy import fft
import BearingModel
import matplotlib
import mplcursors


def load_K(filename):
    with open(f"{filename}.txt", "r") as f:
        lines = f.readlines()
        ki = int(lines[0].strip())
        ko = int(lines[1].strip())
    return ki, ko


def fft_trans(y, fs):
    # nfft = len(y)
    y = y - y.mean()
    nfft = 2 ** int(np.ceil(np.log2(len(y))))
    y = scipy.signal.hilbert(y)
    y_ft = fft.fft(y, nfft) / (nfft / 2)
    y_f = fs * np.arange(nfft // 2) / nfft
    return y_f, abs(y_ft[0:int(nfft / 2)])


def healthy_bearing(savedata=False, fs=1e4, show=False, T=2):
    filename = 'NUP205'
    Ki, Ko = load_K(filename)
    Nb = 13
    mi = 5324.83e-9 * 7850
    mo = 7664.64e-9 * 7850
    Kbh = 1e7
    Cbh = 1e2
    Fr = 500
    Rm = 19.5e-3
    Rr = 3.75e-3
    c = 1e-6
    wi = 1000 / 60 * 2 * np.pi
    wc = wi / 2 * (1 - Rr / Rm)
    fs = int(fs)
    t_span = np.linspace(0, T, T * fs + 1)
    DOF = 4
    y0 = np.zeros(DOF * 2)
    # def ode_system(t, y,
    #                Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
    #                dC=0, phi_do=0, phi_oc=0,
    #                dCi=0, phi_di=0, phi_oci=0):
    result = solve_ivp(BearingModel.ode_system, [0, T], y0, t_eval=t_span,
                       args=(Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c, 0, 0, 0, 0, 0, 0))
    y = result.y
    signal = y[5, :]
    signal = np.diff(signal) * fs
    t = np.linspace(0, T, len(signal))
    plt.figure(1)
    plt.plot(t, signal)
    plt.ylim([-5, 5])
    plt.xlim([0.948, 0.964])
    plt.savefig(f"{filename}/{filename}_t_healthy.png", dpi=300)
    if not show:
        plt.close()
    plt.figure(2)
    y_f, y_a = fft_trans(signal[int(0.2 * fs):], fs)
    plt.plot(y_f, y_a)
    plt.xlim([0, 1000])
    plt.ylim([0, 0.006])
    plt.savefig(f"{filename}/{filename}_f_healthy.png", dpi=300)
    if not show:
        plt.close()
    if show:
        mplcursors.cursor()
        plt.show()
    if savedata:
        data = np.zeros((4, fs * T))
        for i in range(4):
            data[i] = np.diff(y[4 + i, :]) * fs
        data = pandas.DataFrame(data)
        data.to_csv(f"{filename}/{filename}_healthy.csv", header=False, index=False)


def OuterDefect_bearing(savedata=False, fs=1e4, show=False, T=1):
    filename = 'NUP205'
    Ki, Ko = load_K(filename)
    Nb = 13
    mi = 5324.83e-9 * 7850
    mo = 7664.64e-9 * 7850
    Kbh = 1e7
    Cbh = 1e2
    Fr = 500
    Rm = 19.5e-3
    Rr = 3.75e-3
    # filename = '6307'
    # Ki, Ko = 1.89e10, 1.89e10
    # Nb = 8
    # mi = 1.2638
    # mo = 12.638
    # Kbh = 15e6
    # Cbh = 2.2e3
    # Fr = 2000
    # Rm = 19.5e-3
    # Rr = 3.75e-3
    # Rm = 28.75e-3
    # Rr = 7e-3
    c = 1e-6
    wi = 1000 / 60 * 2 * np.pi
    wc = wi / 2 * (1 - Rr / Rm)
    fs = int(fs)
    t_span = np.linspace(0, T, T * fs + 1)
    DOF = 4
    y0 = np.zeros(DOF * 2)
    '''
    基于动力学的滚动轴承多点故障建模及振动特性研究_李君飞.pdf --> page27
    相较于健康工况，增加参数dC ，phi_do, phi_oc
    '''
    B = 0.2e-3  # 故障宽度 mm
    Cd = Rr - np.sqrt(Rr ** 2 - (B / 2) ** 2)
    Ro = Rm + Rr / 2
    Co = Ro - np.sqrt(Ro ** 2 - (B / 2) ** 2)
    dC = Cd - Co
    phi_do = np.arcsin(B / 2 / Ro)
    phi_oc = np.pi / 4 * 2
    # 角度-外滚道故障位移激励测试
    phi = np.linspace(0, 2 * np.pi, 1000)
    Hdo = BearingModel.Hdo_calculate(phi, dC, phi_do, phi_oc)
    plt.figure(1)
    plt.plot(phi, Hdo)
    plt.savefig(f"{filename}/{filename}_Hdo.png", dpi=300)
    plt.xlabel('phi')
    if not show:
        plt.close()
    # def ode_system(t, y,
    #                Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
    #                dC=0, phi_do=0, phi_oc=0,
    #                dCi=0, phi_di=0, phi_oci=0):
    result = solve_ivp(BearingModel.ode_system, [0, T], y0, t_eval=t_span,
                       args=(Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
                             dC, phi_do, phi_oc))
    y = result.y
    signal = y[5, :]
    signal = np.diff(signal) * fs
    t = np.linspace(0, T, len(signal))
    plt.figure(2)
    plt.plot(t, signal)
    plt.ylim([-1500, 1500])
    plt.xlim([0.5, 1])
    plt.savefig(f"{filename}/{filename}_t_outer.png", dpi=300)
    if not show:
        plt.close()
    plt.figure(3)
    y_f, y_a = fft_trans(signal[int(0.4 * fs):], fs)
    plt.plot(y_f, y_a)
    plt.xlim([0, 1000])
    plt.ylim([0, 0.05])
    plt.savefig(f"{filename}/{filename}_f_outer.png", dpi=300)
    if not show:
        plt.close()
    if show:
        mplcursors.cursor()
        plt.show()

    if savedata:
        data = np.zeros((4, fs * T))
        for i in range(4):
            data[i] = np.diff(y[4 + i, :]) * fs
        data = pandas.DataFrame(data)
        data.to_csv(f"{filename}/{filename}_outer.csv", header=False, index=False)
    return t, signal


def InnerDefect_bearing(savedata=False, fs=1e4, show=False, T=1):
    filename = 'NUP205'
    Ki, Ko = load_K(filename)
    Nb = 13
    mi = 5324.83e-9 * 7850
    mo = 7664.64e-9 * 7850
    Kbh = 1e7
    Cbh = 1e3
    Fr = 500
    Rm = 19.5e-3
    Rr = 3.75e-3
    c = 1e-6
    wi = 1000 / 60 * 2 * np.pi
    wc = wi / 2 * (1 - Rr / Rm)
    fs = int(fs)
    t_span = np.linspace(0, T, T * fs + 1)
    DOF = 4
    y0 = np.zeros(DOF * 2)
    '''
    基于动力学的滚动轴承多点故障建模及振动特性研究_李君飞.pdf --> page27
    相较于健康工况，增加参数dC ，phi_do, phi_oc
    '''
    B = 0.1e-3  # 故障宽度 mm
    Cd = Rr - np.sqrt(Rr ** 2 - (B / 2) ** 2)
    Ri = Rm - Rr / 2
    Ci = Ri - np.sqrt(Ri ** 2 - (B / 2) ** 2)
    dCi = Cd - Ci
    phi_di = np.arcsin(B / 2 / Ri)
    phi_oci = 0
    # 角度-内滚道故障位移激励测试
    t_test = np.linspace(0, T, T * fs + 1)
    phi_i = t_test * wi + phi_oci
    phi_r = t_test * wc + 0
    Hdi = np.zeros_like(phi_i)
    for i in range(len(t_test)):
        Hdi[i] = BearingModel.Hdi_calculate(np.mod(phi_r[i], 2 * np.pi), dCi, phi_di, np.mod(phi_i[i], 2 * np.pi))
    plt.figure(1)
    plt.plot(t_test, Hdi)
    plt.savefig(f"{filename}/{filename}_Hdi.png", dpi=300)
    plt.xlabel('t (s)')
    if not show:
        plt.close()
    # def ode_system(t, y,
    #                Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
    #                dC=0, phi_do=0, phi_oc=0,
    #                dCi=0, phi_di=0, phi_oci=0):
    result = solve_ivp(BearingModel.ode_system, [0, T], y0, t_eval=t_span,
                       args=(Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
                             0, 0, 0,
                             dCi, phi_di, phi_oci))
    y = result.y
    signal = y[5, :]
    signal = np.diff(signal) * fs
    t = np.linspace(0, T, len(signal))
    plt.figure(2)
    plt.plot(t, signal)
    plt.ylim([-4000, 4000])
    plt.xlim([0.5, 1])
    plt.savefig(f"{filename}/{filename}_t_inner.png", dpi=300)
    if not show:
        plt.close()
    plt.figure(3)
    y_f, y_a = fft_trans(signal[int(round(0.2 * fs)):], fs)
    plt.plot(y_f, y_a)
    plt.xlim([0, 1000])
    plt.ylim([0, 0.04])
    plt.savefig(f"{filename}/{filename}_f_inner.png", dpi=300)
    if not show:
        plt.close()
    if show:
        mplcursors.cursor()
        plt.show()
    if savedata:
        data = np.zeros((4, fs * T))
        for i in range(4):
            data[i] = np.diff(y[4 + i, :]) * fs
        data = pandas.DataFrame(data)
        data.to_csv(f"{filename}/{filename}_inner.csv", header=False, index=False)


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    '''
    内圈转速: 1000 RPM
    转频：16.67 Hz
    变柔度频率: 87.5 Hz
    外圈故障特征频率: 87.5 Hz
    内圈故障特征频率: 129.17 Hz
    滚动体故障特征频率： 41.73 Hz
    保持架故障特征频率： 6.73 Hz
    '''
    # healthy_bearing(savedata=False, show=True, T=2, fs=1e4)
    # OuterDefect_bearing(savedata=False, show=True, T=2, fs=1e5)
    InnerDefect_bearing(savedata=False, show=True, T=2, fs=1e5)
