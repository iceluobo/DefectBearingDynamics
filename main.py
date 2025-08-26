import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
import pandas
from scipy import fft
import BearingModel
import matplotlib
import mplcursors
import dynamics_plot


def load_K(filename):
    with open(f"{filename}.txt", "r") as f:
        lines = f.readlines()
        ki = int(lines[0].strip())
        ko = int(lines[1].strip())
    return ki, ko


def fft_trans(y, fs):
    fs = int(fs)
    nfft = 2 ** int(np.ceil(np.log2(len(y))))
    y = np.abs(scipy.signal.hilbert(y))
    y = y - y.mean()
    y_ft = fft.fft(y, nfft) / (nfft / 2)
    y_f = fs * np.arange(nfft // 2) / nfft
    return y_f, abs(y_ft[0:int(nfft / 2)])


def healthy_bearing(fs=1e4, T=1):
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
    # def ode_system(t, y,
    #                Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
    #                dC=0, phi_do=0, phi_oc=0,
    #                dCi=0, phi_di=0, phi_oci=0):
    result = solve_ivp(BearingModel.ode_system, [0, T], y0, t_eval=t_span,
                       args=(Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c, 0, 0, 0, 0, 0, 0))
    dy = BearingModel.y_2_acc(result.t, result.y,
                              Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c)
    return result.t, dy


def OuterDefect_bearing(fs=1e5, T=1):
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
    plt.close()
    # def ode_system(t, y,
    #                Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
    #                dC=0, phi_do=0, phi_oc=0,
    #                dCi=0, phi_di=0, phi_oci=0):
    result = solve_ivp(BearingModel.ode_system, [0, T], y0, t_eval=t_span,
                       args=(Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
                             dC, phi_do, phi_oc))
    dy = BearingModel.y_2_acc(result.t, result.y,
                              Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
                              dC, phi_do, phi_oc)
    return result.t, dy


def InnerDefect_bearing(fs=1e5, T=1):
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
    t_test = np.linspace(0, T, T * fs * 10 + 1)
    phi_i = t_test * wi + phi_oci
    phi_r = t_test * wc + 0
    Hdi = np.zeros_like(phi_i)
    for i in range(len(t_test)):
        Hdi[i] = BearingModel.Hdi_calculate(np.mod(phi_r[i], 2 * np.pi), dCi, phi_di, np.mod(phi_i[i], 2 * np.pi))
    plt.figure(1)
    plt.plot(t_test, Hdi)
    plt.savefig(f"{filename}/{filename}_Hdi.png", dpi=300)
    plt.xlabel('t (s)')
    plt.ylabel('Hdi (mm)')
    plt.close()
    # def ode_system(t, y,
    #                Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
    #                dC=0, phi_do=0, phi_oc=0,
    #                dCi=0, phi_di=0, phi_oci=0):
    result = solve_ivp(BearingModel.ode_system, [0, T], y0, t_eval=t_span,
                       args=(Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
                             0, 0, 0,
                             dCi, phi_di, phi_oci))
    dy = BearingModel.y_2_acc(result.t, result.y,
                              Fr, Nb, mi, mo, wi, wc, Ki, Ko, Kbh, Cbh, c,
                              0, 0, 0,
                              dCi, phi_di, phi_oci)
    return result.t, dy


def RollerDefect(fs=1e5, T=1):
    pass


def void_function():
    pass


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    '''
    内圈转速: 1000 RPM
    NUP205 理论频率分量
    转频 fi：16.67 Hz
    变柔度频率 fvc: 87.5 Hz
    外圈故障特征频率 fbpfo: 87.5 Hz
    内圈故障特征频率 fbpfi: 129.17 Hz
    滚动体故障特征频率 fbsf： 41.73 Hz
    保持架故障特征频率 fc： 6.73 Hz
    '''
    T = 2
    fs = 1e5
    # 健康轴承 频率分量：fvc
    t, y = healthy_bearing(fs=fs, T=T)
    index = 11  # 5
    signal = y[index, int(0.2 * fs):]
    dynamics_plot.plt_time_domain(signal, fs, show=False, xlim=0.055,
                                  img_save_path='NUP205/healthy_t.png', title='Healthy Bearing')
    dynamics_plot.plt_envelope_spectrum(signal, fs, show=False, xlim=500,
                                        img_save_path='NUP205/healthy_f.png', title='Healthy Bearing')

    # 外圈故障 频率分量：n * fbpfo
    t, y = OuterDefect_bearing(fs=fs, T=T)
    index = 11  # 5
    signal = y[index, int(0.2 * fs):]
    dynamics_plot.plt_time_domain(signal, fs, show=False, xlim=0.055,
                                  img_save_path='NUP205/outerdefect_t.png', title='Outer Raceway Defected Bearing')
    dynamics_plot.plt_envelope_spectrum(signal, fs, show=False, xlim=500,
                                        img_save_path='NUP205/outerdefect_f.png',
                                        title='Outer Raceway Defected Bearing')

    # 内圈故障 频率分量：n * fi, n * fbpfi +- m * fi
    t, y = InnerDefect_bearing(fs=fs, T=T)
    index = 11  # 5
    signal = y[index, int(0.2 * fs):]
    dynamics_plot.plt_time_domain(signal, fs, show=False, xlim=0.31,
                                  img_save_path='NUP205/innerdefect_t.png', title='Inner Raceway Defected Bearing')
    dynamics_plot.plt_envelope_spectrum(signal, fs, show=False, xlim=500,
                                        img_save_path='NUP205/innerdefect_f.png',
                                        title='Inner Raceway Defected Bearing')

    # 滚动体故障 频率分量：n * fc, n * fbsf +- m * fc
