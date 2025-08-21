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
    nfft = 2 ** int(np.ceil(np.log2(len(y))))
    y = y - y.mean()
    y = scipy.signal.hilbert(y)
    y_ft = fft.fft(y, nfft) / (nfft / 2)
    y_f = fs * np.arange(nfft // 2) / nfft
    return y_f, abs(y_ft[0:int(nfft / 2)])


def healthy_bearing(savedata=False):
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
    wi = 3000 / 60 * 2 * np.pi
    wc = wi / 2 * (1 - Rr / Rm)
    fs = int(1e5)
    T = 1
    t_span = np.linspace(0, T, T * fs + 1)
    DOF = 4
    y0 = np.zeros(DOF * 2)
    # (t, y, Fr, Nb, mi, mo, wc, Ki, Ko, Kbh, Cbh, c)
    result = solve_ivp(BearingModel.ode_system, [0, T], y0, t_eval=t_span,
                       args=(Fr, Nb, mi, mo, wc, Ki, Ko, Kbh, Cbh, c),
                       max_step=1e-5, rtol=1, atol=1)
    y = result.y
    signal = y[7, :]
    signal = np.diff(signal) * fs
    t = np.linspace(0, T, len(signal))
    plt.figure(1)
    plt.plot(t, signal)
    plt.ylim([-3, 3])
    plt.xlim([0.948, 0.964])
    plt.savefig(f"{filename}/{filename}_t_healthy.png", dpi=300)
    plt.close()
    plt.figure(2)
    y_f, y_a = fft_trans(signal[int(0.2 * fs):], fs)
    plt.plot(y_f, y_a)
    plt.xlim([0, 1000])
    plt.ylim([0, 0.0011])
    plt.savefig(f"{filename}/{filename}_f_healthy.png", dpi=300)
    plt.close()
    if savedata:
        data = np.zeros((4, fs * T))
        for i in range(4):
            data[i] = np.diff(y[4 + i, :]) * fs
        data = pandas.DataFrame(data)
        data.to_csv(f"{filename}/{filename}_healthy.csv", header=False, index=False)


def OuterDefect_bearing(savedata=False):
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
    wi = 3000 / 60 * 2 * np.pi
    wc = wi / 2 * (1 - Rr / Rm)
    fs = int(1e5)
    T = 1
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
    plt.close()

    # def ode_system(t, y, Fr, Nb, mi, mo, wc, Ki, Ko, Kbh, Cbh, c, dC=0, phi_Do=0, phi_oc=0)
    result = solve_ivp(BearingModel.ode_system, [0, T], y0, t_eval=t_span,
                       args=(Fr, Nb, mi, mo, wc, Ki, Ko, Kbh, Cbh, c, dC, phi_do, phi_oc),
                       max_step=1e-5, rtol=1, atol=1)
    y = result.y
    signal = y[7, :]
    signal = np.diff(signal) * fs
    t = np.linspace(0, T, len(signal))
    plt.figure(2)
    plt.plot(t, signal)
    plt.ylim([-1500, 1500])
    plt.xlim([0.948, 0.964])
    plt.savefig(f"{filename}/{filename}_t_outer.png", dpi=300)
    plt.close()
    plt.figure(3)
    y_f, y_a = fft_trans(signal[int(0.2 * fs):], fs)
    plt.plot(y_f, y_a)
    plt.xlim([0, 1000])
    plt.ylim([0, 0.05])
    plt.savefig(f"{filename}/{filename}_f_outer.png", dpi=300)
    plt.close()

    # plt.show()
    if savedata:
        data = np.zeros((4, fs * T))
        for i in range(4):
            data[i] = np.diff(y[4 + i, :]) * fs
        data = pandas.DataFrame(data)
        data.to_csv(f"{filename}/{filename}_outer.csv", header=False, index=False)

def InnerDefect_bearing(savedata=False):
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
    wi = 3000 / 60 * 2 * np.pi
    wc = wi / 2 * (1 - Rr / Rm)
    fs = int(1e5)
    T = 1
    t_span = np.linspace(0, T, T * fs + 1)
    DOF = 4
    y0 = np.zeros(DOF * 2)
    '''
    基于动力学的滚动轴承多点故障建模及振动特性研究_李君飞.pdf --> page27
    相较于健康工况，增加参数dC ，phi_do, phi_oc
    '''
    B = 0.2e-3  # 故障宽度 mm
    Cd = Rr - np.sqrt(Rr ** 2 - (B / 2) ** 2)
    Ri = Rm - Rr / 2
    Ci = Ri - np.sqrt(Ri ** 2 - (B / 2) ** 2)
    dCi = Cd - Ci
    phi_di = np.arcsin(B / 2 / Ri)
    phi_oci = 0
    # 角度-内滚道故障位移激励测试
    phi = np.linspace(0, 2 * np.pi, 1000)
    Hdo = BearingModel.Hdi_calculate(phi, dCi, phi_di, phi_oci)
    plt.figure(1)
    plt.plot(phi, Hdo)
    plt.savefig(f"{filename}/{filename}_Hdi.png", dpi=300)
    plt.close()

    '''
    def ode_system(t, y, Fr, Nb, mi, mo, wc, Ki, Ko, Kbh, Cbh, c,
                   dC=0, phi_do=0, phi_oc=0,
                   dCi=0, phi_di=0, wi=0, phi_oci=0)
    '''

    result = solve_ivp(BearingModel.ode_system, [0, T], y0, t_eval=t_span,
                       args=(Fr, Nb, mi, mo, wc, Ki, Ko, Kbh, Cbh, c,
                             0, 0, 0,
                             dCi, phi_di, wi, phi_oci),
                       max_step=1e-5, rtol=1, atol=1)
    y = result.y
    signal = y[7, :]
    signal = np.diff(signal) * fs
    t = np.linspace(0, T, len(signal))
    plt.figure(2)
    plt.plot(t, signal)
    plt.ylim([-4000, 4000])
    plt.xlim([0.5, 1])
    plt.savefig(f"{filename}/{filename}_t_inner.png", dpi=300)
    # plt.close()
    plt.figure(3)
    y_f, y_a = fft_trans(signal[int(round(0.2 * fs)):], fs)
    plt.plot(y_f, y_a)
    plt.xlim([0, 1000])
    plt.ylim([0, 0.04])
    plt.savefig(f"{filename}/{filename}_f_inner.png", dpi=300)
    # plt.close()

    mplcursors.cursor()
    plt.show()

if __name__ == '__main__':
    matplotlib.use('TkAgg')

    # healthy_bearing(savedata=True)
    # OuterDefect_bearing(savedata=True)
    InnerDefect_bearing()



