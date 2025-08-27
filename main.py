import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
from scipy import fft
import BearingModel
import matplotlib
import dynamics_plot
import time
import pandas
import mplcursors


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


def healthy_bearing(fs=1e4, T=1, **kwargs):
    params = dict(
        filename='NUP205',
        Nb=13,
        mi=5324.83e-9 * 7850,
        mo=7664.64e-9 * 7850,
        Kbh=1e7,
        Cbh=1e3,
        Fr=500,
        RPM=1000,
        Rm=19.5e-3,
        Rr=3.75e-3,
        c=1e-6
    )
    params.update(kwargs)
    Ki, Ko = load_K(params["filename"])
    wi = params["RPM"] / 60 * 2 * np.pi
    wc = wi / 2 * (1 - params["Rr"] / params["Rm"])
    fs = int(fs)
    t_span = np.linspace(0, T, T * fs + 1)
    DOF = 4
    y0 = np.zeros(DOF * 2)
    result = solve_ivp(
        BearingModel.ode_system,
        [0, T], y0, t_eval=t_span,
        args=(params["Fr"], params["Nb"], params["mi"], params["mo"],
              wi, wc, Ki, Ko, params["Kbh"], params["Cbh"], params["c"])
    )
    dy = BearingModel.y_2_acc(
        result.t, result.y,
        params["Fr"], params["Nb"], params["mi"], params["mo"],
        wi, wc, Ki, Ko, params["Kbh"], params["Cbh"], params["c"]
    )
    return result.t, dy


def OuterDefect_bearing(fs=1e5, T=1, **kwargs):
    # 默认参数
    params = dict(
        filename='NUP205',
        Nb=13,
        mi=5324.83e-9 * 7850,
        mo=7664.64e-9 * 7850,
        Kbh=1e7,
        Cbh=1e3,
        Fr=500,
        RPM=1000,
        Rm=19.5e-3,
        Rr=3.75e-3,
        c=1e-6,
        B=0.2e-3,  # 默认故障
        phi_oc=np.pi / 2  # 仅 0-pi 承载区域内生效
    )
    params.update(kwargs)
    Ki, Ko = load_K(params["filename"])
    wi = params['RPM'] / 60 * 2 * np.pi
    wc = wi / 2 * (1 - params["Rr"] / params["Rm"])
    fs = int(fs)
    t_span = np.linspace(0, T, T * fs + 1)
    DOF = 4
    y0 = np.zeros(DOF * 2)
    if params["B"] > 0:
        Cd = params["Rr"] - np.sqrt(params["Rr"] ** 2 - (params["B"] / 2) ** 2)
        Ro = params["Rm"] + params["Rr"] / 2
        Co = Ro - np.sqrt(Ro ** 2 - (params["B"] / 2) ** 2)
        dC = Cd - Co
        phi_do = np.arcsin(params["B"] / 2 / Ro)
    else:
        dC = 0.0
        phi_do = 0.0
    # if dC != 0:
    #     phi = np.linspace(0, 2 * np.pi, 1000)
    #     Hdo = BearingModel.Hdo_calculate(phi, dC, phi_do, params["phi_oc"])
    #     plt.figure(1)
    #     plt.plot(phi, Hdo)
    #     plt.savefig(f"{params['filename']}/{params['filename']}_Hdo.png", dpi=300)
    #     plt.xlabel('phi')
    #     plt.close()
    result = solve_ivp(
        BearingModel.ode_system,
        [0, T], y0, t_eval=t_span,
        args=(params["Fr"], params["Nb"], params["mi"], params["mo"],
              wi, wc, Ki, Ko, params["Kbh"], params["Cbh"], params["c"],
              dC, phi_do, params["phi_oc"])
    )
    dy = BearingModel.y_2_acc(
        result.t, result.y,
        params["Fr"], params["Nb"], params["mi"], params["mo"],
        wi, wc, Ki, Ko, params["Kbh"], params["Cbh"], params["c"],
        dC, phi_do, params["phi_oc"]
    )
    return result.t, dy


def InnerDefect_bearing(fs=1e5, T=1, **kwargs):
    params = dict(
        filename='NUP205',
        Nb=13,
        mi=5324.83e-9 * 7850,
        mo=7664.64e-9 * 7850,
        Kbh=1e7,
        Cbh=1e3,
        Fr=500,
        RPM=1000,
        Rm=19.5e-3,
        Rr=3.75e-3,
        c=1e-6,
        B=0.1e-3,
        phi_oci=0.0
    )
    params.update(kwargs)
    Ki, Ko = load_K(params["filename"])
    wi = params["RPM"] / 60 * 2 * np.pi
    wc = wi / 2 * (1 - params["Rr"] / params["Rm"])
    fs = int(fs)
    t_span = np.linspace(0, T, T * fs + 1)
    DOF = 4
    y0 = np.zeros(DOF * 2)
    if params["B"] > 0:
        Cd = params["Rr"] - np.sqrt(params["Rr"] ** 2 - (params["B"] / 2) ** 2)
        Ri = params["Rm"] - params["Rr"] / 2
        Ci = Ri - np.sqrt(Ri ** 2 - (params["B"] / 2) ** 2)
        dCi = Cd - Ci
        phi_di = np.arcsin(params["B"] / 2 / Ri)
    else:
        dCi = 0.0
        phi_di = 0.0
    result = solve_ivp(
        BearingModel.ode_system,
        [0, T], y0, t_eval=t_span,
        args=(params["Fr"], params["Nb"], params["mi"], params["mo"],
              wi, wc, Ki, Ko, params["Kbh"], params["Cbh"], params["c"],
              0, 0, 0,
              dCi, phi_di, params["phi_oci"])
    )
    dy = BearingModel.y_2_acc(
        result.t, result.y,
        params["Fr"], params["Nb"], params["mi"], params["mo"],
        wi, wc, Ki, Ko, params["Kbh"], params["Cbh"], params["c"],
        0, 0, 0,
        dCi, phi_di, params["phi_oci"]
    )
    return result.t, dy


def RollerDefect_bearing(fs=1e5, T=1, **kwargs):
    params = dict(
        filename='NUP205',
        Nb=13,
        mi=5324.83e-9 * 7850,
        mo=7664.64e-9 * 7850,
        Kbh=1e7,
        Cbh=1e3,
        Fr=500,
        RPM=1000,
        Rm=19.5e-3,
        Rr=3.75e-3,
        c=1e-6,
        B=2e-3,
        alpha_b=0.0,
        j=1
    )
    params.update(kwargs)
    Ki, Ko = load_K(params["filename"])
    wi = params["RPM"] / 60 * 2 * np.pi
    wc = wi / 2 * (1 - params["Rr"] / params["Rm"])
    fs = int(fs)
    t_span = np.linspace(0, T, T * fs + 1)
    DOF = 4
    y0 = np.zeros(DOF * 2)
    if params["B"] > 0:
        phi_b = 2 * np.arcsin(params["B"] / 2 / params["Rr"])
        wr = wi * params["Rm"] / 2 / params["Rr"] * (1 - (params["Rr"] / params["Rm"]) ** 2)
        Ro = params["Rm"] + params["Rr"]
        Ri = params["Rm"] - params["Rr"]
        Hbomax = params["Rr"] - np.sqrt(params["Rr"] ** 2 - (params["B"] / 2) ** 2) \
                 + Ro - np.sqrt(Ro ** 2 - (params["B"] / 2) ** 2)
        Hbimax = params["Rr"] - np.sqrt(params["Rr"] ** 2 - (params["B"] / 2) ** 2) \
                 + Ri - np.sqrt(Ri ** 2 - (params["B"] / 2) ** 2)
    else:
        phi_b = 0.0
        wr = 0.0
        Hbomax = 0.0
        Hbimax = 0.0
    result = solve_ivp(
        BearingModel.ode_system,
        [0, T], y0, t_eval=t_span,
        args=(params["Fr"], params["Nb"], params["mi"], params["mo"],
              wi, wc, Ki, Ko, params["Kbh"], params["Cbh"], params["c"],
              0, 0, 0,
              0, 0, 0,
              Hbomax, Hbimax, phi_b, wr, params["alpha_b"], params["j"])
    )
    dy = BearingModel.y_2_acc(
        result.t, result.y,
        params["Fr"], params["Nb"], params["mi"], params["mo"],
        wi, wc, Ki, Ko, params["Kbh"], params["Cbh"], params["c"],
        0, 0, 0,
        0, 0, 0,
        Hbomax, Hbimax, phi_b, wr, params["alpha_b"], params["j"]
    )
    return result.t, dy


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
    start = time.time()
    # 健康轴承 频率分量：fvc
    t, y = healthy_bearing(fs=fs, T=T)
    index = 11  # 5
    signal = y[index, int(0.2 * fs):]
    dynamics_plot.plt_time_domain(signal, fs, show=False, xlim=0.055,
                                  img_save_path='NUP205/healthy_t.png', title='Healthy Bearing')
    dynamics_plot.plt_envelope_spectrum(signal, fs, show=False, xlim=500,
                                        img_save_path='NUP205/healthy_f.png', title='Healthy Bearing')
    # 外圈缺陷
    t, dy = OuterDefect_bearing(fs=fs, T=T, B=0.2e-3)
    index = 11  # 5L
    signal = dy[index, int(0.2 * fs):]
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
    t, y = RollerDefect_bearing(fs=fs, T=T)
    index = 11  # 5
    signal = y[index, int(0.2 * fs):]
    dynamics_plot.plt_time_domain(signal, fs, show=False, xlim=0.31,
                                  img_save_path='NUP205/rollerdefect_t.png', title='Roller Defected Bearing')
    dynamics_plot.plt_envelope_spectrum(signal, fs, show=False, xlim=500,
                                        img_save_path='NUP205/rollerdefect_f.png',
                                        title='Roller Defected Bearing')

    end = time.time()
    elapsed = end - start  # 秒
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"运行时间: {minutes} 分 {seconds:.2f} 秒")
