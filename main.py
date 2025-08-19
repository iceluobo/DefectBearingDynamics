import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas
from scipy import fft
import BearingModel
import matplotlib


def load_K(filename):
    with open(f"{filename}.txt", "r") as f:
        lines = f.readlines()
        ki = int(lines[0].strip())
        ko = int(lines[1].strip())
    return ki, ko


def fft_trans(y, fs):
    nfft = len(y)
    # nfft = 2 ** int(np.ceil(np.log2(len(y))))
    y = y - y.mean()
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


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    healthy_bearing(savedata=True)
