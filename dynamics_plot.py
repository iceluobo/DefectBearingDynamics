import numpy as np
import matplotlib.pyplot as plt
import scipy
import mplcursors

'''
https://mp.weixin.qq.com/s?__biz=MzkzNDI1MjA4Ng==&mid=2247508372&idx=1&sn=ce0d8d44db45103cb371e10c4b726441&chksm=c307e94c9cc115ef9d086660767e2faadd0b70983018e3f3c9e8cc3d0ce55f96f7082553a69a&mpshare=1&scene=1&srcid=0825S24tMrEojkvMzxYSKWd6&sharer_shareinfo=e1c6e091fa32a8febaca60336ca3cb70&sharer_shareinfo_first=e1c6e091fa32a8febaca60336ca3cb70&key=daf9bdc5abc4e8d096cad01d2213dc27f79325aa19a62e8a092d21bbe0821211923e95bd8661f7cb5f345c2e3e47349741648fcad91d0fc792fdda4496b656eb6c04c761371fec058da9a2b92793fc4cafe17fe75b31ad225ece1881075614435f8ac707d61e9a0b9bd3a32b6d1bda0749833e0785e215d5468f317135321161&ascene=0&uin=MTQ2MTczMDI2&devicetype=UnifiedPCWindows&version=f254061a&lang=zh_CN&countrycode=CN&exportkey=n_ChQIAhIQVzb8qbnnbgdJAdAmtrBbjRLnAQIE97dBBAEAAAAAABs6AtwjcooAAAAOpnltbLcz9gKNyK89dVj07mYVLR1nLAc30aCQdh%2BQPbKNWshQHDfcLJ4gxUfzxSJ62zb%2B%2BkFPivWQVKaMUX%2FvWcL4Rg1BcZVeEuDWzv2K2F%2B3%2BDDBe3Uw82f9owCmHtsdTwxubMHo64JONEPOBGgVzaJCJgLbyGracpV645LBO5cvJZXGye18sbzUL%2FJJ7oMj2Y%2BCh%2BMafAaKw5hdURC0S5g7aj3%2BrYfKBvD6I2118XwlEBEHLKcF9qmDy5AFHeDU6JbSNZ0fbiyf%2FUkTSJW%2Bfw%3D%3D&acctmode=0&pass_ticket=cqEoyGz2xqfrko1WZANh93ao5KrGA1i9k47piGtwY9j%2FoXn8geCHJw%2FgL5Zcep53&wx_header=0
'''


def plt_time_domain(arr, fs, ylabel='Amp (m/s^2)',
                    title='Data in time-domain', img_save_path=None,
                    x_vline=None, y_hline=None, xlim=None,
                    show=True):
    """
    :fun: 绘制时域图模板
    :param arr: 输入一维数组数据
    :param fs: 采样频率
    :param ylabel: y轴标签
    :param title: 图标题
    :return: None
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    font = {'family': 'Times New Roman', 'size': '20', 'color': '0.5', 'weight': 'bold'}

    plt.figure(figsize=(12, 4))
    length = len(arr)
    t = np.linspace(0, length / fs, length)
    plt.plot(t, arr, c='b')
    plt.xlabel('t(s)')
    plt.ylabel(ylabel)
    plt.title(title)
    mplcursors.cursor(multiple=True)
    if xlim:
        plt.xlim(0, xlim)
    if x_vline:
        plt.vlines(x=x_vline, ymin=np.min(arr), ymax=np.max(arr), linestyle='--', colors='r')
    if y_hline:
        plt.hlines(y=y_hline, xmin=np.min(t), xmax=np.max(t), linestyle=':', colors='y')
    # ===保存图片====#
    if img_save_path:
        plt.savefig(img_save_path, dpi=500, bbox_inches='tight')
    if show:
        plt.show(block=True)
    elif not show:
        plt.close()


##========绘制频域信号图========##
def plt_fft_img(arr, fs, ylabel='Amp (m/s^2)', title='Data in frequency-domain',
                img_save_path=None, vline=None, hline=None, xlim=None,
                show=False):
    """
    :fun: 绘制频域图模板
    :param arr: 输入一维时域数组数据
    :param fs: 采样频率
    :param ylabel: y轴标签
    :param title: 图标题
    :return: None
    """
    # 计算频域幅值
    length = len(arr)
    t = np.linspace(0, length / fs, length)
    fft_result = np.fft.fft(arr)
    fft_freq = np.fft.fftfreq(len(arr), d=t[1] - t[0])  # FFT频率
    fft_amp = 2 * np.abs(fft_result) / len(t)  # FFT幅值
    # 绘制频域图
    plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.plot(fft_freq[0: int(len(t) / 2)], fft_amp[0: int(len(t) / 2)], label='Frequency Spectrum', color='b')
    plt.xlabel('f (Hz)')
    plt.ylabel(ylabel)
    plt.legend()
    if vline:
        plt.vlines(x=vline, ymin=np.min(fft_amp), ymax=np.max(fft_amp), linestyle='--', colors='r')
    if hline:
        plt.hlines(y=hline, xmin=np.min(fft_freq), xmax=np.max(fft_freq), linestyle=':', colors='y')
    # ===保存图片====#
    if img_save_path:
        plt.savefig(img_save_path, dpi=500, bbox_inches='tight')
    if xlim:  # 图片横坐标是否设置xlim
        plt.xlim(0, xlim)
    plt.tight_layout()
    mplcursors.cursor(multiple=True)
    if show:
        plt.show(block=True)
    elif not show:
        plt.close()
    return fft_freq, fft_amp


##========绘制包络谱图========##
def plt_envelope_spectrum(data, fs, ylabel='Amp (m/s^2)', title='Envelope Spectrum',
                          img_save_path=None, vline=None, hline=None, xlim=None,
                          show=True):
    '''
    fun: 绘制包络谱图
    param data: 输入数据，1维array
    param fs: 采样频率
    param xlim: 图片横坐标xlim，default = None
    param vline: 图片垂直线，default = None
    '''
    from scipy import fftpack
    # =========做希尔伯特变换=======#
    xt = data
    ht = fftpack.hilbert(xt)
    at = np.sqrt(xt ** 2 + ht ** 2)  # 获得解析信号at = sqrt(xt^2 + ht^2)
    at = at - np.mean(at)  # 去直流分量
    fft_amp = np.fft.fft(at)  # 对解析信号at做fft变换获得幅值
    fft_amp = np.abs(fft_amp)  # 对幅值求绝对值（此时的绝对值很大）
    fft_amp = fft_amp / len(fft_amp) * 2
    fft_amp = fft_amp[0: int(len(fft_amp) / 2)]  # 取正频率幅值
    fft_freq = np.fft.fftfreq(len(at), d=1 / fs)  # 获取fft频率，此时包括正频率和负频率
    fft_freq = fft_freq[0:int(len(fft_freq) / 2)]  # 获取正频率
    # 绘制包络谱图
    plt.figure(figsize=(12, 4))
    plt.title(title)
    plt.plot(fft_freq, fft_amp, label='Envelope Spectrum', color='b')
    plt.xlabel('f (Hz)')
    plt.ylabel(ylabel)
    plt.legend()
    if vline:
        plt.vlines(x=vline, ymin=np.min(fft_amp), ymax=np.max(fft_amp), linestyle='--', colors='r')
    if hline:
        plt.hlines(y=hline, xmin=np.min(fft_freq), xmax=np.max(fft_freq), linestyle=':', colors='y')
    if xlim:  # 图片横坐标是否设置xlim
        plt.xlim(0, xlim)
    # ===保存图片====#
    if img_save_path:
        plt.savefig(img_save_path, dpi=500, bbox_inches='tight')
    plt.tight_layout()
    mplcursors.cursor(multiple=True)
    if show:
        plt.show(block=True)
    elif not show:
        plt.close()
