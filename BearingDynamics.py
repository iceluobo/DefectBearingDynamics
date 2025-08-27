import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.integrate import solve_ivp
import BearingModel
import matplotlib
import dynamics_plot
import time
import pandas
import mplcursors


def load_K(filename):
    with open(f'{filename}.txt', 'r') as f:
        lines = f.readlines()
        ki = int(lines[0].strip())
        ko = int(lines[1].strip())
    return ki, ko


def Dynamics_bearing(fs=1e5, T=1, **kwargs):
    params = dict(
        filename='NJ208',
        Nb=14, # 滚动体数目
        mi=15, # 内圈部分质量
        mo=6, # 外圈部分质量
        Kbh=1e7, # 外圈接地刚度
        Cbh=1e3, # 外圈接地阻尼
        Fr=400, # 内圈所受径向载荷
        RPM=400, # 内圈转速
        Rm=30.25e-3, # 轴承节圆半径
        Rr=5.5e-3, # 滚动体半径
        c=20e-6, # 轴承径向游隙
        # 外滚道缺陷
        Bo=0, # 故障宽度
        phi_oc=0, # 故障中心角度，0-pi内为承载区
        # 内滚道缺陷
        Bi=0, # 内圈故障宽度
        phi_oci=0,# 故障中心角度
        # 滚动体缺陷
        Br=0, # 滚动体故障宽度
        alpha_b=0.0, # 滚动体故障中心与径向方向夹角
        j=1, # 故障滚动体编号
    )
    model = []
    params.update(kwargs)
    Ki, Ko = load_K(params['filename'])
    Ke = (1 / ((1 / Ki) ** (2 / 3) + (1 / Ko) ** (2 / 3))) ** (3 / 2)
    wi = params['RPM'] / 60 * 2 * np.pi
    wc = wi / 2 * (1 - params['Rr'] / params['Rm'])
    wr = wi * params['Rm'] / 2 / params['Rr'] * (1 - (params['Rr'] / params['Rm']) ** 2)
    Ro = params['Rm'] + params['Rr']
    Ri = params['Rm'] - params['Rr']
    fs = int(fs)
    t_span = np.linspace(0, T, T * fs + 1)
    DOF = 4
    y0 = np.zeros(DOF * 2)
    # 外滚道缺陷参数
    dC = 0.0
    phi_do = 0.0
    if params['Bo'] > 0:
        model.append('OuterDefect')
        Cd = params['Rr'] - np.sqrt(params['Rr'] ** 2 - (params['Bo'] / 2) ** 2)
        Co = Ro - np.sqrt(Ro ** 2 - (params['Bo'] / 2) ** 2)
        dC = Cd - Co
        phi_do = np.arcsin(params['Bo'] / 2 / Ro)
    # 内滚道缺陷参数
    dCi = 0.0
    phi_di = 0.0
    if params['Bi'] > 0:
        model.append('InnerDefect')
        Cd = params['Rr'] - np.sqrt(params['Rr'] ** 2 - (params['Bi'] / 2) ** 2)
        Ci = Ri - np.sqrt(Ri ** 2 - (params['Bi'] / 2) ** 2)
        dCi = Cd - Ci
        phi_di = np.arcsin(params['Bi'] / 2 / Ri)
    # 滚动体缺陷参数
    phi_b = 0.0
    Hbomax = 0.0
    Hbimax = 0.0
    if params['Br'] > 0:
        model.append('RollerDefect')
        phi_b = 2 * np.arcsin(params['Br'] / 2 / params['Rr'])
        wr = wi * params['Rm'] / 2 / params['Rr'] * (1 - (params['Rr'] / params['Rm']) ** 2)
        Hbomax = params['Rr'] - np.sqrt(params['Rr'] ** 2 - (params['Br'] / 2) ** 2) \
                 + Ro - np.sqrt(Ro ** 2 - (params['Br'] / 2) ** 2)
        Hbimax = params['Rr'] - np.sqrt(params['Rr'] ** 2 - (params['Br'] / 2) ** 2) \
                 + Ri - np.sqrt(Ri ** 2 - (params['Br'] / 2) ** 2)
    # 计算
    result = solve_ivp(
        BearingModel.ode_system,
        [0, T], y0, t_eval=t_span,
        args=(params['Fr'], params['Nb'], params['mi'], params['mo'],
              wi, wc, Ke, Ke, params['Kbh'], params['Cbh'], params['c'],
              dC, phi_do, params['phi_oc'],
              dCi, phi_di, params['phi_oci'],
              Hbomax, Hbimax, phi_b, wr, params['alpha_b'], params['j'])
    )
    dy = BearingModel.y_2_acc(
        result.t, result.y,
        params['Fr'], params['Nb'], params['mi'], params['mo'],
        wi, wc, Ke, Ke, params['Kbh'], params['Cbh'], params['c'],
        dC, phi_do, params['phi_oc'],
        dCi, phi_di, params['phi_oci'],
        Hbomax, Hbimax, phi_b, wr, params['alpha_b'], params['j']
    )
    if not model:
        model = ['health']
    model = '_'.join(model)
    return result.t, dy, model


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    # 仿真时间，采样频率
    T = 5
    fs = 1e5
    start = time.time()
    filename = 'NJ208'
    # 计算
    '''
    示例(故障设置参数可留空)：
    t, y, model = Dynamics_bearing(fs=fs, T=T, filename=filename,
                                   W=400, RPM=400, # 载荷设置
                                   Bo=0.2e-3, phi_oc=np.pi / 2, # 外圈故障参数
                                   Bi=0.2e-3, phi_oci=0, # 内圈故障参数
                                   Br=0.2e-3, alpha_b=0.0, j=1) # 滚动体故障参数
    '''
    # t, y, model = Dynamics_bearing(fs=fs, T=T, filename=filename) # 健康
    # t, y, model = Dynamics_bearing(fs=fs, T=T, filename=filename, Bo=0.2e-3, phi_oc=np.pi / 2)  # 外
    # t, y, model = Dynamics_bearing(fs=fs, T=T, filename=filename, Bi=0.2e-3, phi_oci=0)  # 内
    # t, y, model = Dynamics_bearing(fs=fs, T=T, filename=filename, Br=0.2e-3, alpha_b=0, j=1) # 滚
    t, y, model = Dynamics_bearing(fs=fs, T=T, filename=filename,
                                   Bo=0.2e-3, phi_oc=np.pi / 2,
                                   Bi=0.2e-3, phi_oci=0)  # 外+内复合
    index = 11  # 5
    signal = y[index, int(1 * fs):]
    dynamics_plot.plt_time_domain(signal, fs, show=False, xlim=0.5,
                                  img_save_path=f"{filename}/{model}_t.png",
                                  title=model)
    dynamics_plot.plt_envelope_spectrum(signal, fs, show=False, xlim=500,
                                        img_save_path=f"{filename}/{model}_f.png",
                                        title=model)

    end = time.time()
    elapsed = end - start  # 秒
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"运行时间: {minutes} 分 {seconds:.2f} 秒")
    data = pandas.DataFrame(y[10:12, int(1 * fs):int(2 * fs + 1)])
    data.to_csv(f"{filename}/DATA_{model}.csv", index=False, header=False)
    # 每一行分别为xi,yi,xo,yo通道的振动加速度
    print(f'数据保存至{filename}下对应csv文件')
