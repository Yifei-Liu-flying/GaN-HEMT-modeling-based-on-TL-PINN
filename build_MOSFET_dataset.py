import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

q = 1.6e-19


def bsim3_mosfet_ids(Vgs, Vds, params, use_mA=True):
    """
    简化BSIM3模型计算NMOS的漏极电流I_DS
    :param Vgs: 栅源电压 (V)
    :param Vds: 漏源电压 (V)
    :param params: 模型参数字典（物理/结构参数）
    :param use_mA: Ids的单位是否为毫安
    :return: I_DS: 漏极电流 (A)
    """
    # 提取模型参数
    # Vth = params['Vth']               # 阈值电压 (V)
    mu_n = params['mu_n']               # 电子迁移率 (m²/V·s)
    Cox = params['Cox']                 # 栅氧化层单位面积电容 (F/m²)
    W = params['W']                     # 栅宽 (m)
    L = params['L']                     # 栅长 (m)
    lambda_ = params['lambda']          # 沟道长度调制系数 (1/V)
    N_A = params['N_A']                 # 参杂浓度
    V_FB = params['V_FB']               # 平带电压 (V)
    phi_F = params['phi_F']             # 费米势 (V)
    epsilon_si = params['epsilon_si']   # 硅介电常数 (F/cm²)

    # 计算跨导参数 K = μₙCₒₓW/L
    K = mu_n * Cox * (W / L)

    # 计算Vth
    Vth = V_FB + 2 * phi_F + (4 * epsilon_si * q * N_A * phi_F) ** 0.5 / (Cox * 1e-4)

    # 1. 截止区：Vgs < Vth，电流近似为0（忽略泄漏电流）
    if Vgs < Vth:  # 留0.1V裕度，避免噪声
        return 0.0

    # 2. 饱和区判断：Vds ≥ Vgs - Vth（有效栅压）
    Vds_sat = Vgs - Vth  # 饱和漏压
    if Vds >= Vds_sat:
        ids = 0.5 * K * (Vgs - Vth) ** 2 * (1 + lambda_ * Vds)

    # 3. 线性区：Vds < Vgs - Vth
    else:
        ids = K * ((Vgs - Vth) * Vds - 0.5 * Vds ** 2) * (1 + lambda_ * Vds)

    if use_mA:
        return ids * 1000
    return ids


# 转移特性：固定Vds（饱和区），扫Vgs
def transfer_characteristics():
    Vds_fixed = 5.0  # 固定漏压（确保工作在饱和区）
    Vgs_transfer = np.linspace(-0.5, 2.5, 100)  # 栅压范围：-0.5V~2.5V
    Ids_transfer = [bsim3_mosfet_ids(vgs, Vds_fixed, mos_params) for vgs in Vgs_transfer]

    plt.plot(Vgs_transfer, np.array(Ids_transfer), 'b-', linewidth=2.5)
    plt.xlabel('V_GS (V)', fontsize=11)
    plt.ylabel('I_DS (mA)', fontsize=11)
    plt.title('transfer characteristics of NMOS with (V_DS=5V, n=1e18)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(-0.5, 2.5)
    plt.ylim(0, max(Ids_transfer) * 1.1)
    plt.show()


# 输出特性：扫Vds，不同Vgs步进
def output_characteristics(Vgs_steps, Vds_output, draw=False):
    Ids_output = []
    for vgs in Vgs_steps:
        ids_vds = []
        for vds in Vds_output:
            ids_vds.append(bsim3_mosfet_ids(vgs, vds, mos_params))
        Ids_output.append(ids_vds)

    if draw:
        for i, vgs in enumerate(Vgs_steps):
            # plt.scatter(Vds_output, np.array(Ids_output[i]), label=f'V_GS={vgs:.2f}V', s=5)
            plt.scatter(Vds_output, np.array(Ids_output[i]), s=5)
        plt.xlabel('漏源电压 V_DS (V)', fontsize=11)
        plt.ylabel('漏极电流 I_DS (mA)', fontsize=11)
        plt.title('NMOS 输出特性', fontsize=12, fontweight='bold')
        plt.legend(loc='upper right', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 5.0)
        plt.ylim(0, max([max(ids) for ids in Ids_output]) * 1.1)

        plt.tight_layout()
        plt.show()

    return Ids_output


def save_csv(Vgs_steps, Vds_output, Ids_outputs, path):
    vds_len = len(Vds_output)
    ids_len = len(Ids_outputs)
    if len(Vgs_steps) != ids_len:
        raise Exception("the number of Ids not equal to the number of Vgs")

    output_Vgs = []
    for each in Vgs_steps:
        output_Vgs.extend([each] * vds_len)

    output_Vds = Vds_output.tolist() * ids_len

    output_Ids = []
    for each in Ids_outputs:
        if len(each) != vds_len:
            raise Exception("the number of Ids not equal to the number of Vds")
        output_Ids.extend(each)

    data = {
        "Vgs": output_Vgs,
        "Vds": output_Vds,
        "Ids": output_Ids
    }

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)    # , encoding='utf-8-sig')


mos_params = {
    'Vth': 0.7,                 # 阈值电压 (V)，NMOS典型值0.5-1.0V
    'mu_n': 600e-4,             # 电子迁移率 (m²/V·s)，硅基NMOS典型值400-800e-4
    'Cox': 6.9e-3,              # 栅氧化层电容 (F/m²)，对应氧化层厚度~10nm（Cox=ε₀εᵣ/tₒₓ）
    'W': 10e-6,                 # 栅宽 (m)，10μm
    'L': 1e-6,                  # 栅长 (m)，1μm
    'lambda': 0.02,             # 沟道长度调制系数 (1/V)，短沟道器件更大（0.01-0.1）
    'N_A': 1e18,                # 参杂浓度
    'V_FB': -0.2,               # 平带电压 (V)
    'phi_F': 0.1,               # 费米势 (V)
    'epsilon_si': 1.035e-12     # 硅介电常数 (F/cm²)
}


def main():
    transfer_characteristics()
    # Vds_steps = np.linspace(0.0, 4.95, 100)  # 漏压范围：0~5V
    # Vgs_steps = np.linspace(0.8, 2.7, 100)  # 20个栅压点（覆盖亚阈值到饱和区）
    #
    # Ids_output = output_characteristics(Vgs_steps, Vds_steps, False)
    # data_name = f"MOSFET_data_n{mos_params['N_A']}"
    # save_csv(Vgs_steps, Vds_steps, Ids_output, f"data/{data_name}.csv")

    # for _ in range(6):
    #     mos_params['N_A'] += 1e18
    #     Ids_output = output_characteristics(Vgs_steps, Vds_steps, False)
    #     data_name = f"MOSFET_data_n{mos_params['N_A']}"
    #     save_csv(Vgs_steps, Vds_steps, Ids_output, f"data/{data_name}.csv")


if __name__ == "__main__":
    main()
