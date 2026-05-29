import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

import time
from datetime import datetime
from datasets import label_to_onehot
from build_MOSFET_dataset import bsim3_mosfet_ids, mos_params


def evaluate_accuracy(data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    mse_func = nn.MSELoss()
    acc_sum, n = 0.0, 0
    with torch.no_grad():  # 不便更参数
        for X, y in data_iter:
            net.eval()  # 开启评估模式
            X = X.to(device)
            y = y.to(device)
            y_pre = net(X)
            acc_sum += mse_func(y_pre, y).float().cpu().item()   # .to(device)数据传到对应设备，.cpu()数据传到CPU上
            # print(acc_sum)
            net.train()  # 改回训练模式
            # n += y.shape[0]
            n += 1
    return acc_sum / n


def train_ch5(net, train_iter, test_iter, optimizer, device, num_epochs, loss_fuc):
    net = net.to(device)
    print("train on ", device)
    # loss = nn.CrossEntropyLoss()
    loss = loss_fuc
    batch_count = 0
    loss_list = []
    test_mse_list = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y, in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            # print(y_hat)
            # train_acc_sum += (y_hat == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net, device=device)
        print('epoch %d, loss %.4f, test MSE %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, test_acc, time.time() - start))
        loss_list.append(train_l_sum / batch_count)
        test_mse_list.append(test_acc)
    return loss_list, test_mse_list


def train_gan(discriminator, generator, dataloader, optimizer_D, optimizer_G,
              epochs, latent_dim, device):
    # 损失函数
    source_loss = nn.BCELoss()
    classify_loss = nn.CrossEntropyLoss()

    discriminator.to(device)
    generator.to(device)
    print("train on ", device)
    start = time.time()
    d_loss_list, g_loss_list = [], []
    d_each_loss, g_each_loss, d_loss_average_list, g_loss_average_list = [], [], [], []

    for epoch in range(epochs):
        d_loss, g_loss = 0.0, 0.0
        d_loss_sum, g_loss_sum, c_loss_sum = 0.0, 0.0, 0.0
        batch_num = 0
        for real_data, condition in dataloader:
            real_data = real_data.view(real_data.size(0), -1)
            batch_size = real_data.size(0)
            # condition = label_to_onehot(condition)

            # 创建真实和伪造标签
            valid = torch.ones(batch_size, 1).to(device)
            # print(valid.size())
            fake = torch.zeros(batch_size, 1).to(device)

            # 训练判别器
            discriminator.train()
            generator.eval()

            # 对真数据求loss_real
            real_validity, real_pred_label = discriminator(real_data.to(device))  # 判别器得出结果
            real_loss_s = source_loss(real_validity, valid)
            # print(condition.dtype)
            real_loss_c = classify_loss(real_pred_label, condition.to(device))

            # 对生成数据求loss_fake
            d_z = torch.randn(batch_size, latent_dim).to(device)  # 随机高斯分布
            d_conditions = label_to_onehot(condition).to(device)  # 保持相同的标签条件
            d_fake_data = generator(d_z, d_conditions)  # 生成器生成假数据
            d_fake_validity, d_fake_pred_label = discriminator(d_fake_data)  # 判别器得出假数据结果
            fake_loss_s = source_loss(d_fake_validity, fake)
            # print(condition.dtype)
            fake_loss_c = classify_loss(d_fake_pred_label, condition.to(device))

            # 计算判别器loss + 判别器反向传播
            d_loss_s = real_loss_s + fake_loss_s
            d_loss_c = real_loss_c + fake_loss_c
            d_loss = d_loss_s + d_loss_c  # 判别器整体loss
            optimizer_D.zero_grad()  # 判别器优化器梯度清零
            # print(d_loss.dtype)
            d_loss.backward()  # 反向传播
            optimizer_D.step()  # 优化器优化参数

            c_loss_sum += d_loss_c.item()

            # 训练生成器
            discriminator.eval()  # 判别器进入评估模式（不更新参数）
            generator.train()

            z = torch.randn((batch_size, latent_dim))
            z = z.to(device)
            g_condition = d_conditions.to(device)
            fake_data = generator(z, g_condition)
            fake_validity, fake_pred_label = discriminator(fake_data)

            # 计算生成器loss + 生成器反向传播
            g_loss_s = source_loss(fake_validity, valid)
            g_loss_c = classify_loss(fake_pred_label, condition.to(device))
            # print(g_loss1, g_loss2)
            g_loss = g_loss_c + g_loss_s
            optimizer_G.zero_grad()
            # g_loss.backward(retain_graph=True)  # 保留计算图以便多次反向传播
            g_loss.backward()
            optimizer_G.step()

            d_loss_sum += d_loss.item()
            g_loss_sum += g_loss.item()
            batch_num += 1

        d_each_loss.append(d_loss_sum / batch_num)
        g_each_loss.append(g_loss_sum / batch_num)
        cls_loss_average = c_loss_sum / batch_num
        # d_loss_list.append(d_loss_sum / batch_num)
        # g_loss_list.append(g_loss_sum / batch_num)

        d_loss_average = sum(d_each_loss) / len(d_each_loss)
        g_loss_average = sum(g_each_loss) / len(g_each_loss)
        d_loss_average_list.append(d_loss_average)
        g_loss_average_list.append(g_loss_average)
        d_each_loss = []
        g_each_loss = []
        print(
            f'Epoch {epoch + 1}/{epochs} - D Loss: {d_loss_average:.6f}, G Loss: {g_loss_average:.6f}, '
            f'cls Loss: {cls_loss_average}, use {time.time() - start:.2f}s')
        start = time.time()


def evaluate_accuracy_with_physic(data_iter, net, physic_loss, lamda=0.5,
                                  device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    mse_func = nn.MSELoss()
    acc_sum = 0.0
    n = 0
    # with torch.no_grad():  # 不便更参数
    for X, y in data_iter:
        net.eval()  # 开启评估模式
        X = X.to(device).requires_grad_(True)
        y = y.to(device).requires_grad_(True)
        y_pre = net(X)
        data_acc = mse_func(y_pre, y)
        physic_acc = physic_loss(y_pre, X)
        acc_sum += (data_acc + lamda * physic_acc).float().cpu().item()    # .to(device)数据传到对应设备，.cpu()数据传到CPU上
        # print(acc_sum)
        net.train()  # 改回训练模式
        # n += y.shape[0]
        n += 1
    return acc_sum / n


def train_with_physic(net, train_iter, test_iter, optimizer, device, num_epochs, data_loss, physic_loss, lamda=0.5):
    net = net.to(device)
    print("train on ", device)
    batch_count = 0
    loss_list = []
    test_loss_list = []
    test_mse_list = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device).requires_grad_(True)
            y = y.to(device).requires_grad_(True)
            y_hat = net(X)
            data_l = data_loss(y_hat, y)  # data based loss
            physic_l = physic_loss(y_hat, X)  # physic based loss
            # print(net.state_dict())
            loss = data_l + lamda * physic_l  # total loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_l_sum += loss.cpu().item()
            # train_acc_sum += (y_hat == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_loss = evaluate_accuracy_with_physic(test_iter, net, physic_loss=physic_loss, lamda=lamda, device=device)
        test_mse = evaluate_accuracy(test_iter, net, device=device)
        print('epoch %d, loss %.4f, test loss %.4f, test MSE %.4f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, test_loss, test_mse, time.time() - start))
        loss_list.append(train_l_sum / batch_count)
        test_mse_list.append(test_mse)
        test_loss_list.append(test_loss)
    return loss_list, test_mse_list, test_loss_list


def show_net_result_mosfet(net: nn.Module, Vds, Vgs_steps, device):
    net = net.to(device)
    # Vds_steps = np.linspace(0.0, 4.95, 100)
    ids_pre = []
    ids_bsim = []
    for vgs in Vgs_steps:
        fea_tensor = torch.tensor([mos_params['N_A'] / 1e18, vgs, Vds], dtype=torch.float32).detach().to(device)
        ids_pre.append(net(fea_tensor).cpu().item())
        ids_bsim.append(bsim3_mosfet_ids(vgs, Vds, mos_params))

    ids_pre = np.array(ids_pre)
    ids_bsim = np.array(ids_bsim)
    # print(ids_pre)
    # print(ids_bsim)
    # print(ids_pre - ids_bsim)

    plt.scatter(Vgs_steps, ids_pre, label=f'pre', s=5, c="r")
    plt.scatter(Vgs_steps, ids_bsim, label=f'bsim', s=5, c="b")
    plt.xlabel('V_GS (V)', fontsize=11)
    plt.ylabel('I_DS (mA)', fontsize=11)
    plt.title('transfer feature of NMOS (Vd=5v)', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(-1, 3)
    plt.ylim(0, max([max(ids_bsim), max(ids_pre)]) * 1.1)
    plt.tight_layout()
    plt.show()


def show_net_result_gan(net: nn.Module, device):
    net = net.to(device)
    ids_pre = []

    df = pd.read_csv("data/exp_data/n12_t0.020_x0.25_N2e18_Lg0.8_Vd5.csv")
    N_A = 2
    Vds = 5
    Vgs_steps = df[df.columns[0]].tolist()
    ids_test = df[df.columns[1]].tolist()

    for vgs in Vgs_steps:
        fea_tensor = torch.tensor([N_A, vgs, Vds], dtype=torch.float32).detach().to(device)
        ids_pre.append(net(fea_tensor).cpu().item())

    ids_pre = np.array(ids_pre)
    ids_test = np.array(ids_test)
    # print(ids_pre)
    # print(ids_test)
    # print(ids_pre - ids_test)

    plt.scatter(Vgs_steps, ids_pre, label=f'pre', s=5, c="r")
    plt.scatter(Vgs_steps, ids_test, label=f'test dataset', s=5, c="b")
    plt.xlabel('V_GS (V)', fontsize=11)
    plt.ylabel('I_DS (mA)', fontsize=11)
    plt.title('transfer feature of GaN HEMT', fontsize=12, fontweight='bold')
    plt.legend(loc='upper right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.xlim(-5, 2)
    plt.ylim(0, max([max(ids_test), max(ids_pre)]) * 1.1)
    plt.tight_layout()
    plt.show()


def save_loss_list(loss_list, test_list, folder_path):
    df = pd.DataFrame({"train loss": loss_list, "test loss": test_list})
    csv_name = folder_path + "/" + datetime.now().strftime("%Y_%m_%d_%H_%M") + ".csv"
    df.to_csv(csv_name, index=False)


def plot_loss(loss_list, test_list):
    plt.plot(range(1, len(loss_list) + 1), loss_list, label="train set")
    plt.plot(range(1, len(loss_list) + 1), test_list, label="test set")
    plt.legend(loc='upper right')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()


def plot_mse(mse_list):
    plt.plot(range(1, len(mse_list) + 1), mse_list)
    plt.xlabel("epoch")
    plt.ylabel("MSE")
    plt.show()


def load_net(net, params_path):
    net.load_state_dict(torch.load(params_path))
