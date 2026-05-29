import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import T_co
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

my_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# 筛选出标签不为某些数的数据
def exclude_digits(full_train_dataset, exclude_label: list):
    filtered_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label not in exclude_label]
    return torch.utils.data.Subset(full_train_dataset, filtered_indices)


def plot_distribution(dataset, name: str):
    nums = [0 for i in range(10)]
    for _, label in dataset:
        nums[label] += 1
    plt.bar(range(10), nums, align="center", color="steelblue", alpha=0.6)
    # 添加y轴标签
    plt.ylabel("Count")
    # 添加x轴刻度标签
    plt.xticks(range(10), range(10))
    # 添加标题
    plt.title(name)
    # 显示图形
    plt.show()


def plot_gan_data(path):
    df = pd.read_csv(path)
    vgs = df[df.columns[0]].tolist()
    ids = df[df.columns[1]].tolist()
    plt.plot(vgs, ids)
    plt.show()


def label_to_onehot(labels, num_classes=10):
    """
    将标签转换为one-hot编码

    参数:
        labels: 输入的标签数组，形状为(n_samples,)
        num_classes: 类别数量，MNIST默认为10（数字0-9）

    返回:
        onehot_labels: one-hot编码后的标签数组，形状为(n_samples, num_classes)
    """
    # 获取样本数量
    n_samples = labels.shape[0]

    # 初始化one-hot编码数组
    onehot_labels = np.zeros((n_samples, num_classes))

    # 为每个样本设置对应的1
    onehot_labels[np.arange(n_samples), labels] = 1

    return torch.tensor(onehot_labels, dtype=torch.float32)


def load_mnist(batch_size=64, tl=True):
    train_set = datasets.MNIST("../MNIST/", download=False, train=True, transform=my_transform)
    test_set = datasets.MNIST("../MNIST/", download=False, train=False, transform=my_transform)
    torch.manual_seed(42)  # random seed

    if tl:
        total_length = len(train_set)
        source_size = int(total_length * 0.9)
        target_size = total_length - source_size
        source_set, target_set = random_split(train_set, [source_size, target_size])

        total_length = len(test_set)
        source_size = int(total_length * 0.9)
        target_size = total_length - source_size
        test_source_set, test_target_set = random_split(test_set, [source_size, target_size])

        plot_distribution(source_set, "source training set")
        plot_distribution(target_set, "target training set")

        source_set = DataLoader(source_set, batch_size, shuffle=True)
        target_set = DataLoader(target_set, batch_size, shuffle=True)
        test_source_set = DataLoader(test_source_set, batch_size, shuffle=False)
        test_target_set = DataLoader(test_target_set, batch_size, shuffle=False)

        return source_set, target_set, test_source_set, test_target_set

    else:
        total_length = len(train_set)
        split_size = total_length // 3  # 3 clients
        part1, part2, part3 = random_split(train_set, [split_size] * 3)

        part1 = exclude_digits(part1, [1, 3, 7])
        part2 = exclude_digits(part2, [2, 5, 8])
        part3 = exclude_digits(part3, [4, 6, 9])
        test_set_137 = exclude_digits(test_set, [1, 3, 7])
        test_set_258 = exclude_digits(test_set, [2, 5, 8])
        test_set_469 = exclude_digits(test_set, [4, 6, 9])

        # plot_distribution(part1, "part1")
        # plot_distribution(part2, "part2")
        # plot_distribution(part3, "part3")
        # plot_distribution(test_set_137, "test_set_137")
        # plot_distribution(test_set_258, "test_set_258")
        # plot_distribution(test_set_469, "test_set_469")

        # part1 = DataLoader(part1, batch_size=batch_size, shuffle=True)
        # part2 = DataLoader(part2, batch_size=batch_size, shuffle=True)
        # part3 = DataLoader(part3, batch_size=batch_size, shuffle=True)
        # test_set_137 = DataLoader(test_set_137, batch_size=batch_size, shuffle=False)
        # test_set_258 = DataLoader(test_set_258, batch_size=batch_size, shuffle=False)
        # test_set_469 = DataLoader(test_set_469, batch_size=batch_size, shuffle=False)

        # print(part1[0][0].size())

        return part1, part2, part3, test_set_137, test_set_258, test_set_469, test_set


class HEMTDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.Vgs = df["Vgs"].tolist()
        self.Vds = df["Vds"].tolist()
        self.Ids = df["Ids"].tolist()

    def __getitem__(self, index):
        vgs = self.Vgs[index]
        vds = self.Vds[index]
        ids = self.Ids[index] * 100

        # fea_tensor = torch.from_numpy(np.array([vgs, vds]))
        # ids_tensor = torch.from_numpy(np.array(ids))
        fea_tensor = torch.tensor([vgs, vds], dtype=torch.float32).detach()
        ids_tensor = torch.tensor([ids], dtype=torch.float32).detach()

        return fea_tensor, ids_tensor

    def __len__(self):
        return len(self.Ids)


class MOSFETDataset(Dataset):
    def __init__(self, folder_path):
        folder_list = os.listdir(folder_path)
        self.N_A_list = []
        self.Vgs_list = []
        self.Vds_list = []
        self.Ids_list = []

        for each_csv in folder_list:
            n = each_csv.split(".")[0].split("_")[-1][1:]
            df = pd.read_csv(folder_path + '/' + each_csv)
            self.N_A_list += [float(n)/1e18] * len(df)
            self.Vgs_list += df['Vgs'].tolist()
            self.Vds_list += df['Vds'].tolist()
            self.Ids_list += df['Ids'].tolist()
            # print(len(self.N_A_list), len(self.Vgs_list), len(self.Ids_list), len(self.Vds_list))
            if len(self.N_A_list) != len(self.Vgs_list) or len(self.Vgs_list) != len(self.Vds_list) or len(
                    self.Ids_list) != len(self.Vds_list):
                raise Exception(f"some data are missing in file {each_csv}")

    def __len__(self):
        return len(self.Ids_list)

    def __getitem__(self, item):
        N_A = self.N_A_list[item]
        Vgs = self.Vgs_list[item]
        Vds = self.Vds_list[item]
        Ids = self.Ids_list[item]

        fea_tensor = torch.tensor([N_A, Vgs, Vds], dtype=torch.float32).detach()
        ids_tensor = torch.tensor([Ids], dtype=torch.float32).detach()

        return fea_tensor, ids_tensor

"""
n:  总栅宽
t:  势垒层厚度
x:  AlGaN 势垒层 Al 组分
N:  掺杂浓度
Lg: 栅长
Vd: 漏源电压
Vg: 栅源电压
"""

class GaNHEMTDataset(Dataset):
    def __init__(self, folder_path):
        folder_list = os.listdir(folder_path)
        self.N_A_list = []
        self.Vgs_list = []
        self.Vds_list = []
        self.Ids_list = []

        for each_csv in folder_list:
            n = each_csv[:-4].split("_")[3][1:]
            vds = each_csv[:-4].split("_")[-1][2:]
            df = pd.read_csv(folder_path + '/' + each_csv)
            self.N_A_list += [float(n)/1e18] * len(df)
            self.Vgs_list += df[df.columns[0]].tolist()
            self.Vds_list += [float(vds)] * len(df)
            self.Ids_list += df[df.columns[1]].tolist()
            # print(len(self.N_A_list), len(self.Vgs_list), len(self.Ids_list), len(self.Vds_list))
            if len(self.N_A_list) != len(self.Vgs_list) or len(self.Vgs_list) != len(self.Vds_list) or len(
                    self.Ids_list) != len(self.Vds_list):
                raise Exception(f"some data are missing in file {each_csv}")

    def __len__(self):
        return len(self.Ids_list)

    def __getitem__(self, item):
        N_A = self.N_A_list[item]
        Vgs = self.Vgs_list[item]
        Vds = self.Vds_list[item]
        Ids = self.Ids_list[item]

        fea_tensor = torch.tensor([N_A, Vgs, Vds], dtype=torch.float32).detach()
        ids_tensor = torch.tensor([Ids], dtype=torch.float32).detach()

        return fea_tensor, ids_tensor


def load_data(dataset, batch_size=64):
    start = time.time()
    # dataset = HEMTDataset(path)
    num_train = int(len(dataset) * 0.8)
    num_test = len(dataset) - num_train

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Loading dataset takes {time.time() - start:.3f}s")

    return train_loader, test_loader


def main():
    # plot_gan_data("data/exp_data/n356_t0.020_x0.27_N2e18_Lg0.8_Vd5.csv")
    # dataset = MOSFETDataset("data/MOSFET")
    dataset = GaNHEMTDataset("data/exp_data")
    print(len(dataset), min(dataset.Ids_list), max(dataset.Ids_list))
    train, test = load_data(dataset)
    for X, y in test:
        print(X)
        print(X[:, 1])
        break


if __name__ == "__main__":
    main()
