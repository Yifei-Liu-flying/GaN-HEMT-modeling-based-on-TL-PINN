import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import lr_scheduler  # 学习速率衰减

import datasets
from federated_learning import Client, Server, SimpleModel
from ACGAN import ACGAN
import train_function
from physic_loss import MonotonicityLoss
from transfer_learning import TransferModel
from datetime import datetime

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def fl():
    batch_size = 64
    lr = 1e-3
    gan_lr = 3e-5
    server_epochs = 3
    client_epochs = 2
    acgan_epochs = 30
    loss_fuc = nn.CrossEntropyLoss()

    part1, part2, part3, test_set_137, test_set_258, test_set_469, test_set = \
        datasets.load_mnist(batch_size=batch_size, tl=False)
    test_set = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    client1 = Client(1, 10, train_set=part1, test_set=test_set_137, lr=lr, device=device,
                     loss_fuc=loss_fuc, batch_size=batch_size)
    client2 = Client(1, 10, train_set=part2, test_set=test_set_258, lr=lr, device=device,
                     loss_fuc=loss_fuc, batch_size=batch_size)
    client3 = Client(1, 10, train_set=part3, test_set=test_set_469, lr=lr, device=device,
                     loss_fuc=loss_fuc, batch_size=batch_size)
    server = Server([client1, client2, client3],
                    input_dim=1, output_dim=10,
                    test_set=test_set, device=device)
    acgan1 = ACGAN(fea_dim=28 * 28, lab_dim=10, latent_dim=32, lr_D=gan_lr, lr_G=gan_lr,
                   data_set=part1, batch_size=batch_size, epochs=acgan_epochs, device=device)

    server.federated_learning_loop(server_epochs, client_epochs)

    # acgan1.train_acgan()


# transfer learning
def tl(source_net_params=None, target_net_params=None, train_without_phy=False):
    batch_size = 64
    source_epochs = 10
    target_epochs = 300
    source_loss_fuc = nn.MSELoss()
    target_loss_fuc = nn.MSELoss()
    physic_loss_fuc = MonotonicityLoss()

    mosfet_dataset = datasets.MOSFETDataset("data/MOSFET")
    gan_dataset = datasets.GaNHEMTDataset("data/exp_data")
    source_set, test_source_set = datasets.load_data(mosfet_dataset, batch_size=batch_size)
    target_set, test_target_set = datasets.load_data(gan_dataset, batch_size=batch_size)

    # train source model
    print("train source model")
    source_lr = 1e-3
    source_model = TransferModel(3, 1)
    if source_net_params:
        print("load source model parameters from " + source_net_params)
        source_model.load_state_dict(torch.load(source_net_params))
    source_optimizer = torch.optim.Adam(source_model.parameters(), lr=source_lr)

    source_loss_list, source_test_mse_list = train_function.train_ch5(
        source_model, source_set, test_source_set, source_optimizer, device=device,
        num_epochs=source_epochs, loss_fuc=source_loss_fuc
    )

    Vgs_transfer = np.linspace(-0.5, 2.5, 100)
    train_function.show_net_result_mosfet(source_model, 5, Vgs_transfer, device)
    train_function.plot_loss(source_loss_list, source_test_mse_list)
    train_function.plot_mse(source_test_mse_list)
    is_save = input("save loss and parameters of source net? n/y: ")
    if is_save == "y" or is_save == "Y":
        train_function.save_loss_list(loss_list=source_loss_list,test_list=source_test_mse_list, folder_path="loss/source_loss")
        torch.save(source_model.state_dict(),
                   "parameters/source_parameters/" + datetime.now().strftime("%Y_%m_%d_%H_%M") + ".pkl")

    # knowledge transfer and train target model
    print("transfer learning")
    target_lr = 1e-3
    target_model, target_optimizer = source_model.get_target_model(target_lr)
    if target_net_params:
        print("load target model parameters from " + target_net_params)
        target_model.load_state_dict(torch.load(target_net_params))

    if train_without_phy:
        target_loss_list, target_test_mse_list = train_function.train_ch5(
            target_model, target_set, test_target_set,
            optimizer=target_optimizer, device=device,
            num_epochs=target_epochs, loss_fuc=target_loss_fuc
        )
        train_function.show_net_result_gan(target_model, device)
        train_function.plot_loss(target_loss_list, target_test_mse_list)
        train_function.plot_mse(target_test_mse_list)
        is_save = input("save loss and parameters of target net? n/y: ")
        if is_save == "y" or is_save == "Y":
            train_function.save_loss_list(loss_list=target_loss_list, test_list=target_test_mse_list, folder_path="loss/target_loss")
            torch.save(target_model.state_dict(),
                       "parameters/target_parameters/" + datetime.now().strftime("%Y_%m_%d_%H_%M") + ".pkl")
    else:
        target_loss_list, target_test_mse_list, target_test_loss_list = train_function.train_with_physic(
            target_model, target_set, test_target_set,
            optimizer=target_optimizer, device=device,
            num_epochs=target_epochs, data_loss=target_loss_fuc,
            physic_loss=physic_loss_fuc, lamda=1
        )
        train_function.show_net_result_gan(target_model, device)
        train_function.plot_loss(target_loss_list, target_test_loss_list)
        train_function.plot_mse(target_test_mse_list)
        is_save = input("save loss and parameters of target net? n/y: ")
        if is_save == "y" or is_save == "Y":
            train_function.save_loss_list(loss_list=target_loss_list, test_list=target_test_loss_list, folder_path="loss/target_loss")
            torch.save(target_model.state_dict(),
                       "parameters/target_parameters/" + datetime.now().strftime("%Y_%m_%d_%H_%M") + ".pkl")




def ml(net_params=None):
    batch_size = 64
    epochs = 300
    loss_fuc = nn.MSELoss()
    physic_loss_fuc = MonotonicityLoss()

    gan_dataset = datasets.GaNHEMTDataset("data/exp_data")
    train_set, test_set = datasets.load_data(gan_dataset, batch_size=batch_size)

    # train source model
    print("train source model")
    lr = 1e-3
    model = TransferModel(3, 1)
    if net_params:
        print("load source model parameters from " + net_params)
        model.load_state_dict(torch.load(net_params))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_list, test_mse_list = train_function.train_ch5(
        model, train_set, test_set, optimizer, device=device,
        num_epochs=epochs, loss_fuc=loss_fuc
    )

    train_function.show_net_result_gan(model, device)
    train_function.plot_loss(loss_list, test_mse_list)
    train_function.plot_mse(test_mse_list)
    is_save = input("save loss and parameters of source net? n/y: ")
    if is_save == "y" or is_save == "Y":
        train_function.save_loss_list(loss_list=loss_list, test_list=test_mse_list, folder_path="loss/target_loss")
        torch.save(model.state_dict(),
                   "parameters/target_parameters/" + datetime.now().strftime("%Y_%m_%d_%H_%M") + ".pkl")


if __name__ == "__main__":
    # fl()
    # tf & PINN:    2026_04_06_16_39 / 2026_05_25_15_14
    # tf:           2026_04_06_17_04 / 2026_05_26_20_51
    # ml:           2026_04_06_17_16 / 2026_05_26_20_32
    tl(
        source_net_params="parameters/source_parameters/2026_05_25_15_13.pkl",
        train_without_phy=True
    )
    # ml()
