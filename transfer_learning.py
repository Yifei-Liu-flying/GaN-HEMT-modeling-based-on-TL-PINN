import torch
import torch.nn as nn


class TransferModel(nn.Module):
    def __init__(self, input_d, output_d):
        super(TransferModel, self).__init__()
        self.input_d = input_d
        self.output_d = output_d

        # input layer
        self.input = nn.Sequential(
            nn.Linear(input_d, 128),
            nn.PReLU()
        )

        # 5 hidden layers
        self.hidden = nn.Sequential(
            nn.Linear(128, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
        )

        # output layer
        self.output = nn.Linear(256, output_d)

    def forward(self, x):
        h = self.input(x)
        h = self.hidden(h)
        y = self.output(h)
        return y

    def get_target_model(self, target_lr):
        """
        get the target model with new output layer and optimizer of the new output layer
        knowledge transfer
        :param target_lr: learning rate of the target model
        :return: TransferModel, torch.optim.Adam
        """
        target_model = TransferModel(self.input_d, self.output_d)
        target_model.load_state_dict(self.state_dict())
        target_model.output = nn.Linear(256, self.output_d)
        target_optimizer = torch.optim.Adam(target_model.output.parameters(), lr=target_lr)
        return target_model, target_optimizer
