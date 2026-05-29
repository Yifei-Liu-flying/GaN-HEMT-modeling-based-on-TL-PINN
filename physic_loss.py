import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicLoss(nn.Module):
    def __init__(self, V_th0=-3, delta_V_th=0.005, alpha_T=2, T0=25, T_amb=25, R_th=20,
                 I_pk=0.5, V_pk1=-3, V_pk2=-1, m_ipk=2.5, alpha=0.2, k_ipk=-0.3):
        super(PhysicLoss, self).__init__()

        self.V_th0 = V_th0
        self.delta_V_th = delta_V_th
        self.alpha_T = alpha_T
        self.T0 = T0
        self.T_amb = T_amb
        self.R_th = R_th
        self.I_pk = I_pk
        self.V_pk1 = V_pk1
        self.V_pk2 = V_pk2
        self.m_ipk = m_ipk
        self.alpha = alpha
        self.k_ipk = k_ipk

    def forward(self, pres, inputs):
        """
        Angelov model based physic loss
        :param pres: predict I_ds
        :param inputs: V_ds, V_gs
        :return: nn.MSELoss
        """
        mse = nn.MSELoss()
        tanh = nn.Tanh()
        T_j = self.T_amb + self.R_th * pres * inputs[0]
        delta_T = T_j - self.T0
        V_th = self.V_th0 + self.delta_V_th * inputs[0] + self.alpha_T * delta_T
        V_gseff = inputs[1] - V_th
        p_I1 = (1 + (self.k_ipk * delta_T) / 273.15)
        p_I2 = (1 - ((self.V_pk1 - V_gseff) / (self.V_pk2 - V_gseff)) ** self.m_ipk)
        physic_I_ds = self.I_pk * p_I1 * p_I2 * tanh(self.alpha * inputs[0])
        # print(physic_I_ds)
        return mse(pres, physic_I_ds)

    # def physic_model(self, inputs):
    #     # I = 1
    #     # last_I = 0
    #     # while abs(I - last_I)
    #     tanh = nn.Tanh()
    #     T_j = self.T_amb + self.R_th * pres * inputs[0]
    #     delta_T = T_j - self.T0
    #     V_th = self.V_th0 + self.delta_V_th * inputs[0] + self.alpha_T * delta_T
    #     V_gseff = inputs[1] - V_th
    #     p_I1 = (1 + (self.k_ipk * delta_T) / 273.15)
    #     p_I2 = (1 - ((self.V_pk1 - V_gseff) / (self.V_pk2 - V_gseff)) ** self.m_ipk)
    #     physic_I_ds = self.I_pk * p_I1 * p_I2 * tanh(self.alpha * inputs[0])
        # print(physic_I_ds)


# Monotonicity constraint loss (gm = dId/dVg >= 0 & gds = dId/dVd >= 0)
class MonotonicityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Id_pred, x):
        # 确保输入需要梯度
        if not x.requires_grad:
            x.requires_grad_(True)

        # print(Id_pred.size())

        I_grad = torch.autograd.grad(
            outputs=Id_pred,
            inputs=x,
            grad_outputs=torch.ones_like(Id_pred),
            create_graph=True,
            retain_graph=True,
        )[0]
        # print(I_grad.size())

        # print(I_grad)
        loss_gm = F.relu(-I_grad[:, 1])
        # print(loss_gm)
        loss_gds = F.relu(-I_grad[:, 2])
        # print(loss_gds)
        loss = torch.mean(loss_gm + loss_gds)
        return loss


def test():
    x = torch.tensor([[1., 2, 3], [1, 3, 2]], requires_grad=True)
    print(x.size())
    y = torch.tensor([-14., -13], requires_grad=True)
    print(y)
    loss = MonotonicityLoss()
    # print(loss(y, b.clone().detach().requires_grad_(True), c.clone().detach().requires_grad_(True), ))
    print(loss(y, x))


if __name__ == "__main__":
    test()
