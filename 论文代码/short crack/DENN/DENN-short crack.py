
from Elasticity2D import DEM2D_2
import torch
import torch.nn as nn
from NodesGenerater import AcceptanceSampling2D,plot_points,genMeshNodes2D,genRandomNodes2D
from NN import stack_net,MultilayerNN,AxisScalar2D
from Integral import trapz1D,montecarlo
from Geometry import LineSegement
import Enrichment
import NodesGenerater
import Geometry
import RELUPSI
import soft_multidomain
import NN
from get_grad import get_grad
import numpy as np


class Plate_whole(DEM2D_2):

    def hard_u(self, u, x, y):
        return u*x
        # return u*(x ** 2 + (y + 1) ** 2)
        # return u * (y + 1) / 2

    def hard_v(self, v, x, y):
        return v * (y + 1) / 2

    def add_BCPoints(self, num=[100]):
        x_up, y_up = NodesGenerater.genMeshNodes2D(1e-4-1, 1 - 1e-4, num[0], 1, 1, 1)
        self.x_up, self.y_up, self.xy_up = self._set_points(x_up, y_up)
        self.up_zero = torch.zeros_like(self.x_up)

    def E_ext(self) -> torch.Tensor:
        u, v = self.pred_uv(self.xy_up)
        u, v = self.mm_to_m(u, v)
        return trapz1D(v * fy, self.x_up)
        # return trapz1D((u + v) * fy, self.x_up)


E = 100.0e3
nu = 0.3
fy = 10.0

crack_center_x_1 = 0
crack_center_y_1 = 0
beta = 0
beta_1 = beta/360*2*torch.pi

half_length = 0.1
a_x_1 = np.cos(beta_1) * half_length
a_y_1 = np.sin(beta_1) * half_length

crack_extension_1 = RELUPSI.RELU2PSILine(xy0=[crack_center_x_1-a_x_1, crack_center_y_1-a_y_1],
                                         xy1=[crack_center_x_1+a_x_1, crack_center_y_1+a_y_1],
                                         tip='both')


# crack_net = Enrichment.extendAxisNet(
#         net = AxisScalar2D(
#             stack_net(input=3,output=2,activation=nn.Tanh,width=32,depth=6),
#             A=torch.tensor([1,1,1.0]),
#             B=torch.tensor([0,0,0.0])
#             ),
#         extendAxis= crack_extension_1)
crack_net = Enrichment.extendAxisNet(
            stack_net(input=3,output=2,activation=nn.Tanh,width=32,depth=6),
        extendAxis= crack_extension_1)

# total_params = sum(p.numel() for p in crack_net.parameters() if p.requires_grad)
# print(f"\n总可训练参数数量: {total_params}")

pinn = Plate_whole(crack_net)
pinn.add_BCPoints()


pinn.setMaterial(E=E, nu=nu)

pinn.set_meshgrid_inner_points(0.0001-1, 0.99999, 300, 0.001-1, 0.9999, 300)


pinn.set_loss_func(losses=[pinn.Energy_loss], weights=[10000.0,])
pinn.set_Optimizer(0.008)
# pinn.load(path='C:/Users/15844/PycharmProjects/pythonProject4/DEM/test')
model_name = 'crack-no-softdomain'
pinn.train(path=model_name, patience=300, epochs=50000, eval_sep=100)
# pinn.load(path='C:/Users/15844/PycharmProjects/pythonProject4/DEM/crack-no-softdomain')

# data_path = f"C:/Users/15844/Desktop/crack-s/one-bottom-2-0.1/2-0.1-{beta}-local.csv"

# pinn.readData(data_path)
#
# pinn.plot_result_NN()
# pinn.plot_result_FEM()
# pinn.evaluate(name=None)




"J积分计算应力强度因子"
from SIF_1 import IIM_circle2D, max_stress_theta

def get_kappa(v):
    return (3 - v) / (1 + v)


def get_mu(E, v):
    return (E / (2 * (1 + v)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kappa = get_kappa(nu)
mu = get_mu(E, nu)
r0 = 0.075
local_axis = Enrichment.LocalAxis(x0=crack_center_x_1+a_x_1, y0=crack_center_y_1+a_y_1, beta=beta_1)
# local_axis = Enrichment.LocalAxis(x0=crack_center_x_1-a_x_1, y0=crack_center_y_1-a_y_1, beta=beta_1+np.pi)
num = [5000]

# IIM_pinn = IIM_circle2D(r0,local_axis,num,int_type = 'domain')
IIM_pinn = IIM_circle2D(r0,local_axis,num,int_type = 'contour')
# pinn.todevice()

sigma_pi_a = fy * np.sqrt(np.pi * 0.1 * 1000)
K1, K2 = IIM_pinn.compute_K_IIM(pinn, kappa, mu, E)
K1 = K1.cpu().detach().numpy();K2 = K2.cpu().detach().numpy()
ref_theta = max_stress_theta(K1, K2)
print(ref_theta * 180 / np.pi)
# open_theta = beta_1 + ref_theta
# print(open_theta * 180 / np.pi)
print(ref_theta*180/np.pi+beta)

K1 = K1/sigma_pi_a;K2 = K2/sigma_pi_a
print("K1 =", K1.item(), "K2 =", K2.item())
print(sigma_pi_a)
