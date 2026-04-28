from Elasticity2D import DEM2D_2
import torch
import torch.nn as nn
from NodesGenerater import AcceptanceSampling2D, plot_points, genMeshNodes2D, genRandomNodes2D
from NN import stack_net, MultilayerNN, AxisScalar2D
from Integral import trapz1D, montecarlo
from Geometry import LineSegement
import Enrichment
import NodesGenerater
import Geometry
import RELUPSI
import soft_multidomain
import NN
from get_grad import get_grad
import numpy as np

'''如果修改角度 其对应硬边界条件也需要进行修改'''
class Plate_whole(DEM2D_2):

    def hard_u(self, u, x, y):
        return u*x
        # return u * (x**2+(y+1)**2)
        # return u * (1+x) * (1-x)
    def hard_v(self, v, x, y):
        return v * (y + 1) / 2


plate_whole = Plate_whole(model=stack_net(input=2, output=2, activation=nn.Tanh,
                                          width=25, depth=4, net=NN.ResidualNet))
# total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print(f"\n总可训练参数数量: {total_params}")

class Plate(soft_multidomain.soft_multidomain):
    def todevice(self):
        device_cpu = "cpu"
        return self.model.to(device_cpu)


    def add_BCPoints(self, num=[256]):
        x_up, y_up = NodesGenerater.genMeshNodes2D(1e-4 - 1, 1 - 1e-4, num[0], 1, 1, 1)
        self.x_up, self.y_up, self.xy_up = self._set_points(x_up, y_up)
        self.up_zero = torch.zeros_like(self.x_up)

    def E_ext(self) -> torch.Tensor:
        u, v = self.pred_uv(self.xy_up)
        u, v = self.mm_to_m(u, v)
        # return trapz1D(v * 10.0, self.x_up)
        return trapz1D(v * fy, self.x_up)


E = 100.0e3
nu = 0.3
fy = 10.0
# first crack
crack_center_x_1 = 0
crack_center_y_1 = 0

beta = 0
beta_1 = beta / 360 * 2 * torch.pi
"""若需要修改局部区域大小，则在此处进行修改"""
local_area = 0.4
half_length = 0.1
width = 25
depth = 6
a_x_1 = np.cos(beta_1) * half_length
a_y_1 = np.sin(beta_1) * half_length

scalar_1 = 1 / local_area
translate_x1 = -1 * crack_center_x_1 / local_area
translate_y1 = -1 * crack_center_y_1 / local_area
"""若需要修改窗口函数，则在此处进行修改"""
kernel_1 = soft_multidomain.cubic_BSpline_kernel(crack_center_x_1, crack_center_y_1, local_area)

crack_extension_1 = RELUPSI.RELU2PSILine(xy0=[crack_center_x_1 - a_x_1, crack_center_y_1 - a_y_1],
                                         xy1=[crack_center_x_1 + a_x_1, crack_center_y_1 + a_y_1],
                                         tip='both')

crack_net_1 = Enrichment.extendAxisNet(
    net=AxisScalar2D(
        stack_net(input=3, output=2, activation=nn.Tanh, width=width, depth=depth),
        A=torch.tensor([scalar_1, scalar_1, 1.0]),
        B=torch.tensor([translate_x1, translate_y1, 0.0])
    ),
    extendAxis=crack_extension_1)
# total_params = sum(p.numel() for p in crack_net_1.parameters() if p.requires_grad)
# print(f"\n总可训练参数数量: {total_params}")


plate_center_1 = soft_multidomain.weighted_DEM(model=crack_net_1,
                                               weight=kernel_1)

pinn = Plate([plate_whole, plate_center_1])
pinn.set_meshgrid_inner_points(-0.99, 0.999, 300, -0.999, 0.9999, 300)

pinn.add_BCPoints()

pinn.setMaterial(E=E, nu=nu)

# u,v = pinn.pred_uv(pinn.XY)
# eXX,eYY,eXY = pinn.compute_Strain(u,v,pinn.XY)
# sx,sy,sxy = pinn.constitutive(eXX,eYY,eXY)
# print(sx.size())
# print(eXX.size())
# print(pinn.get_energy_density(pinn.XY).size())


pinn.set_loss_func(losses=[pinn.Energy_loss,
                           #    pinn.Equilibrium_loss,
                           ],
                   weights=[10000.0,
                            #    1e-6,1e-6
                            ]
                   )
pinn.set_Optimizer(0.008)

# model_name_all = '2-0.1-crack'
# model_name = f"{model_name_all}-beta={beta}"
# model_path = f"C:/Users/15844/PycharmProjects/pythonProject4/DEM/{model_name}"
# pinn.load(path=model_path)
pinn.train()

# data_path = f"C:/Users/15844/Desktop/crack-s/one-bottom-2-0.1/2-0.1-{beta}-local.csv"
# pinn.readData(data_path)
# pinn.plot_result_NN()
# pinn.plot_result_FEM()
pinn.evaluate(name=None)

#----------------------------------------------------------------------------------
"J积分计算应力强度因子"
from SIF_1 import IIM_circle2D

def get_kappa(v):
    return (3 - v) / (1 + v)


def get_mu(E, v):
    return (E / (2 * (1 + v)))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kappa = get_kappa(nu)
mu = get_mu(E, nu)
r0 = 0.075
local_axis = Enrichment.LocalAxis(x0=crack_center_x_1+a_x_1, y0=crack_center_y_1+a_y_1, beta=beta_1)
num = [5000]

IIM_pinn = IIM_circle2D(r0,local_axis,num,int_type = 'contour')
# pinn.todevice()

sigma_pi_a = fy * np.sqrt(np.pi * 0.1 * 1000)
K1, K2 = IIM_pinn.compute_K_IIM(pinn, kappa, mu, E)

K1 = K1/sigma_pi_a;K2 = K2/sigma_pi_a
print("K1 =", K1.item(), "K2 =", K2.item())

K1_J, K2_J = IIM_pinn.compute_K_J(pinn, E)
print("K1_J =", K1_J.item(), "K2_J =", K2_J.item())



# -------------------------------------------------------------------------
'张开位移求解应力强度因子'
#
#
#
from SIF import DispExpolation_homo
def get_kappa(v):
    return (3 - v) / (1 + v)
def get_mu(E, v):
    return (E / (2 * (1 + v)))

kappa = get_kappa(nu)
mu = get_mu(E, nu)




crack_surface = LineSegement([crack_center_x_1-a_x_1, crack_center_y_1-a_y_1], [crack_center_x_1+a_x_1, crack_center_y_1+a_y_1])

# extrapolation_surface = Geometry.LineSegement(crack_surface.clamp(dist1=0.04)
#                                             ,crack_surface.clamp(dist1=0.049))

# local_axis = Enrichment.LocalAxis(x0=crack_center_x_1-a_x_1, y0=crack_center_y_1-a_y_1, beta=beta_1)

extrapolation_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.049)
                                            ,crack_surface.clamp(dist2=0.04))

local_axis = Enrichment.LocalAxis(x0=crack_center_x_1+a_x_1, y0=crack_center_y_1+a_y_1, beta=beta_1)


K1, K2 = DispExpolation_homo(pinn,
                            crack_extension_1,
                            extrapolation_surface, 8,
                            local_axis,
                            kappa, mu)


sigma_pi_a = fy * np.sqrt(np.pi * 0.1 * 1000)
# print(sigma_pi_a)
len = np.sqrt(1000)

print(K1 / sigma_pi_a)
print(K2 / sigma_pi_a)

# print(K1)
# print(K2)

