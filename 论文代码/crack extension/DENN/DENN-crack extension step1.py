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
        return u * (x**2+(y+1)**2)

    def hard_v(self, v, x, y):
        return v  * (y + 1) /2*(y-1)/2+(y+1)/2*a*0.5

    def add_BCPoints(self, num=[500]):
        x_up, y_up = NodesGenerater.genMeshNodes2D(-1+1e-4, 1-1e-4, num[0], 1, 1, 1)
        self.x_up, self.y_up, self.xy_up = self._set_points(x_up, y_up)
        self.up_zero = torch.zeros_like(self.x_up)

    def E_ext(self) -> torch.Tensor:
        u_up, v_up = self.pred_uv(self.xy_up)
        u_up, v_up = self.mm_to_m(u_up, v_up)
        return trapz1D((v_up) * fy, self.x_up)


def get_RELUPSI_extension(crack_points, tip='right'):
    return RELUPSI.RELU2PSIPolyLine(crack_points, tip=tip)


E = 200.0e3
nu = 0.3
fy = 0.0

length = 0.5
a_increment = 0.1
open_thetas_left = [-0.88, ]
open_thetas_right = [-0.16,]
# open_thetas_left = [-0.88,3.89,3.51,11.64,15.63,0.98,-22.67,-40.50 ]
# open_thetas_right = [-0.16,4.35,5.70,10.23,18.36,-2.27,-23.18,-41.79]
'左边裂纹扩展'
x_crackTip2 = -1+length
y_crackTip2 = 0.05
x_crackCenter2 = -1
y_crackCenter2 = 0.05


p0_left = [x_crackCenter2, y_crackCenter2]
p1_left = [x_crackTip2-0.15, y_crackTip2]
p2_left = [x_crackTip2, y_crackTip2]
points_left = [p0_left, p1_left,p2_left]


thetas_left = np.radians(open_thetas_left)
current_point_left = p2_left
for theta in thetas_left:
    next_point_left = [
        current_point_left[0] + a_increment * np.cos(theta),
        current_point_left[1] + a_increment * np.sin(theta)
    ]
    points_left.append(next_point_left)
    current_point_left = next_point_left
'右边裂纹扩展'
x_crackTip1 = 1-length
y_crackTip1 = -0.05
x_crackCenter1 = 1
y_crackCenter1 = -0.05

p0_right = [x_crackCenter1, y_crackCenter1]
p1_right = [x_crackTip1+0.15, y_crackTip1]
p2_right = [x_crackTip1, y_crackTip1]
points_right = [p0_right, p1_right,p2_right]


thetas_right = np.radians(open_thetas_right)
current_point_right = p2_right
for theta in thetas_right:
    next_point_right = [
        current_point_right[0] - a_increment * np.cos(theta),
        current_point_right[1] - a_increment * np.sin(theta)
    ]
    points_right.append(next_point_right)
    current_point_right = next_point_right
get_extension = get_RELUPSI_extension
width = 32
depth = 6


left_crack_extension = get_extension(points_left, tip='right')
right_crack_extension = get_extension(points_right, tip='left')
multiExtension = Enrichment.multiBasis(BasisList=
                                       [left_crack_extension, right_crack_extension])

crack_net = Enrichment.extendAxisNet(
            stack_net(input=4,output=2,activation=nn.Tanh,width=32,depth=6),
             extendAxis= multiExtension)

pinn = Plate_whole(crack_net)
pinn.set_meshgrid_inner_points(-0.9999,0.9998,350,-0.9998,0.9999,350)
pinn.add_BCPoints()
pinn.setMaterial(E= E, nu = nu)

pinn.set_loss_func(losses=[pinn.Energy_loss,],
                   weights=[10000.0,])
pinn.set_Optimizer(0.01)
model_name = 'crack_extension-two-edgecrack-DENN-step2'
a = 3
pinn.train(path=model_name, patience=300, epochs=25000, eval_sep=100)
# pinn.load(path='C:/Users/15844/PycharmProjects/pythonProject4/DEM'
#                '/crack_extension-two-edgecrack-DENN-step1')
# data_path = "C:/Users/15844/Desktop/crack_extension/" \
#             "comsol63crack逐步加载/edge-twocrack-0.5-uhard-step1plot.csv"
# pinn.readData(data_path)

# pinn.evaluate(name = None)
# pinn.plot_result_NN()


from SIF_1 import IIM_circle2D, max_stress_theta

def get_kappa(v):
    return (3 - v) / (1 + v)

def get_mu(E, v):
    return (E / (2 * (1 + v)))
"J积分计算应力强度因子"
#

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kappa = get_kappa(nu)
mu = get_mu(E, nu)

local_axis_right = Enrichment.LocalAxis(
    x0=points_right[-1][0],
    y0=points_right[-1][1],
    beta=open_thetas_right[-1]*np.pi/180 + np.pi
)

local_axis_left = Enrichment.LocalAxis(
    x0=points_left[-1][0],
    y0=points_left[-1][1],
    beta=open_thetas_left[-1]*np.pi/180
)

num = [3000]
r_list = np.arange(0.05, 0.15, 0.01)  # 半径从 0.05 到 0.30
K1_left_list, K2_left_list = [], []
K1_right_list, K2_right_list = [], []
theta_left_list, theta_right_list = [], []
open_left_list, open_right_list = [], []
G_left_list, G_right_list = [], []  #
# ------------------------- 循环计算 -------------------------
for r0 in r_list:
    IIM_pinn_left = IIM_circle2D(r0, local_axis_left, num, int_type='contour')
    IIM_pinn_right = IIM_circle2D(r0, local_axis_right, num, int_type='contour')

    K1_left, K2_left = IIM_pinn_left.compute_K_IIM(pinn, kappa, mu, E)
    K1_right, K2_right = IIM_pinn_right.compute_K_IIM(pinn, kappa, mu, E)

    # 转为数值
    K1_left = K1_left.cpu().detach().numpy().item()/np.sqrt(1000)
    K2_left = K2_left.cpu().detach().numpy().item()/np.sqrt(1000)
    K1_right = K1_right.cpu().detach().numpy().item()/np.sqrt(1000)
    K2_right = K2_right.cpu().detach().numpy().item()/np.sqrt(1000)

    # 计算主应力方向与新裂纹扩展角
    ref_theta_left = max_stress_theta(K1_left, K2_left)
    ref_theta_right = max_stress_theta(K1_right, K2_right)
    open_theta_left = thetas_left[-1] + ref_theta_left
    open_theta_right = thetas_right[-1] + ref_theta_right
    G_left = (K1_left**2 + K2_left**2) / E*1000
    G_right = (K1_right**2 + K2_right**2) / E*1000

    # 存储
    K1_left_list.append(K1_left)
    K2_left_list.append(K2_left)
    K1_right_list.append(K1_right)
    K2_right_list.append(K2_right)
    theta_left_list.append(ref_theta_left)
    theta_right_list.append(ref_theta_right)
    open_left_list.append(open_theta_left)
    open_right_list.append(open_theta_right)
    G_left_list.append(G_left)
    G_right_list.append(G_right)
#
# ------------------------- 汇总输出 -------------------------
print("\n================= J积分路径无关性结果 =================")
print(" r0     |   K1_left    K2_left   |   K1_right   K2_right   |  θ_left(°)  θ_right(°)  |  open_left(°)  open_right(°)  |  G_left     G_right")
print("-------------------------------------------------------------------------------------------------------------------------------")
# for i, r0 in enumerate(r_list):
#     print(f" {r0:5.3f} | {K1_left_list[i]:10.4f} {K2_left_list[i]:10.4f} | "
#           f"{K1_right_list[i]:10.4f} {K2_right_list[i]:10.4f} | "
#           f"{theta_left_list[i]*180/np.pi:10.2f} {theta_right_list[i]*180/np.pi:10.2f} | "
#           f"{open_left_list[i]*180/np.pi:10.2f} {open_right_list[i]*180/np.pi:10.2f}")
for i, r0 in enumerate(r_list):
    print(f" {r0:5.3f} | {K1_left_list[i]:10.4f} {K2_left_list[i]:10.4f} | "
          f"{K1_right_list[i]:10.4f} {K2_right_list[i]:10.4f} | "
          f"{theta_left_list[i]*180/np.pi:10.2f} {theta_right_list[i]*180/np.pi:10.2f} | "
          f"{open_left_list[i]*180/np.pi:10.2f} {open_right_list[i]*180/np.pi:10.2f} | "
          f"{G_left_list[i]:10.6f} {G_right_list[i]:10.6f}")


