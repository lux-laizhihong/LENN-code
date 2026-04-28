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

def get_RELUPSI_extension(crack_points, tip='right'):
    return RELUPSI.RELU2PSIPolyLine(crack_points, tip=tip)


class Plate(soft_multidomain.soft_multidomain):
    def add_BCPoints(self, num=[500]):
        x_up, y_up = NodesGenerater.genMeshNodes2D(-1+1e-4, 1-1e-4, num[0], 1, 1, 1)
        self.x_up, self.y_up, self.xy_up = self._set_points(x_up, y_up)
        self.up_zero = torch.zeros_like(self.x_up)

    def E_ext(self) -> torch.Tensor:
        u_up, v_up = self.pred_uv(self.xy_up)
        u_up, v_up = self.mm_to_m(u_up, v_up)
        return trapz1D((v_up) * fy, self.x_up)


E = 200.0e3
nu = 0.3
fy = 0.0

length = 0.5
a_increment = 0.1

open_thetas_left = [1.44,]
open_thetas_right = [1.51,]

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
width = 25
depth = 6


left_crack_extension = get_extension(points_left, tip='right')
right_crack_extension = get_extension(points_right, tip='left')
multiExtension = Enrichment.multiBasis(BasisList=
                                       [left_crack_extension, right_crack_extension])

crack_whole = Enrichment.extendAxisNet(
        net = AxisScalar2D(
            stack_net(input=4,output=2,activation=nn.Tanh,width=25,depth=4),
            A=torch.tensor([1,1,1.0,1.0]),
            B=torch.tensor([0,0,0.0,0.0])
            ),
        extendAxis= multiExtension)
plate_whole = Plate_whole(model=crack_whole)
local_area = 0.3


local_area_center_x_left = (points_left[-1][0]+points_left[-2][0])/2
local_area_center_y_left = (points_left[-1][1]+points_left[-2][1])/2
scalar = 1/local_area
translate_x_left = -1*local_area_center_x_left/local_area
translate_y_left = -1*local_area_center_y_left/local_area
kernel_left = soft_multidomain.cubic_BSpline_kernel(local_area_center_x_left, local_area_center_y_left, local_area)

crack_net_left = Enrichment.extendAxisNet(
        net = AxisScalar2D(
            stack_net(input=3,output=2,activation=nn.Tanh,width=width,depth=depth),
            A=torch.tensor([scalar,scalar,1.0]),
            B=torch.tensor([translate_x_left,translate_y_left,0.0])
            ),
        extendAxis= left_crack_extension)

plate_center_left = soft_multidomain.weighted_DEM(model=crack_net_left,
                                                  weight=kernel_left)


local_area_center_x_right = (points_right[-1][0] + points_right[-2][0]) / 2
local_area_center_y_right = (points_right[-1][1] + points_right[-2][1]) / 2
scalar = 1 / local_area
translate_x_right = -1 * local_area_center_x_right / local_area
translate_y_right = -1 * local_area_center_y_right / local_area

kernel_right = soft_multidomain.cubic_BSpline_kernel(local_area_center_x_right, local_area_center_y_right, local_area)

crack_net_right = Enrichment.extendAxisNet(
    net = AxisScalar2D(
        stack_net(input=3, output=2, activation=nn.Tanh, width=width, depth=depth),
        A=torch.tensor([scalar, scalar, 1.0]),
        B=torch.tensor([translate_x_right, translate_y_right, 0.0])
    ),
    extendAxis = right_crack_extension
)

plate_center_right = soft_multidomain.weighted_DEM(model=crack_net_right,
                                                   weight=kernel_right)

pinn = Plate([plate_whole,plate_center_left,plate_center_right])
pinn.set_meshgrid_inner_points(-0.9999,0.9998,350,-0.9998,0.9999,350)
pinn.add_BCPoints()
pinn.setMaterial(E= E, nu = nu)

pinn.set_loss_func(losses=[pinn.Energy_loss,],
                   weights=[10000.0,])
pinn.set_Optimizer(0.01)
model_name = 'crack_extension-two-edgecrack-domain-step1-disload'
a = 3
pinn.train(path=model_name, patience=300, epochs=100000, eval_sep=100)

# pinn.load(path=model_path)
# pinn.load(path='C:/Users/15844/PycharmProjects/pythonProject4/DEM/crack_extension-length0.2-domain-0.1-step1')
# pinn.load(path='C:/Users/15844/PycharmProjects/pythonProject4/DEM'
#                '/crack_extension-two-edgecrack-domain-step1-disload')
# data_path = "C:/Users/15844/Desktop/crack_extension/" \
#             "comsol63crack逐步加载/edge-twocrack-0.5-uhard-step1plot.csv"
# pinn.readData(data_path)
# pinn.plot_result_FEM()
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
r_list = np.arange(0.10, 0.20, 0.01)  # 半径从 0.05 到 0.30
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




'积分随点数变化'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# kappa = get_kappa(nu)
# mu = get_mu(E, nu)
#
# local_axis_right = Enrichment.LocalAxis(
#     x0=points_right[-1][0],
#     y0=points_right[-1][1],
#     beta=open_thetas_right[-1]*np.pi/180 + np.pi
# )
#
# local_axis_left = Enrichment.LocalAxis(
#     x0=points_left[-1][0],
#     y0=points_left[-1][1],
#     beta=open_thetas_left[-1]*np.pi/180
# )
#
# # 固定积分半径
# r0 = 0.15  # mm
#
# # 积分点数从 1000 到 10000，每次步长 1000
# num_list = np.arange(1000, 11000, 1000).astype(int)
#
# K1_left_list, K2_left_list = [], []
# K1_right_list, K2_right_list = [], []
# theta_left_list, theta_right_list = [], []
# open_left_list, open_right_list = [], []
#
# # ------------------------- 循环计算 -------------------------
# for n in num_list:
#     num = [int(n)]  # 生成 [n] 形式的列表
#     IIM_pinn_left = IIM_circle2D(r0, local_axis_left, num, int_type='contour')
#     IIM_pinn_right = IIM_circle2D(r0, local_axis_right, num, int_type='contour')
#
#     K1_left, K2_left = IIM_pinn_left.compute_K_IIM(pinn, kappa, mu, E)
#     K1_right, K2_right = IIM_pinn_right.compute_K_IIM(pinn, kappa, mu, E)
#
#     # 转为数值
#     K1_left = K1_left.cpu().detach().numpy().item()/np.sqrt(1000)
#     K2_left = K2_left.cpu().detach().numpy().item()/np.sqrt(1000)
#     K1_right = K1_right.cpu().detach().numpy().item()/np.sqrt(1000)
#     K2_right = K2_right.cpu().detach().numpy().item()/np.sqrt(1000)
#
#     # 计算主应力方向与新裂纹扩展角
#     ref_theta_left = max_stress_theta(K1_left, K2_left)
#     ref_theta_right = max_stress_theta(K1_right, K2_right)
#     open_theta_left = thetas_left[-1] + ref_theta_left
#     open_theta_right = thetas_right[-1] + ref_theta_right
#
#     # 存储
#     K1_left_list.append(K1_left)
#     K2_left_list.append(K2_left)
#     K1_right_list.append(K1_right)
#     K2_right_list.append(K2_right)
#     theta_left_list.append(ref_theta_left)
#     theta_right_list.append(ref_theta_right)
#     open_left_list.append(open_theta_left)
#     open_right_list.append(open_theta_right)
#
# # ------------------------- 汇总输出 -------------------------
# print("\n================= J积分点数敏感性结果 =================")
# print(" num     |   K1_left    K2_left   |   K1_right   K2_right   |  θ_left(°)  θ_right(°)  |  open_left(°)  open_right(°)")
# print("-------------------------------------------------------------------------------------------------------------")
#
# for i, n in enumerate(num_list):
#     print(f" {n:5d} | {K1_left_list[i]:10.4f} {K2_left_list[i]:10.4f} | "
#           f"{K1_right_list[i]:10.4f} {K2_right_list[i]:10.4f} | "
#           f"{theta_left_list[i]*180/np.pi:10.2f} {theta_right_list[i]*180/np.pi:10.2f} | "
#           f"{open_left_list[i]*180/np.pi:10.2f} {open_right_list[i]*180/np.pi:10.2f}")



# kappa = get_kappa(nu)
# mu = get_mu(E, nu)
# r0 = 0.05
# local_axis_right = Enrichment.LocalAxis(
#     x0=points_right[-1][0], y0=points_right[-1][1], beta=open_thetas_right[-1]*np.pi/180+np.pi)
#
# local_axis_left = Enrichment.LocalAxis(
#     x0=points_left[-1][0], y0=points_left[-1][1], beta=open_thetas_left[-1]*np.pi/180)
# num = [5000]
#
#
# IIM_pinn_left = IIM_circle2D(r0,local_axis_left,num,int_type = 'contour')
# IIM_pinn_right = IIM_circle2D(r0,local_axis_right,num,int_type = 'contour')
# sigma_pi_a = fy * np.sqrt(np.pi * 0.1 * 1000)
#
#
# K1_left, K2_left = IIM_pinn_left.compute_K_IIM(pinn, kappa, mu, E)
# K1_left = K1_left.cpu().detach().numpy();K2_left = K2_left.cpu().detach().numpy()
# K1_right, K2_right = IIM_pinn_right.compute_K_IIM(pinn, kappa, mu, E)
# K1_right = K1_right.cpu().detach().numpy();K2_right = K2_right.cpu().detach().numpy()
# print("K1_left =", K1_left.item(), "K2_left =", K2_left.item())
# print("K1_right =", K1_right.item(), "K2_right =", K2_right.item())
#
#
# ref_theta_left = max_stress_theta(K1_left, K2_left)
# ref_theta_right = max_stress_theta(K1_right, K2_right)
# print(ref_theta_left * 180 / np.pi,ref_theta_right * 180 / np.pi)
# open_theta_left = thetas_left[-1] + ref_theta_left
# open_theta_right = thetas_right[-1] + ref_theta_right
# print(open_theta_left * 180 / np.pi,open_theta_right * 180 / np.pi)







from SIF import DispExpolation_homo,DispExpolation_homo_lzh,DispExpolation_homo_data_FEM
#
# kappa = get_kappa(nu)
# mu = get_mu(E, nu)
#
#
# crack_surface_extension = get_extension(points_left, tip='right')
#
# crack_surface = LineSegement(points_left[-2], points_left[-1])

# crack_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.04),
#                                       crack_surface.clamp(dist2=0.03))

# pinn.model[0].set_extend_axis(multiExtension)
# pinn.model[1].set_extend_axis(left_crack_extension)
# pinn.model[2].set_extend_axis(right_crack_extension)
# len = np.sqrt(1000)


# K1, K2 = DispExpolation_homo(pinn,
#                              left_crack_extension,
#                              crack_surface, 8,
#                              Geometry.LocalAxis(points_left[-1][0], points_left[-1][1],
#                                                 beta=crack_surface.tangent_theta),
#                              kappa, mu)

# print(K1/len, K2/len)


# crack_surface = Geometry.LineSegement(crack_surface.clamp(dist2=0.08),
#                                       crack_surface.clamp(dist2=0.06))

# K1, K2 = DispExpolation_homo_lzh(pinn,
#                              left_crack_extension,
#                              crack_surface, 18,
#                              Geometry.LocalAxis(points_left[-1][0], points_left[-1][1],
#                                                 beta=crack_surface.tangent_theta),
#                              kappa, mu,beta = 0.002)
# print(K1/len, K2/len)

# K1, K2 = DispExpolation_homo(pinn,
#                              left_crack_extension,
#                              crack_surface, 18,
#                              Geometry.LocalAxis(points_left[-1][0], points_left[-1][1],
#                                                 beta=crack_surface.tangent_theta),
#                              kappa, mu)
# print(K1/len, K2/len)



# print(ref_theta*180/np.pi)
# open_theta = crack_surface.tangent_theta.numpy() + ref_theta
# print(open_theta * 180 / np.pi)
'NN张开位移计算'

#
# import copy
# d_values = np.arange(0.04, 0.091, 0.005)
# d_fixed = 0.03
# results = []
#
# for d in d_values:
#     # 深拷贝原始 crack_surface，防止 clamp 改变原对象
#     surface = copy.deepcopy(crack_surface)
#
#     # clamp 裁剪出新的裂纹线段
#     surface = Geometry.LineSegement(
#         surface.clamp(dist2=d),
#         surface.clamp(dist2=d_fixed)
#     )
#
#     # 计算 K1, K2
#     # K1, K2 = DispExpolation_homo_lzh(
#     #     pinn,
#     #     left_crack_extension,
#     #     surface, 18,
#     #     Geometry.LocalAxis(points_left[-1][0], points_left[-1][1],
#     #                        beta=surface.tangent_theta),
#     #     kappa, mu,
#     #     beta=0.002
#     # )
#     K1, K2 = DispExpolation_homo(pinn,
#                                  left_crack_extension,
#                                  surface, 8,
#                                  Geometry.LocalAxis(points_left[-1][0], points_left[-1][1],
#                                                     beta=crack_surface.tangent_theta),
#                                  kappa, mu)
#
#     # 转为标量并归一化
#     K1n = float(np.array(K1).flatten()[0]) / np.sqrt(1000)
#     K2n = float(np.array(K2).flatten()[0]) / np.sqrt(1000)
#     results.append((d, K1n, K2n))
#     print(f"dist2 = {d:.3f} -> K1 = {K1n:.4f}, K2 = {K2n:.4f}")

'NN查看裂纹表面位移'
# import pandas as pd
#
# file_path = "C:/Users/15844/Desktop/displace/uvtrue.csv"
#
# # 读取 CSV 文件（逗号分隔）
# df = pd.read_csv(file_path, names=['x', 'y', 'dv', 'du'])
#
# # 转换为数值类型，去除空行
# df = df.apply(pd.to_numeric, errors='coerce').dropna()
#
# # 转为 PyTorch tensor
# x = torch.tensor(df['x'].values, dtype=torch.float32, requires_grad=True)
# y = torch.tensor(df['y'].values, dtype=torch.float32, requires_grad=True)
# dv = torch.tensor(df['dv'].values, dtype=torch.float32)
# du = torch.tensor(df['du'].values, dtype=torch.float32)
# x, y, xy = pinn._set_points(x, y)
# u,v = pinn.pred_uv(xy)
# pinn.model[0].set_extend_axis(multiExtension)
# pinn.model[1].set_extend_axis(left_crack_extension)
# pinn.model[2].set_extend_axis(right_crack_extension)
# left_crack_extension.set_ls(-1)
# u,v = pinn.pred_uv(xy)
# print(du,dv)



'有限元张开位移部分'
# import torch
# import pandas as pd
#
# file_path = "C:/Users/15844/Desktop/displace/uvtrue.csv"
#
# # 读取 CSV 文件（逗号分隔）
# df = pd.read_csv(file_path, names=['x', 'y', 'dv', 'du'])
#
# # 转换为数值类型，去除空行
# df = df.apply(pd.to_numeric, errors='coerce').dropna()
#
# # 转为 PyTorch tensor
# x = torch.tensor(df['x'].values, dtype=torch.float32, requires_grad=True)
# y = torch.tensor(df['y'].values, dtype=torch.float32, requires_grad=True)
# dv = torch.tensor(df['dv'].values, dtype=torch.float32)
# du = torch.tensor(df['du'].values, dtype=torch.float32)
#
#
# x0 = points_left[-1][0]
# y0 = points_left[-1][1]
#
# # 计算每个点到 (x0, y0) 的欧几里得距离
# r = torch.sqrt((x - x0)**2 + (y - y0)**2)
#
# # 选择距离在 [0.3, 0.5] 范围内的点
# mask = (r >= 0.03) & (r <= 0.05)
# # 筛选相应数据
# x_sel = x[mask]
# y_sel = y[mask]
# dv_sel = dv[mask]*1000
# du_sel = du[mask]*1000
#
# import matplotlib.pyplot as plt
#
#
# K1, K2 = DispExpolation_homo_data_FEM(x_sel,y_sel,du_sel,dv_sel,
#                              Geometry.LocalAxis(points_left[-1][0], points_left[-1][1],
#                                                 beta=crack_surface.tangent_theta),
#                              kappa, mu)
# print(K1/len, K2/len)






'有限元导出的张开位移计算'
#
#
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt
#
# file_path = "C:/Users/15844/Desktop/displace/1/uvtrue.csv"
#
# # 读取 CSV 文件
# df = pd.read_csv(file_path, names=['x', 'y', 'dv', 'du'])
# df = df.apply(pd.to_numeric, errors='coerce').dropna()
# df[['x', 'y']] = df[['x', 'y']].round(3)
# # 转为 torch 张量
# x = torch.tensor(df['x'].values, dtype=torch.float32, requires_grad=True)
# y = torch.tensor(df['y'].values, dtype=torch.float32, requires_grad=True)
# dv = torch.tensor(df['dv'].values, dtype=torch.float32)
# du = torch.tensor(df['du'].values, dtype=torch.float32)
#
# # 已知参考点
# x0 = points_left[-1][0]
# y0 = points_left[-1][1]
#
# # 计算距离
# r = torch.sqrt((x - x0)**2 + (y - y0)**2)
#
# # 定义 a, b 的范围
# a_values = torch.arange(0.01, 0.041, 0.005)  # [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04]
# b_values = torch.arange(0.05, 0.091, 0.005)  # [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09]
#
# results = []
#
#
# scale_dv = 0.98 + (1.00 - 0.98) * torch.rand(56)
# scale_du = 0.95 + (1.00 - 0.95) * torch.rand(56)
# for a in a_values:
#     for b in b_values:
#         mask = (r >= a) & (r <= b)
#         if mask.sum() == 0:
#             continue  # 没有点就跳过
#
#         x_sel = x[mask]
#         y_sel = y[mask]
#         dv_sel = dv[mask] * 1000*scale_dv[mask]
#         du_sel = du[mask] * 1000*scale_du[mask]
#
#         # 调用你的函数计算 K1, K2
#         K1, K2 = DispExpolation_homo_data_FEM(
#             x_sel, y_sel, du_sel, dv_sel,
#             Geometry.LocalAxis(x0, y0, beta=crack_surface.tangent_theta),
#             kappa, mu
#         )
#
#         results.append((float(a), float(b), float(K1/len), float(K2/len)))
#
# table = pd.DataFrame(results, columns=['a', 'b', 'K1/len', 'K2/len'])

# 美观地打印表格
# print("\n=== 结果表格 ===")
# print(table.to_string(index=False, justify='center', float_format="{:.5f}".format))
#
# 可选：保存结果到文件
# table_export = table.copy()
# table_export['a'] = table_export['a'].round(3)
# table_export['b'] = table_export['b'].round(3)
# table_export.to_csv("C:/Users/15844/Desktop/K_results.csv", index=False, float_format="%.5f")

# print("\n结果已保存为 C:/Users/15844/Desktop/K_results.csv")

'自己画图可调colorbar'
# pinn.set_meshgrid_inner_points(-0.95,0.9998,1,0.05,0.9999,1)
# u,v = pinn.pred_uv(pinn.XY)
# print(u)
#
#
# def to_np(tensor):
#     if hasattr(tensor, 'detach'):
#         return tensor.detach().cpu().numpy()
#     else:
#         return np.array(tensor)
# xy = to_np(pinn.labeled_xy)
# X, Y = xy[:, 0], xy[:, 1]
#
# u, v = pinn.pred_uv(pinn.labeled_xy)
# sxx, syy, sxy = pinn.pred_stress(pinn.labeled_xy)
#
# u = to_np(u)
# v = to_np(v)
# sxx = to_np(sxx)
# syy = to_np(syy)
# sxy = to_np(sxy)
#
# u_d = to_np(pinn.labeled_u)
# v_d = to_np(pinn.labeled_v)
# sxx_d = to_np(pinn.labeled_sx)
# syy_d = to_np(pinn.labeled_sy)
# sxy_d = to_np(pinn.labeled_sxy)
# x = xy[:,0]
# y = xy[:,1]
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# def plot_field_2x2(x, y, pred, data, name, n=201, rel_max=0.2, abs_max=None):
#
#     ny = nx = n
#
#     # 转 numpy 并清理异常值
#     x = np.array(x)
#     y = np.array(y)
#     pred = np.nan_to_num(np.array(pred), nan=0.0, posinf=0.0, neginf=0.0)
#     data = np.nan_to_num(np.array(data), nan=0.0, posinf=0.0, neginf=0.0)
#
#     # reshape
#     X = x.reshape(ny, nx)
#     Y = y.reshape(ny, nx)
#     Z_pred = pred.reshape(ny, nx)
#     Z_data = data.reshape(ny, nx)
#
#     # 误差计算
#     abs_err = np.abs(Z_pred - Z_data)
#     abs_err = np.nan_to_num(abs_err, nan=0.0, posinf=0.0, neginf=0.0)
#     rel_err = abs_err / (np.max(np.abs(Z_data) + 1e-3))
#     rel_err = abs_err / (np.abs(Z_data) + 1e-3)
#     rel_err = np.nan_to_num(rel_err, nan=0.0, posinf=0.0, neginf=0.0)
#
#     # 数据范围
#     vmin = min(np.min(Z_pred), np.min(Z_data))
#     vmax = max(np.max(Z_pred), np.max(Z_data))
#     levels_data = np.linspace(vmin, vmax, 200)
#
#     # 误差范围控制
#     if abs_max is None:
#         abs_max = np.max(abs_err)
#     levels_err_abs = np.linspace(0, abs_max, 200)
#     levels_err_rel = np.linspace(0, rel_max, 200)
#
#     # 绘图
#     plt.close()
#     fig, axes = plt.subplots(2, 2, figsize=(6, 5))
#
#     im1 = axes[0, 0].contourf(X, Y, Z_pred, levels=levels_data, cmap='jet', extend='both')
#     axes[0, 0].set_title(f'{name} (PINN)')
#     plt.colorbar(im1, ax=axes[0, 0])
#
#     im2 = axes[0, 1].contourf(X, Y, Z_data, levels=levels_data, cmap='jet', extend='both')
#     axes[0, 1].set_title(f'{name} (Data)')
#     plt.colorbar(im2, ax=axes[0, 1])
#
#     im3 = axes[1, 0].contourf(X, Y, abs_err, levels=levels_err_abs, cmap='jet', extend='both')
#     axes[1, 0].set_title(f'|{name}_PINN - {name}_Data|')
#     plt.colorbar(im3, ax=axes[1, 0])
#
#     im4 = axes[1, 1].contourf(X, Y, rel_err, levels=levels_err_rel, cmap='jet', extend='both')
#     axes[1, 1].set_title(f'Relative Error of {name}')
#     plt.colorbar(im4, ax=axes[1, 1])
#
#     for ax in axes.flat:
#         ax.set_aspect('equal')
#         ax.axis('off')
#
#     plt.tight_layout(pad=0.3)
#     plt.show()
#     plt.close(fig)
#
# n=301
# plot_field_2x2(x, y, u,   u_d,   'u',   n=n, rel_max=0.05, abs_max=0.01)
# plot_field_2x2(x, y, v,   v_d,   'v',   n=n, rel_max=0.05, abs_max=0.01)
# plot_field_2x2(x, y, sxx, sxx_d, 'σxx', n=n, rel_max=0.2, abs_max=1)
# plot_field_2x2(x, y, syy, syy_d, 'σyy', n=n, rel_max=0.2, abs_max=1)
# plot_field_2x2(x, y, sxy, sxy_d, 'σxy', n=n, rel_max=0.2, abs_max=1)

