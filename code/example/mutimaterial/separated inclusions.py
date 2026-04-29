import Elasticity2D_muti
import torch
import torch.nn as nn
import stats
import NodesGenerater
from NN import stack_net, MultilayerNN
import NN
from Integral import trapz1D, montecarlo
# from torch.nn.functional import relu
from activation import ReLU2
import Enrichment
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, Polygon
from multidomain import LSNet
from Geometry import LineSegement
import Geometry
import RELUPSI
from get_grad import get_grad
import soft_multidomain
from NN import AxisScalar2D
from NN import AxisScalar2D_withinput
import soft_multidomain
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Plate_whole(Elasticity2D_muti.DEM_bimaterial_muti):

    def hard_u(self, u, x, y):
        return u * y
        # return u * x * (1 - x)

    def hard_v(self, v, x, y):
        return v * y


class Plate(soft_multidomain.soft_multidomain_Bi_muti):

    def add_BCPoints(self, num=[256]):
        x_up, y_up = NodesGenerater.genMeshNodes2D(1e-4, 1 - 1e-4, num[0], 1, 1, 1)
        self.x_up, self.y_up, self.xy_up = self._set_points(x_up, y_up)
        self.up_zero = torch.zeros_like(self.x_up)

        x_down, y_down = NodesGenerater.genMeshNodes2D(1e-3, 1 - 1e-3, num[0], 0, 0, 1)

        x_left, y_left = NodesGenerater.genMeshNodes2D(0, 0, 1, 1e-3, 1 - 1e-3, num[0])

        x_right, y_right = NodesGenerater.genMeshNodes2D(1, 1, 1, 1e-3, 1 - 1e-3, num[0])

        x_tau = torch.cat((x_up, x_down, x_left, x_right))
        y_tau = torch.cat((y_up, y_down, y_left, y_right))
        self.x_tau, self.y_tau, self.xy_tau = self._set_points(x_tau, y_tau)

    def soft_BC_loss(self):
        _, _, sxy = self.pred_stress(self.xy_tau)
        sxy_loss = self.criterion(sxy, torch.zeros_like(sxy))
        _, sy_up, _ = self.pred_stress(self.xy_up)
        sy_up_loss = self.criterion(sy_up - self.fy, torch.zeros_like(sy_up))
        return torch.stack([sxy_loss, sy_up_loss])

    def E_ext(self) -> torch.Tensor:
        u, v = self.pred_uv(self.xy_up)
        u, v = self.mm_to_m(u, v)
        return trapz1D((v+u) * fy, self.x_up)
        # return trapz1D(v * fy, self.x_up)


fy = 5.0
E_outer = 1.0e3;
E_inner = 100.0e3
# E_outer = 100.0e3; E_inner = 0.0
nu1 = 0.3;
nu2 = 0.3


net_whole = AxisScalar2D_withinput(
            stack_net(input=2,output=2,activation=nn.Tanh,width=20,depth=4),
            A=torch.tensor([2,2]),
            B=torch.tensor([-1,-1])
            )


plate_whole = Plate_whole(model=net_whole)


class input_extend(nn.Module):
    def __init__(self, net: nn.Module, X0:torch.Tensor,Y0:torch.Tensor) -> None:
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = net
        self.X0 = X0.to(self.device)
        self.Y0 = Y0.to(self.device)


    def forward(self, xy):

        k = self.X0.size(0)
        n = xy.size(0)

        xy = xy.repeat(k,1)
        '''注意形状 可能返回的是行或者列'''
        # 方法一：设置重复矩阵
        # block_diag_matrix = torch.eye(n).repeat(k, 1)
        # xy = torch.matmul(block_diag_matrix, xy)

        # 方法二：使用 Kronecker 乘积进行行重复
        # repeat_matrix = torch.ones(k, 1)
        # xy = torch.kron(repeat_matrix, xy)

        # 将 x0 和 y0 的每个元素重复 n 次，并与 a 组合成 (k*n, 4)
        x0_repeated = self.X0.repeat_interleave(n)  # 每个元素重复 n 次
        y0_repeated = self.Y0.repeat_interleave(n)  # 每个元素重复 n 次

        # 将 x0_repeated 和 y0_repeated 作为后两列，组合到 a 中
        result = torch.cat([xy, x0_repeated.unsqueeze(1), y0_repeated.unsqueeze(1)], dim=1).to(device)
        return self.net(result), result


class muti_AxisScalar(nn.Module):

    def __init__(self, net: nn.Module, local_area:torch.Tensor) -> None:
        '''X_out=A*X+B'''
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = net
        self.local_area = local_area.to(self.device)

    def forward(self, xy):

        A = torch.ones_like(xy)
        B = torch.zeros_like(xy)
        scalar = 1/self.local_area
        A[:,0] = scalar
        A[:, 1] = scalar
        translate_x = -xy[:,2] /self.local_area
        translate_y = -xy[:,3] / self.local_area
        B[:, 0] = translate_x
        B[:, 1] = translate_y
        xy_normed = A * xy + B
        # print(torch.max(xy_normed,dim=0))
        # print(torch.min(xy_normed,dim=0))
        # print(xy_normed[:,2:3])
        # print(xy_normed)
        return self.net(xy_normed)


class muti_weighted_DEM(Elasticity2D_muti.DEM2D_2):
    def __init__(self, model: nn.Module,local_area: torch.Tensor
                #  x_span:list,y_span:list
                 ):
        super().__init__(model)
        self.local_area = local_area

    def weight(self,xy):
        xy = xy.cpu()

        x0 = xy[:,2]
        y0 = xy[:,3]
        return soft_multidomain.cubic_BSpline_kernel(x0, y0, self.local_area)

    def pred_uv(self, xy):
        u,v,result = super().pred_uv(xy)
        # u, v, xy = pred_uv(xy)
        xy = xy.cpu().float()
        result = result.cpu()
        n = int(xy.size(0))
        k = int(result.size(0) / n)
        weight_instance = self.weight(result)
        weight = weight_instance.getWeight(result).to(device)

        u_m = weight * u
        v_m = weight * v

        #方法一：直接shape变形后sum计算
        '''需要验证'''
        u_m_reshaped = u_m.view(k, n)
        u = u_m_reshaped.sum(dim=0, keepdim=True).t().reshape(-1)

        v_m_reshaped = v_m.view(k, n)
        v = v_m_reshaped.sum(dim=0, keepdim=True).t().reshape(-1)
        #方法二：矩阵乘法
        # block_diag_matrix = torch.eye(n).repeat(1, k)
        # xy = torch.matmul(block_diag_matrix, xy)
        return u, v


local_area = 0.15
material_r = 0.1

scalar = 1/local_area


x0 = torch.tensor([0.25,0.25,0.75,0.75])
y0 = torch.tensor([0.25,0.75,0.25,0.75])
material_surface0 = Geometry.Circle(0, 0, material_r*scalar)
material_surface1 = Geometry.Circle(0.25, 0.25, material_r)
material_surface2 = Geometry.Circle(0.25, 0.75, material_r)
material_surface3 = Geometry.Circle(0.75, 0.25, material_r)
material_surface4 = Geometry.Circle(0.75, 0.75, material_r)

net_center1 = input_extend(
    net = muti_AxisScalar(
        net = Enrichment.extendAxisNet(
            net = stack_net(input=5,output=2,activation=nn.Tanh,width=25,depth=6),
            extendAxis= Enrichment.BimaterialLSBasis(material_surface0)),
        local_area = torch.tensor([local_area])),
    X0 = x0,
    Y0 = y0
)

plate_center1 = muti_weighted_DEM(model = net_center1 ,local_area = torch.tensor([local_area]))


pinn = Plate([plate_whole,plate_center1])


pinn.add_BCPoints()
pinn.setMaterial(E1=E_inner, E2=E_outer,nu1=nu1,nu2=nu2,type='plane stress')

pinn.set_LevelSet([material_surface1,material_surface2,material_surface3,material_surface4])
# pinn.set_meshgrid_inner_points(0,1,200,0,1,200)
pinn.set_meshgrid_inner_points(1e-4,0.999,250,2e-4,0.9999,250)
# pinn.set_meshgrid_trapz_Tip_Dense(1e-4, 1 - 1e-4, 2e-4, 1 - 1e-5,
#                                         0.5, 0.5,
#                                         250, 250, 150, 150,
#                                         x_inteval=0.3, y_inteval=0.3)

pinn.set_loss_func(losses=[pinn.Energy_loss,
                           #    pinn.Equilibrium_loss,
                           ],
                   weights=[1000.0,
                            #    1e-6,1e-6
                            ]
                   )
pinn.set_Optimizer(0.008)
pinn.train(path=model_name, patience=30, epochs=80000, eval_sep=100)
# pinn.load(path='C:/Users/15844/PycharmProjects/pythonProject4/DEM/3circle-allscalar')
# model_name = 'test-for-mutiinput-fourcircle'

# pinn.load(path='C:/Users/15844/PycharmProjects/pythonProject5/DEM_formuti/test-for-mutiinput-fourcircle')
# pinn.readData("C:/Users/15844/Desktop/DEDEM/circle0.1/fourcircle0.1.csv")
# pinn.evaluate(name=None)
# pinn.plot_result_NN()
# pinn.plot_result_FEM()
# pinn.plot_abs_error()

























# plt.show()
#
# x, y = NodesGenerater.genMeshNodes2D(0, 1, 201, 0, 1, 201)
# pinn.set_label_xy(x, y)
