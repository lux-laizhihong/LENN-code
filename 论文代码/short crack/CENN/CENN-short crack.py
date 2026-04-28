from Elasticity2D import DEM2D_2
from multidomain import BiDomain
import torch
import torch.nn as nn
import stats
from NodesGenerater import AcceptanceSampling2D, plot_points, genMeshNodes2D
from NN import stack_net, MultilayerNN
from Integral import trapz1D, montecarlo
from activation import ReLU2, ELU2, ReLU3
from get_grad import get_grad
import numpy as np
from shapely.geometry import Point, Polygon
from multidomain import interface, multidomain,muti_interface
from Geometry import LineSegement

import matplotlib.pyplot as plt
import NodesGenerater
import Geometry


class Plate_domain1(DEM2D_2):
    def __init__(self, model: nn.Module, fy):
        super().__init__(model)
        self.fy = fy
        self.add_BCPoints()

    def hard_u(self, u, x, y):
        return u * x

    def hard_v(self, v, x, y):
        return v*(y+1)/2

    def add_BCPoints(self, num=[500]):
        x_up, y_up = genMeshNodes2D(-1-1e-4, 0.99999, num[0], 1, 1, 1)
        self.x_up, self.y_up, self.xy_up = self._set_points(x_up, y_up)
        self.up_zero = torch.zeros_like(self.x_up)

    def E_ext(self) -> torch.Tensor:
        u, v = self.pred_uv(self.xy_up)
        u, v = self.mm_to_m(u, v)
        return trapz1D(v * self.fy, self.x_up)


class Plate_domain2(DEM2D_2):
    def __init__(self, model: nn.Module, fy):
        super().__init__(model)
        self.fy = fy
        self.add_BCPoints()

    def hard_u(self, u, x, y):
        return u * x

    def hard_v(self, v, x, y):
        '''约束点(1,-1)'''

        return v * (y+1)/2

    def add_BCPoints(self, num=[500]):
        x_down, y_down = genMeshNodes2D(0, 1, num[0], -1, -1, 1)
        self.x_down, self.y_down, self.xy_down = self._set_points(x_down, y_down)
        self.down_zero = torch.zeros_like(self.x_down)

    def E_ext(self) -> torch.Tensor:
        u_down, v_down = self.pred_uv(self.xy_down)
        u_down, v_down = self.mm_to_m(u_down, v_down)

        return trapz1D(v_down * (-self.fy), self.x_down)


width = 25
depth = 6
half_length = 0.1
# 设置上半区域网络
net = stack_net(input=2, output=2, width=width, activation=nn.Tanh, depth=depth)
# total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# print(f"\n总可训练参数数量: {total_params}")

pinn_domain_upper = Plate_domain1(net, fy=10.0)
# pinn_domain_upper.add_BCPoints()
pinn_domain_upper.set_geometry(Polygon([(-1.0, 0.0),
                                        (1.0, 0.0),
                                        (1.0, 1.0),
                                        (-1.0, 1.0)]))

pinn_domain_upper.setMaterial(E=100.0e3, nu=0.3)
# pinn_domain1.set_Equilibrium_points()

# 设置下半区域网络
net = stack_net(input=2, output=2, width=width, activation=nn.Tanh, depth=depth)
pinn_domain_lower = Plate_domain2(net, fy=10.0)
pinn_domain_lower.set_geometry(Polygon([(-1.0, -1.0),
                                        (1.0, -1.0),
                                        (1.0, 0.0),
                                        (-1.0, 0.0)]))
pinn_domain_lower.setMaterial(E=100.0e3, nu=0.3)

# pinn_domain2.set_Equilibrium_points()
interface_tip = muti_interface(pinn_domain_upper, pinn_domain_lower,
                          geometry1=LineSegement([half_length, 0.0], [1.0, 0.0]),geometry2=LineSegement([-1.0, 0.0], [-half_length, 0.0]))

CENN = multidomain([pinn_domain_upper, pinn_domain_lower])

# params = [{'params': x.parameters()} for x in pinn_bimaterial.models]
# print(params)

# 设置点
points_num = 90000
# points_distribution = stats.Stats2D(stats.Uniform(-1,1),stats.Uniform(-1,1))
# points = AcceptanceSampling2D(-1,1,-1,1,points_num,points_distribution)
# points_pdf = points_distribution.pdf(points[:,0],points[:,1])
points, points_pdf = NodesGenerater.genDenseCircles(-1, 1, -1, 1, 85000,
                                                    [Geometry.Circle(half_length, 0.0, 0.15),Geometry.Circle(-half_length, 0.0, 0.15)],
                                                    5000)
CENN.set_inner_points(points, points_pdf)

interface_num = 2000
interface_tip.generate_points(interface_num)

# interface_weight = -100 * np.log(np.tanh(interface_num/points_num))
interface_weight = -1000 * np.log(np.tanh(interface_num / points_num))

# CENN.set_scalars(E=100.0,delta=10.0)
# print(pinn_bimaterial.E_ext())
# print(pinn_bimaterial.E_int())

CENN.set_loss_func(losses=[CENN.Energy_loss,
                           interface_tip.u_loss,
                           ],
                   weights=[1000.0]
                           + [interface_weight] * 2 * (CENN.domain_num - 1)
                   )

model_name = 'CENN-short crack'
CENN.train(path='' + model_name, patience=1000, epochs=35000)


# CENN.load(path='C:/Users/15844/PycharmProjects/pythonProject4/DEM/' +model_name)
# CENN.load(model_name)
# print(model_name)

# pinn.showResult(pinn.XY)

# pinn.readData('plate/homo_mode1/edge_homo_fixlow_5050.txt')
# CENN.readData('C:/Users/15844/Desktop/crack-s/one-bottom-2-0.1/2-0.1-0-local.csv')
# pinn.load(path='plate/'+model_name)


# CENN.plot_result_NN()
# CENN.plot_result_FEM()
# CENN.evaluate(name=None, levels=100)


# print(CENN.Energy_loss().cpu().detach().numpy())