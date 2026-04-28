import Elasticity2D
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


class Plate_whole(Elasticity2D.DEM_bimaterial_muti_interfaces):

    def hard_u(self, u, x, y):
        return u * y
        # return u * x * (1 - x)

    def hard_v(self, v, x, y):
        return v * y

class Plate(soft_multidomain.soft_multidomain_Bi_muti_interfaces):

    def add_BCPoints(self, num=[256]):
        x_up, y_up = NodesGenerater.genMeshNodes2D(1e-4, 1 - 1e-4, num[0], 1, 1, 1)
        self.x_up, self.y_up, self.xy_up = self._set_points(x_up, y_up)
        self.up_zero = torch.zeros_like(self.x_up)

        x_down, y_down = NodesGenerater.genMeshNodes2D(1e-3, 1 - 1e-3, num[0], 0, 0, 1)

        x_left, y_left = NodesGenerater.genMeshNodes2D(0, 0, 1, 1e-3, 1 - 1e-3, num[0])

        x_right, y_right = NodesGenerater.genMeshNodes2D(1, 1, 1, 1e-3, 1 - 1e-3, num[0])

        x_tau1 = torch.cat((x_left, x_right))
        y_tau1 = torch.cat((y_left, y_right))
        self.x_tau1, self.y_tau1, self.xy_tau1 = self._set_points(x_tau1, y_tau1)

        x_tau_up = x_up
        y_tau_up = y_up
        self.x_tau_up, self.y_tau_up, self.xy_tau_up = self._set_points(x_tau_up, y_tau_up)


    def soft_BC_loss(self):
        _, _, sxy1 = self.pred_stress(self.xy_tau1)
        sxy1_loss = self.criterion(sxy1, torch.zeros_like(sxy1))
        _, _, sxy_up = self.pred_stress(self.xy_tau_up)
        sxy_up_loss = self.criterion(sxy_up-fy, torch.zeros_like(sxy_up))
        _, sy_up, _ = self.pred_stress(self.xy_up)
        sy_up_loss = self.criterion(sy_up - fy, torch.zeros_like(sy_up))
        return torch.stack([sxy1_loss, sxy_up_loss,sy_up_loss])

    def E_ext(self) -> torch.Tensor:
        u, v = self.pred_uv(self.xy_up)
        u, v = self.mm_to_m(u, v)
        return trapz1D((v+u) * fy, self.x_up)


fy = 5
E_outer = 10.0e3
E_inner = 1.0e3
# E_outer = 100.;0e3; E_inner = 0.0
nu1 = 0.3;
nu2 = 0.3

net_whole =  AxisScalar2D(
            stack_net(input=2,output=2,activation=nn.Tanh,width=20,depth=4),
            A=torch.tensor([2,2]),
            B=torch.tensor([-1,-1])
            )

plate_whole = Plate_whole(model=net_whole)


n=250
center_width = 25;center_depth = 6


local_area = 0.15
material_r = 0.025

center_x1 = 0.5;center_y1 = 0.55
center_x2 = 0.5+material_r*np.sqrt(3);center_y2 = 0.475
center_x3 = 0.5-material_r*np.sqrt(3);center_y3 = 0.475
local_center_x = 0.5;local_center_y = 0.5

scalar = 1/local_area
translate_x = -local_center_x/local_area;translate_y = -local_center_y/local_area


material_surface1_local = Geometry.Circle(center_x1*scalar+translate_x, center_y1*scalar+translate_y, material_r*scalar)
material_surface2_local = Geometry.Circle(center_x2*scalar+translate_x, center_y2*scalar+translate_y, material_r*scalar)
material_surface3_local = Geometry.Circle(center_x3*scalar+translate_x, center_y3*scalar+translate_y, material_r*scalar)

material_surface1 = Geometry.Circle(center_x1, center_y1, material_r)
material_surface2 = Geometry.Circle(center_x2, center_y2, material_r)
material_surface3 = Geometry.Circle(center_x3, center_y3, material_r)

multiExtension = Enrichment.multiBasis(BasisList=
                                       [Enrichment.BimaterialLSBasis(material_surface1_local),
                                        Enrichment.BimaterialLSBasis(material_surface2_local),
                                        Enrichment.BimaterialLSBasis(material_surface3_local)])

net_center = AxisScalar2D(
    net=Enrichment.extendAxisNet(stack_net(input=5, output=2, activation=nn.Tanh, width=center_width, depth=center_depth),
        extendAxis=multiExtension),
    A=torch.tensor([scalar, scalar]),
    B=torch.tensor([translate_x, translate_y])
)

kernel = soft_multidomain.cubic_BSpline_kernel(local_center_x, local_center_y, local_area)
plate_center = soft_multidomain.weighted_DEM(model=net_center,
                                               weight=kernel)
pinn = Plate([plate_whole,plate_center])

pinn.add_BCPoints()
pinn.setMaterial(E1=E_inner, E2=E_outer, nu1=nu1, nu2=nu2, type='plane stress')
pinn.set_LevelSet([material_surface1,material_surface2,material_surface3])
pinn.set_meshgrid_inner_points(0+1e-4,1-1e-4,n,0+2e-4,1-1e-4,n)


pinn.set_loss_func(losses=[pinn.Energy_loss,
                           #    pinn.Equilibrium_loss,
                           ],
                   weights=[1000.0,
                            #    1e-6,1e-6
                            ]
                   )
pinn.set_Optimizer(0.008)
model_name = 'muticircle-separate-0.025-10-1'

pinn.train(path=model_name, patience=300, epochs=25000, eval_sep=100)
# pinn.load(path='C:/Users/15844/PycharmProjects/pythonProject4/DEM/muticircle-separate-0.025-10-1')
# pinn.readData("C:/Users/15844/Desktop/Example/model-comsol/muticircle-Bi1-10-0.025-local.csv")
# # pinn.evaluate(name=None)
# pinn.plot_result_FEM()