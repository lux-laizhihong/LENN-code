import Elasticity2D
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


class Plate_whole(Elasticity2D.DEM_bimaterial):

    def hard_u(self, u, x, y):
        return u * y
        # return u * x * (1 - x)

    def hard_v(self, v, x, y):
        return v * y


class Plate(soft_multidomain.soft_multidomain_Bi):

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
        return trapz1D(v * fy, self.x_up)
        # return trapz1D(v * fy, self.x_up)


fy = 5.0
E_outer = 50.0e3;
E_inner = 1.0e3
# E_outer = 100.0e3; E_inner = 0.0
nu1 = 0.3;
nu2 = 0.3

net_whole = AxisScalar2D(
            stack_net(input=2,output=2,activation=nn.Tanh,width=25,depth=4),
            A=torch.tensor([2.,2]),
            B=torch.tensor([-1,-1])
            )
plate_whole = Plate_whole(model=net_whole)


local_area = 0.3
material_r = 0.15;center_x = 0.5;center_y = 0.5
scalar = 1/local_area;translate = -center_x/local_area
theta = np.pi/1.5

material_surface0 = Geometry.Circle(0, 0, material_r*scalar)
material_surface1 = Geometry.Circle(center_x, center_y, material_r)


crack_x = center_x;crack_y = center_y;local_area_crack = local_area
# crack_x = center_x;crack_y = center_y+material_r*np.cos(theta/2);local_area_crack = material_r*1.5*np.sin(theta/2)
# crack_extension_0 = RELUPSI.RELU2PSIcircle_crack(x0=0.,
#                                        y0=0.,
#                                        r = material_r*scalar,
#                                        tip='both',
#                                        theta = theta,local_area=1)

crack_extension_1 = RELUPSI.RELU2PSIcircle_crack(x0=center_x,
                                       y0=center_y,
                                       r = material_r,
                                       tip='both',
                                       theta = theta,local_area=local_area_crack)

kernel_crack = soft_multidomain.cubic_BSpline_kernel(crack_x, crack_y, local_area_crack)
scalar_crack = 1/local_area_crack;translate_crack_x = -crack_x/local_area_crack;translate_crack_y = -crack_y/local_area_crack;

net_center = Enrichment.extendAxisNet_muti(
        net = AxisScalar2D(
            stack_net(input=4,output=2,activation=nn.Tanh,width=25,depth=6),
            A=torch.tensor([scalar_crack,scalar_crack,1.0,1.0]),
            B=torch.tensor([translate_crack_x,translate_crack_y,0.0,0.0])
            ),
        extendAxis1= crack_extension_1,
        extendAxis2=Enrichment.BimaterialLSBasis(material_surface1))


plate_center = soft_multidomain.weighted_DEM(model=net_center,
                                              weight=kernel_crack)


pinn = Plate([plate_whole,plate_center])

pinn.add_BCPoints()
pinn.setMaterial(E1=E_inner, E2=E_outer, nu1=nu1, nu2=nu2, type='plane stress')

pinn.set_LevelSet(material_surface1)
pinn.set_meshgrid_inner_points(1e-4,0.999,300,2e-4,0.9999,300)


pinn.set_loss_func(losses=[pinn.Energy_loss,
                           #    pinn.Equilibrium_loss,
                           ],
                   weights=[10000.0,
                            #    1e-6,1e-6
                            ]
                   )
pinn.set_Optimizer(0.008)

# data_path = "C:/Users/15844/Desktop/Example\model-comsol\crack-circle-0.15-50-1-small.csv"
# pinn.readData(data_path)
model_name = 'crack-circle_onelocal'
pinn.train(path=model_name, patience=300, epochs=100000, eval_sep=100)



# pinn.load(path='C:/Users/15844/PycharmProjects/pythonProject4/DEM'
#                f'/crack-circle_onelocal-50')
#
# pinn.readData(data_path)
# pinn.evaluate(name = None)
# von_mises_rMSE,sx_rMSE,sy_rMSE,sxy_rMSE = pinn.rMSE_stress(401,401)
# print(von_mises_rMSE,sx_rMSE,sy_rMSE,sxy_rMSE)
# displacement_rmse,u_rmse,v_rmse = pinn.rMSE_displacement(401,401)
# print(displacement_rmse,u_rmse,v_rmse)
#
# pinn.l2_error_vonmises()
# pinn.R2_squre_vonmises()
# pinn.plot_result_NN()
# pinn.plot_result_FEM()


