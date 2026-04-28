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
        return u*y
        # return u*(x ** 2 + (y + 1) ** 2)
        # return u * (y + 1) / 2

    def hard_v(self, v, x, y):
        return v * y

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

material_r = 0.15;center_x = 0.5;center_y = 0.5
# scalar = 1/local_area;translate = -center_x/local_area
theta = np.pi/1.5
material_surface1 = Geometry.Circle(center_x, center_y, material_r)

crack_x = center_x;crack_y = center_y;crack_n_length = 0.5
#这里local_area=0.5是基本上覆盖全域的 用于归一化裂纹信息
crack_extension_1 = RELUPSI.RELU2PSIcircle_crack(x0=center_x,
                                       y0=center_y,
                                       r = material_r,
                                       tip='both',
                                       theta = theta,local_area=crack_n_length)


crack_net = Enrichment.extendAxisNet_muti(
            stack_net(input=4,output=2,activation=nn.Tanh,width=32,depth=6),
        extendAxis1= crack_extension_1,
        extendAxis2= Enrichment.BimaterialLSBasis(material_surface1))

pinn = Plate_whole(crack_net)
pinn.add_BCPoints()
pinn.setMaterial_Bi(E1=E_inner, E2=E_outer, nu1=nu1, nu2=nu2, type='plane stress')


pinn.set_LevelSet(material_surface1)
pinn.set_meshgrid_inner_points(1e-4,0.999,300,2e-4,0.9999,300)
# pinn.set_meshgrid_trapz_Tip_Dense(1e-4, 1 - 1e-4, 2e-4, 1 - 1e-5,
#                                         0.5, 0.5,
#                                         150, 150, 150, 150,
#                                         x_inteval=0.2, y_inteval=0.2)
pinn.set_loss_func(losses=[pinn.Energy_loss,
                           #    pinn.Equilibrium_loss,
                           ],
                   weights=[10000.0,
                            #    1e-6,1e-6
                            ]
                   )
pinn.set_Optimizer(0.008)
# pinn.load(path='C:/Users/15844/PycharmProjects/pythonProject4/DEM/crack-circle_same-area')

model_name = 'crack-circle_DENN'
pinn.train(path=model_name, patience=300, epochs=100000, eval_sep=100)



# pinn.load(path='')
# data_path =
pinn.readData(data_path)
pinn.evaluate(name = None)

# pinn.l2_error_vonmises()
# pinn.R2_squre_vonmises()
# pinn.plot_result_NN()
# pinn.plot_result_FEM()









