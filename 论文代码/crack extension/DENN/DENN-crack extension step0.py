import torch
import torch.nn as nn

import Enrichment
import NN
import NodesGenerater
import RELUPSI
import soft_multidomain
from Elasticity2D import DEM2D_2
from Integral import trapz1D
from NN import stack_net, AxisScalar2D


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

length = 0.5

E = 200.0e3
nu = 0.3
fy = 0.0
width = 32
depth = 6

x_crackTip1 = 1-length
y_crackTip1 = -0.05
x_crackCenter1 = 1
y_crackCenter1 = -0.05

crack_extension_1 = RELUPSI.RELU2PSILine(xy0=[x_crackCenter1, y_crackCenter1],
                                         xy1=[x_crackTip1, y_crackTip1],
                                         tip='right')


x_crackTip2 = -1+length
y_crackTip2 = 0.05
x_crackCenter2 = -1
y_crackCenter2 = 0.05
crack_extension_2 = RELUPSI.RELU2PSILine(xy0=[x_crackCenter2, y_crackCenter2],
                                         xy1=[x_crackTip2, y_crackTip2],
                                         tip='right')

crack_net = Enrichment.extendAxisNet_muti(
            stack_net(input=4,output=2,activation=nn.Tanh,width=32,depth=6),
             extendAxis1= crack_extension_1,
             extendAxis2=crack_extension_2,)

pinn = Plate_whole(crack_net)
pinn.add_BCPoints()
pinn.setMaterial(E=E, nu=nu)
pinn.set_meshgrid_inner_points(-1+0.0001, 0.99999, 350, 0.001-1, 0.9999, 350)
pinn.set_loss_func(losses=[pinn.Energy_loss,
                           #    pinn.Equilibrium_loss,
                           ],
                   weights=[10000.0,
                            #    1e-6,1e-6
                            ]
                   )
pinn.set_Optimizer(0.01)
model_name = 'crack_extension-two-edgecrack-DENN-step0'
a=2
pinn.train(path=model_name, patience=300, epochs=25000, eval_sep=100)

# pinn.load(path='C:/Users/15844/PycharmProjects/pythonProject4/DEM/crack_extension-two-edgecrack-DENN-step0')
# data_path = "C:/Users/15844/Desktop/crack_extension/crack_interval0.1m/edge-twocrack-0.5-uhard-step0.csv"
# pinn.readData(data_path)
# pinn.evaluate(name = None)
# pinn.plot_result_NN()




"J积分计算应力强度因子"
from SIF_1 import IIM_circle2D, max_stress_theta

def get_kappa(v):
    return (3 - v) / (1 + v)

def get_mu(E, v):
    return (E / (2 * (1 + v)))



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kappa = get_kappa(nu)
mu = get_mu(E, nu)

local_axis_right = Enrichment.LocalAxis(x0=x_crackTip1, y0=y_crackTip1, beta=np.pi)
local_axis_left = Enrichment.LocalAxis(x0=x_crackTip2, y0=y_crackTip2, beta=0)

num = [3000]
r_list = np.arange(0.13, 0.25, 0.01)

K1_left_list, K2_left_list = [], []
K1_right_list, K2_right_list = [], []
theta_left_list, theta_right_list = [], []

# === 计算循环 ===
for r0 in r_list:
    IIM_pinn_left = IIM_circle2D(r0, local_axis_left, num, int_type='contour')
    IIM_pinn_right = IIM_circle2D(r0, local_axis_right, num, int_type='contour')

    K1_left, K2_left = IIM_pinn_left.compute_K_IIM(pinn, kappa, mu, E)
    K1_right, K2_right = IIM_pinn_right.compute_K_IIM(pinn, kappa, mu, E)

    K1_left = K1_left.cpu().detach().numpy().item()/np.sqrt(1000)
    K2_left = K2_left.cpu().detach().numpy().item()/np.sqrt(1000)
    K1_right = K1_right.cpu().detach().numpy().item()/np.sqrt(1000)
    K2_right = K2_right.cpu().detach().numpy().item()/np.sqrt(1000)
    # print((K1_right**2+K2_right**2)/E*1e6,(K1_left**2+K2_left**2)/E*1e6)

    K1_left_list.append(K1_left)
    K2_left_list.append(K2_left)
    K1_right_list.append(K1_right)
    K2_right_list.append(K2_right)

    theta_left_list.append(max_stress_theta(K1_left, K2_left))
    theta_right_list.append(max_stress_theta(K1_right, K2_right))

# === 汇总输出 ===
print("\n================= 最终结果汇总 =================")
print(" r0     |   K1_left    K2_left   |   K1_right   K2_right   |  θ_left(°)   θ_right(°)")
print("-------------------------------------------------------------")
for i, r0 in enumerate(r_list):
    print(f" {r0:6.2f} | {K1_left_list[i]:10.4f} {K2_left_list[i]:10.4f} | "
          f"{K1_right_list[i]:10.4f} {K2_right_list[i]:10.4f} | "
          f"{theta_left_list[i]*180/np.pi:10.2f} {theta_right_list[i]*180/np.pi:10.2f}")

