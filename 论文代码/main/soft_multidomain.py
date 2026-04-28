import torch
import Elasticity2D
import Enrichment
from torch.nn import functional as F
import torch.nn as nn
import NN
from EarlyStopping import MultiEarlyStopping
import Geometry
from matplotlib import colors,cm
import numpy as np
import matplotlib.pyplot as plt
from get_grad import get_grad
import pickle
import NodesGenerater


class SoftWeight(Enrichment.EnrichBasis):
    def __init__(self,
                 HeavisideZero=0):
        super().__init__(HeavisideZero)
    
    def getWeight(self, xy) :...



class SoftRelu2Weight1D(SoftWeight):
    def __init__(self, x0 , x1, 
                 inverse = False):
        super().__init__()
        self.x0 = x0
        self.x1 = x1
        if inverse:
           self.weight_func = self.RELU4_inverse
        else: 
            self.weight_func = self.RELU4
    def linear_interp(self, x):
        return (x - self.x0) / (self.x1 - self.x0)
    
    # def cubic_B_spline(self,x):
    #     z = self.linear_interp(x)
    
    def RELU4(self,x):
        z = self.linear_interp(x)
        sign_0 = self.Heaviside(z)
        sign_1 = self.Heaviside(1 - z)
        weight = 0.5*(torch.cos((z)*torch.pi)+1)
        return sign_0 * sign_1 * weight + 1 - sign_0
        # return F.relu(1.0 - F.relu(self.linear_interp(x))**2)**2

    def RELU4_inverse(self,x):
        return 1 - self.RELU4(x)
    
    def getWeight(self, xy) :
        x = xy[...,1]
        return self.weight_func(x)
        # return F.relu(1.0 - F.relu(self.linear_interp(x)))


class cubic_BSpline_kernel(SoftWeight):
    def __init__(self, x0 , y0, a):
        super().__init__()
        self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
    #  heaviside 式子统一
    def getWeight(self, xy):
        z = self.circle.norm_dist_2(xy[...,0],xy[...,1])

        # def norm_dist(self, x, y):
           # return self.dist(x, y) / self.r
        # def norm_dist_2(self, x, y):
           # return self.dist(x, y) / self.r
       # z = self.circle.norm_dist(xy[..., 0], xy[..., 1])
       #  z = torch.tensor(0.51, requires_grad=True)
        f_0 = 2/3 - 4*z**2 + 4*z**3
        f_1 = 4 / 3 - 4 * z + 4 * z ** 2 - 4 / 3 * z ** 3
        w0 = self.Heaviside(0.5-z)*1
        w1 = self.Heaviside(z-0.5) * self.Heaviside(1-z)*1
        w = w0*f_0+w1*f_1
        # print(xy)
        # print(z)
        # print(w0)
        # print(w1)
        # print(get_grad(w0*f_0 + w1*f_1,z))
        # print(get_grad(z,xy))
        # print(get_grad(w0 + w1,z))
        # print(w0  + w1 )
        return (w0*f_0+w1*f_1)

class cubic_BSpline_kernel_inverse(SoftWeight):
    def __init__(self, x0 , y0, a):
        super().__init__()
        self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
    #  heaviside 式子统一
    def getWeight(self, xy):
        z = self.circle.norm_dist_2(xy[...,0],xy[...,1])

        # def norm_dist(self, x, y):
           # return self.dist(x, y) / self.r
        # def norm_dist_2(self, x, y):
           # return self.dist(x, y) / self.r
       # z = self.circle.norm_dist(xy[..., 0], xy[..., 1])
       #  z = torch.tensor(0.51, requires_grad=True)
        f_0 = 2/3 - 4*z**2 + 4*z**3
        f_1 = 4 / 3 - 4 * z + 4 * z ** 2 - 4 / 3 * z ** 3
        w0 = self.Heaviside(0.5-z)*1
        w1 = self.Heaviside(z-0.5) * self.Heaviside(1-z)*1
        w = w0*f_0+w1*f_1
        # print(xy)
        # print(z)
        # print(w0)
        # print(w1)
        # print(get_grad(w0*f_0 + w1*f_1,z))
        # print(get_grad(z,xy))
        # print(get_grad(w0 + w1,z))
        # print(w0  + w1 )
        return 1-(w0*f_0+w1*f_1)*3/2

class sigmoid_kernelv1(SoftWeight):
    def __init__(self, x0 , y0, a,scale = 0.1):
        super().__init__()
        self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
        self.scale = scale
    #  heaviside 式子统一
    def getWeight(self, xy):
        dist = self.circle.dist(xy[...,0],xy[...,1])
        w = torch.sigmoid((1 - dist / self.circle.r) / (self.scale * self.circle.r))
        # print(w)
        # print(get_grad(w,dist))
        return w

class sigmoid_kernelv2(SoftWeight):
    def __init__(self, x0 , y0, a,scale = 0.05):
        super().__init__()
        self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
        self.scale = scale
    #  heaviside 式子统一
    def getWeight(self, xy):
        dist = self.circle.dist(xy[...,0],xy[...,1])
        w = torch.sigmoid((1 - dist / self.circle.r) /self.scale)
        # print(w.size())
        # print(w)
        # print(get_grad(w,dist))
        return w


class sigmoid_kernelv3(SoftWeight):
    def __init__(self, x0 , y0, a,scale = 0.1):
        super().__init__()
        self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
        self.scale = scale
    #  heaviside 式子统一
    def getWeight(self, xy):
        dist = self.circle.dist(xy[...,0],xy[...,1])
        w = torch.sigmoid((self.circle.r - dist) / self.scale)
        # print(w)
        # print(get_grad(w,dist))
        return w






class BSpline2(SoftWeight):
    def __init__(self, x0 , y0, a):
        super().__init__()
        self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
    #  heaviside 式子统一
    def getWeight(self, xy):
        z = self.circle.norm_dist_2(xy[...,0],xy[...,1])

        # def norm_dist(self, x, y):
           # return self.dist(x, y) / self.r
        # def norm_dist_2(self, x, y):
           # return self.dist(x, y) / self.r
       # z = self.circle.norm_dist(xy[..., 0], xy[..., 1])
       #  z = torch.tensor(0.51, requires_grad=True)
        A=-1;B=8
        f_0 =  -8*A/27+ (4*A/3)*(z-1/3)- 2*A*(z-1/3)**2+ B*(z-1/3)**3
        f_1 = A*(z-1)**3
        w0 = self.Heaviside(1/3-z)*1
        w1 = self.Heaviside(z-1/3) * self.Heaviside(1-z)*1
        w = w0*f_0+w1*f_1
        # print(xy)
        # print(z)
        # print(w0)
        # print(w1)
        # print(get_grad(w0*f_0 + w1*f_1,z))
        # print(get_grad(z,xy))
        # print(get_grad(w0 + w1,z))
        # print(w0  + w1 )
        return (w0*f_0+w1*f_1)*3


class cos1(SoftWeight):
    def __init__(self, x0 , y0, a):
        super().__init__()
        self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
    #  heaviside 式子统一
    def getWeight(self, xy):
        z = self.circle.norm_dist_2(xy[...,0],xy[...,1])
        w = self.Heaviside(z) * self.Heaviside(1-z)*torch.cos(torch.pi/2*z)
        return w


class ReLu4(SoftWeight):
    def __init__(self, x0 , y0, a):
        super().__init__()
        self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
    #  heaviside 式子统一
    def getWeight(self, xy):
        z = self.circle.norm_dist_2(xy[...,0],xy[...,1])
        z0=0.9;z1=1.1
        z_n = (z-z0)/(z1-z0)
        sign_0 = self.Heaviside(z_n)
        sign_1 = self.Heaviside(1 - z_n)
        weight = 0.5*(torch.cos((z_n)*torch.pi)+1)
        return sign_0 * sign_1 * weight + 1 - sign_0


class function01(SoftWeight):
    def __init__(self, x0 , y0, a,scale = 0.1):
        super().__init__()
        self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
        self.scale = scale
    #  heaviside 式子统一
    def getWeight(self, xy):
        z = self.circle.norm_dist_2(xy[..., 0], xy[..., 1])
        sign_0 = self.Heaviside(z)
        sign_1 = self.Heaviside(1 - z)

        w = sign_0*sign_1
        # print(w)
        # print(get_grad(w,dist))
        return w


class cubic_BSpline_kernel_gloabl(SoftWeight):
    def __init__(self, x0 , y0, a):
        super().__init__()
        self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
    #  heaviside 式子统一
    def getWeight(self, xy):
        z = self.circle.norm_dist_2(xy[...,0],xy[...,1])

        # def norm_dist(self, x, y):
           # return self.dist(x, y) / self.r
        # def norm_dist_2(self, x, y):
           # return self.dist(x, y) / self.r
       # z = self.circle.norm_dist(xy[..., 0], xy[..., 1])

        f_0 = 2/3 - 4*z**2 + 4*z**3
        w0 = self.Heaviside(0.5-z)*1
        f_1 = 4/3 - 4*z + 4*z**2 - 4/3*z**3
        w1 = self.Heaviside(z-0.5) * self.Heaviside(1-z)*1
        w2 = -0.25*self.Heaviside(z - 1)

        return w2*f_1+w1*f_1





# class funtion1(SoftWeight):
#     def __init__(self, x0 , y0, a):
#         super().__init__()
#         self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
#     #  heaviside 式子统一
#     def getWeight(self, xy):
#         z = self.circle.norm_dist(xy[...,0],xy[...,1])
#         c1 = 0.49692398903405594
#         d1 = 0.7006152014148574
#
#         # Recalculate the coefficients with full precision
#         a0 = 2.0 * c1 + 14.0 * d1 - 8.0
#         b0 = -3.0 * c1 - 19.0 * d1 + 8.0
#         c0 = c1 + 6.0 * d1
#         d0 = 0.0
#
#         a1 = 2.0 * c1 + 6.0 * d1 - 8.0
#         b1 = -3.0 * c1 - 7.0 * d1 + 8.0
#         c1_final = c1
#         d1_final = d1
#         f_0 = a0 * z**3 + b0 * z**2 + c0 * z + d0
#         f_1 = a1 * z**3 + b1 * z**2 + c1_final * z + d1_final
#
#         w0 = self.Heaviside(0.5-z)*1
#         w1 = self.Heaviside(z-0.5) * self.Heaviside(1-z)*1
#         return w0*f_0 + w1*f_1
#
# class funtion2(SoftWeight):
#     def __init__(self, x0 , y0, a):
#         super().__init__()
#         self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
#     #  heaviside 式子统一
#     def getWeight(self, xy):
#         z = self.circle.norm_dist(xy[...,0],xy[...,1])
#         sigma = 0.075
#         a = 0.33
#         f_0 = (1 - z) * 0.3 + 1 * torch.exp(-((z-a)**2) / (2 * sigma**2)) * torch.sin(np.pi * z)
#         f_1 = (1 - z) * 0.3 + 1 * torch.exp(-((z-a)**2) / (2 * sigma**2)) * torch.sin(np.pi * z)
#
#         w0 = self.Heaviside(0.5-z)*1
#         w1 = self.Heaviside(z-0.5) * self.Heaviside(1-z)*1
#         w2 = self.Heaviside(z) * self.Heaviside(1 - z) * 1
#         return w2*f_1
#
# class funtion3(SoftWeight):
#     def __init__(self, x0 , y0, a):
#         super().__init__()
#         self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
#     #  heaviside 式子统一
#     def getWeight(self, xy):
#         z = self.circle.norm_dist(xy[...,0],xy[...,1])
#         f_0 = torch.sin(torch.pi*z)
#         f_1 = torch.sin(torch.pi * z)
#
#
#         w0 = self.Heaviside(0.5-z)*1
#         w1 = self.Heaviside(z-0.5) * self.Heaviside(1-z)*1
#         w2 = self.Heaviside(z-0) * self.Heaviside(1-z)*1
#         return w2*f_1
#
# class funtion4(SoftWeight):
#     def __init__(self, x0 , y0, a):
#         super().__init__()
#         self.circle = Geometry.Circle(x0=x0,y0=y0,r=a)
#     #  heaviside 式子统一
#     def getWeight(self, xy):
#         z = self.circle.norm_dist(xy[...,0],xy[...,1])
#         f_0 = torch.sin(torch.pi*z)
#         f_1 = torch.sin(torch.pi * z)
#
#
#         w0 = self.Heaviside(0.5-z)*1
#         w1 = self.Heaviside(z-0.5) * self.Heaviside(1-z)*1
#         w2 = self.Heaviside(z-0) * self.Heaviside(1-z)*1
#
#         weight = z-z+1
#         return weight*w1+weight*w0




class weighted_DEM(Elasticity2D.DEM2D_2):
    def __init__(self, model: nn.Module, 
                 weight: SoftWeight,
                #  x_span:list,y_span:list
                 ):
        super().__init__(model)
        self.weight = weight
        # self.x_span = x_span
        # self.y_span = y_span
    # def axis_norm(self,xy):
    #     '''放缩到[-1,1],硬边界时也需考虑[-1,1]'''
    #     x = (xy[...,0] - self.x_span[0]) / (self.x_span[1] - self.x_span[0]) * 2 -1
    #     y = (xy[...,1] - self.y_span[0]) / (self.y_span[1] - self.y_span[0]) * 2 -1
    #     return torch.stack((x,y),dim=1)
    def pred_uv(self, xy):
        # uv = self.model(xy)
        # u,v = self.hard_u(uv[0].squeeze(-1),xy[...,0],xy[...,1]) , self.hard_v(uv[1].squeeze(-1),xy[...,0],xy[...,1])
        # xy_norm = self.axis_norm(xy)
        # u,v = super().pred_uv(xy_norm)
        u,v = super().pred_uv(xy)
        # u = u/1000;v = v/1000
        # du_dxy = get_grad(u, xy)
        # dv_dxy = get_grad(v, xy)
        # eXX = du_dxy[..., 0]
        # eYY = dv_dxy[..., 1]
        # eXY = du_dxy[..., 1] + dv_dxy[..., 0]

        # print(f"eYY:{eYY}")
        # print(f"eXY:{eXY}")
        weight = self.weight.getWeight(xy)
        weight_xy = get_grad(weight,xy)
        # weight_x = weight_xy[...,0]
        # weight_y = weight_xy[...,1]
        # print(f"weight:{weight}")
        # print(f"dweight_dx:{weight_x}")
        # print(f"in_u_c:{u}")
        # print(f"eXX:{eXX}")

        # print(weight_x.size())
        # print(xy)
        return weight * u, weight * v

    def pred_grad(self, xy):
        # uv = self.model(xy)
        # u,v = self.hard_u(uv[0].squeeze(-1),xy[...,0],xy[...,1]) , self.hard_v(uv[1].squeeze(-1),xy[...,0],xy[...,1])
        # xy_norm = self.axis_norm(xy)
        # u,v = super().pred_uv(xy_norm)
        u,v = super().pred_uv(xy)
        weight = self.weight.getWeight(xy)
        weight_xy = get_grad(weight,xy)
        weight_x = weight_xy[...,0]
        weight_y = weight_xy[...,1]

        return weight_x, weight_y




class weighted_DEM_global(Elasticity2D.DEM2D_2):
    def __init__(self, model: nn.Module,
                 weight: SoftWeight,
                #  x_span:list,y_span:list
                 ):
        super().__init__(model)
        self.weight = weight
        # self.x_span = x_span
        # self.y_span = y_span
    # def axis_norm(self,xy):
    #     '''放缩到[-1,1],硬边界时也需考虑[-1,1]'''
    #     x = (xy[...,0] - self.x_span[0]) / (self.x_span[1] - self.x_span[0]) * 2 -1
    #     y = (xy[...,1] - self.y_span[0]) / (self.y_span[1] - self.y_span[0]) * 2 -1
    #     return torch.stack((x,y),dim=1)
    def pred_uv(self, xy):
        # uv = self.model(xy)
        # u,v = self.hard_u(uv[0].squeeze(-1),xy[...,0],xy[...,1]) , self.hard_v(uv[1].squeeze(-1),xy[...,0],xy[...,1])
        # xy_norm = self.axis_norm(xy)
        # u,v = super().pred_uv(xy_norm)
        u,v = super().pred_uv(xy)
        x = xy[...,0];y = xy[...,1]
        # u = u * y ;
        # v = v * y;
        v = v*(x+1)
        # u = u * (y + 1) / 2
        # v = v * (1 - y) * (y + 1) * (x + 1) * (1 - x)
        weight = self.weight.getWeight(xy)
        return weight * u, weight * v

    def pred_grad(self, xy):
        # uv = self.model(xy)
        # u,v = self.hard_u(uv[0].squeeze(-1),xy[...,0],xy[...,1]) , self.hard_v(uv[1].squeeze(-1),xy[...,0],xy[...,1])
        # xy_norm = self.axis_norm(xy)
        # u,v = super().pred_uv(xy_norm)
        u,v = super().pred_uv(xy)
        weight = self.weight.getWeight(xy)
        weight_xy = get_grad(weight,xy)
        weight_x = weight_xy[...,0]
        weight_y = weight_xy[...,1]

        return weight_x, weight_y



class extendAxisNet_weight(nn.Module):
    def __init__(self, net: nn.Module,
                 weight: SoftWeight) -> None:
        super().__init__()
        self.net = net
        self.weight = weight

    def forward(self, xy):
        basis = self.weight.getWeight(xy)

        num = xy.shape[0]
        basis = basis.reshape(num,-1)
        axis = torch.cat((xy, basis), dim=1)

        return self.net(axis)

    def infer(self, axis):
        return self.net(axis)

    def set_extend_axis(self, weight:SoftWeight):
        self.weight = weight

class weighted_DEM_no(Elasticity2D.DEM2D_2):
    def __init__(self, model: nn.Module
                #  x_span:list,y_span:list
                 ):
        super().__init__(model)

    def pred_uv(self, xy):
        u,v = super().pred_uv(xy)

        return u, v

class soft_multidomain(Elasticity2D.DEM2D_2):
    def __init__(self,  sub_domains:list[Elasticity2D.DEM2D_2]):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sub_domains = sub_domains
        # self.sub_domain_num = len(sub_domains)
        self.model = nn.ModuleList([domain.model for domain in self.sub_domains])
        self.models_name = ['model_'+str(i+1) for i in range(len(sub_domains))]
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.history = []
        self.params = self.model.parameters()
        self.domain_num = len(self.models_name)
    
    def pred_uv(self, xy):
        uv = [domain.pred_uv(xy) for domain in self.sub_domains]
        u = sum([uv_i[0] for uv_i in uv])
        v = sum([uv_i[1] for uv_i in uv])
        return u,v
    
    def setMaterial(self, E, nu, type='plane stress'):
        super().setMaterial(E, nu, type)
        for i in range(self.domain_num):
            self.sub_domains[i].setMaterial(E, nu, type)

    # def soft_BC_loss(self) -> torch.Tensor:
    #     loss = list(map(lambda x:x.soft_BC_loss() ,self.sub_domains))
    #     return torch.cat(loss)
    def save_loss_history(self, model,filename):
        # 可以把所有损失历史组合成一个字典，方便保存
        loss_dict = {
            'soft_BC_loss_history_xy': model.soft_BC_loss_history_xy,
            'soft_BC_loss_history_yy': model.soft_BC_loss_history_yy,
            'Equilibrium_loss_history_u': model.Equilibrium_loss_history_u,
            'Equilibrium_loss_history_v': model.Equilibrium_loss_history_v
        }

        # 使用 pickle 来保存
        with open(filename, 'wb') as f:
            pickle.dump(loss_dict, f)

        print(f"Loss history saved to {filename}")

    def eval(self):
        loss_array,loss_sum = self.get_loss()
        print(self.iter, loss_sum.cpu().detach().numpy())
        # self.print_loss()
        self.EarlyStopping(loss_sum,self.model)      #判断是否需要提前结束训练
        self.history.append(loss_sum.cpu().detach().numpy())

        # if self.iter % 100 == 0:
        #     # print(self.Equilibrium_loss_history)
        #     self.soft_BC_loss_history_xy.append(self.soft_BC_loss()[0].cpu().detach())
        #     self.soft_BC_loss_history_yy.append(self.soft_BC_loss()[1].cpu().detach())
        #
        #     self.Equilibrium_loss_history_u.append(self.Equilibrium_loss_strongform()[0].cpu().detach())
        #     self.Equilibrium_loss_history_v.append(self.Equilibrium_loss_strongform()[1].cpu().detach())
        #     # print(self.Equilibrium_loss_history_u)
        #
        # if self.iter % 500 == 0:
        #     self.save_loss_history(self.pinn_name, self.loss_history_name)

        if self.iter % 1000 == 0:
            for loss in self.losses:
                print(loss.__name__,':',
                      loss().cpu().detach().numpy())
                
        if self.iter % 10000 == 0:
            self.save(self.path+str(self.iter)) 

    def train(self,epochs = 100000, patience=100,path = 'test',eval_sep=100,numx = 161,numy = 161):
        self.path = path
        # self.iter = 1
        # self.optimizer = torch.optim.Adam(params=self.params, lr= 0.01)
        self.EarlyStopping=MultiEarlyStopping(patience=patience,verbose=True,
                                              paths= [self.path + model + '.pth' for model in self.models_name])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                milestones=[6500,13500,20000,26000,32000], gamma = 0.5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        #         milestones=[10000,20500,30000,40000,50000], gamma = 0.5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10000,16000,19000,23500,28000], gamma = 0.5)
        self.Equilibrium_loss_history_u = []
        self.Equilibrium_loss_history_v = []
        self.soft_BC_loss_history_xy = []
        self.soft_BC_loss_history_yy = []
        self.rMSE_calculate = []

        for i in range(epochs):
            
            self.train_step()
            if hasattr(self, 'labeled_x') and self.labeled_x is not None and self.labeled_x.numel() > 0:

                if self.iter % eval_sep == 0:
                    vm_rMSE, sx_rMSE, sy_rMSE, sxy_rMSE = self.rMSE_stress(numx, numy)
                    self.rMSE_calculate.append([
                        vm_rMSE.item(),
                        sx_rMSE.item(),
                        sy_rMSE.item(),
                        sxy_rMSE.item()
                    ])

                if self.iter % 500 == 0 and self.rMSE_calculate:
                    self.save_rMSE_to_txt(self.rMSE_calculate, self.path, self.iter)
            if self.iter % (eval_sep) == 0:
                self.eval()
            # if self.iter % (eval_sep*5) == 0:
            #     print(self.optimizer)

            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break

        self.save(path+'_final')


    def save(self, name):
        for i,domain in enumerate(self.sub_domains):
            domain.save(name+self.models_name[i])
        self.save_hist(name)

    def load(self, path, loadtype='state'):
        for i,domain in enumerate(self.sub_domains):
            domain.load(path+self.models_name[i],loadtype)
    
    def plot_branch(self, index,name = None):
        x = self.labeled_x.cpu().detach().numpy()
        y = self.labeled_y.cpu().detach().numpy()
        xy= self.labeled_xy

        f = self.infer(xy)[index].cpu().detach().numpy()
        norm = colors.Normalize(vmin=np.min(f), vmax=np.max(f))

        fig,axs = plt.subplots(1,self.domain_num+1,figsize=(10, 4))
        # plot = self.showStress(x,y,f,axs[0],colorbar_norm=norm,cmap='jet',levels=100,cbar=False)
        plot = self.showStress(x,y,f,axs[0],cmap='jet',levels=100,cbar=True,cbar_shrink=0.6)
        # cb = plt.colorbar(mappable=plot,cax = axs[0], ax=axs[0],location = 'bottom')

        for i in range(self.domain_num):
            ax = axs[i+1]
            f_i = self.sub_domains[i].infer(xy)[index].cpu().detach().numpy()
            # plot = self.showStress(x,y,f_i,ax,colorbar_norm=norm,cmap='jet',levels=100,cbar=False)
            plot = self.showStress(x,y,f_i,ax,cmap='jet',levels=100,cbar=True,cbar_shrink=0.6)
            # cb = plt.colorbar(mappable=plot,cax = ax, ax=ax,location = 'bottom')
        

        # cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet',), ax=axs,location = 'bottom')


        # # 设置颜色条标签为科学计数法
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((-2,2))
        # cb.formatter = formatter
        # cb.update_ticks()        

        
        
        if name is None:
            plt.show()
        else:
            plt.savefig(name+'.jpg', dpi=300)


class soft_multidomain_Bi(Elasticity2D.DEM_bimaterial):
    def __init__(self, sub_domains: list[Elasticity2D.DEM_bimaterial]):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sub_domains = sub_domains
        # self.sub_domain_num = len(sub_domains)
        self.model = nn.ModuleList([domain.model for domain in self.sub_domains])
        self.models_name = ['model_' + str(i + 1) for i in range(len(sub_domains))]
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.history = []
        self.params = self.model.parameters()
        self.domain_num = len(self.models_name)

    def mask_d(self,xy):

        x = xy[..., 0];
        y = xy[..., 1]
        d = (x - 0.5) ** 2 + (y - 0.5) ** 2
        rmin = 0.08;
        rmax = 0.12
        mask = torch.sigmoid(100 * (d - rmin ** 2)) * torch.sigmoid(-100 * (d - rmax ** 2))
        max_mask = max(mask)
        return mask / max_mask

    def pred_uv(self, xy):
        x = xy[..., 0];
        y = xy[..., 1];
        uv = [domain.pred_uv(xy) for domain in self.sub_domains]
        u = sum([uv_i[0] for uv_i in uv])
        v = sum([uv_i[1] for uv_i in uv])

        # umax = 0.0762
        # u = 10*umax * torch.tanh(u) * y
        #     # return u * x * (1 - x)
        # vmax = 0.51
        # v = 10*vmax * torch.tanh(v) * y
        # u1 = u*(1-self.mask_d(xy))
        # umax = 2e-3
        # u2 = self.mask_d(xy)*torch.tanh(u)*umax
        # u = (u1+u2)*y
        # v1 = v*(1-self.mask_d(xy))
        # vmax = 0.215
        # v2 = self.mask_d(xy)*torch.tanh(v)*vmax
        # v = (v1+v2)*y
        return u, v

    def setMaterial(self, E1, E2, nu1 = 0.3, nu2 = 0.3, type='plane stress'):
        super().setMaterial_Bi(E1, E2, nu1=0.3, nu2=0.3, type='plane stress')
        for i in range(self.domain_num):
            self.sub_domains[i].setMaterial_Bi(E1, E2, nu1 = 0.3, nu2 = 0.3, type='plane stress')

    # def soft_BC_loss(self) -> torch.Tensor:
    #     loss = list(map(lambda x:x.soft_BC_loss() ,self.sub_domains))
    #     return torch.cat(loss)

    def eval(self):
        loss_array, loss_sum = self.get_loss()
        print(self.iter, loss_sum.cpu().detach().numpy())
        self.EarlyStopping(loss_sum, self.model)  # 判断是否需要提前结束训练
        self.history.append(loss_sum.cpu().detach().numpy())
        if self.iter % 1000 == 0:
            for loss in self.losses:
                print(loss.__name__, ':',
                      loss().cpu().detach().numpy())

        if self.iter % 10000 == 0:
            self.save(self.path + str(self.iter))

    def train(self, epochs=100000, patience=100, path='test', eval_sep=100):
        self.path = path
        # self.optimizer = torch.optim.Adam(params=self.params, lr= 0.01)
        self.EarlyStopping = MultiEarlyStopping(patience=patience, verbose=True,
                                                paths=[self.path + model + '.pth' for model in self.models_name])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5000, 10000, 15000,20000,25000,30000,35000], gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5000, 15000, 25000,35000,45000], gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2500, 5000, 7500,10000,25000,30000], gamma=0.5)
        self.Equilibrium_loss_history_u = []
        self.Equilibrium_loss_history_v = []
        self.soft_BC_loss_history_xy = []
        self.soft_BC_loss_history_yy = []
        self.rMSE_calculate = []
        numx = 201;numy = 201
        for i in range(epochs):

            self.train_step()
            if hasattr(self, 'labeled_x') and self.labeled_x is not None and self.labeled_x.numel() > 0:

                if self.iter % eval_sep == 0:
                    vm_rMSE, sx_rMSE, sy_rMSE, sxy_rMSE = self.rMSE_stress(numx, numy)
                    # self.rMSE_calculate.append([
                    #     vm_rMSE.item(),
                    #     sx_rMSE.item(),
                    #     sy_rMSE.item(),
                    #     sxy_rMSE.item()
                    # ])
                    dis_rMSE, u_rMSE, v_rMSE = self.rMSE_displacement(numx, numy)
                    self.rMSE_calculate.append([
                        dis_rMSE.item(),
                        u_rMSE.item(),
                        v_rMSE.item(),
                        vm_rMSE.item(),
                    ])
                if self.iter % 500 == 0 and self.rMSE_calculate:
                    self.save_rMSE_to_txt(self.rMSE_calculate, self.path, self.iter)
            if self.iter % eval_sep == 0:
                self.eval()

            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break
        self.save(path + '_final')

    def save(self, name):
        for i, domain in enumerate(self.sub_domains):
            domain.save(name + self.models_name[i])
        self.save_hist(name)

    def load(self, path, loadtype='state'):
        for i, domain in enumerate(self.sub_domains):
            domain.load(path + self.models_name[i], loadtype)

    def plot_branch(self, index, name=None):
        x = self.labeled_x.cpu().detach().numpy()
        y = self.labeled_y.cpu().detach().numpy()
        xy = self.labeled_xy

        f = self.infer(xy)[index].cpu().detach().numpy()
        norm = colors.Normalize(vmin=np.min(f), vmax=np.max(f))

        fig, axs = plt.subplots(1, self.domain_num + 1, figsize=(10, 4))
        # plot = self.showStress(x,y,f,axs[0],colorbar_norm=norm,cmap='jet',levels=100,cbar=False)
        plot = self.showStress(x, y, f, axs[0], cmap='jet', levels=100, cbar=True, cbar_shrink=0.6)
        # cb = plt.colorbar(mappable=plot,cax = axs[0], ax=axs[0],location = 'bottom')

        for i in range(self.domain_num):
            ax = axs[i + 1]
            f_i = self.sub_domains[i].infer(xy)[index].cpu().detach().numpy()
            # plot = self.showStress(x,y,f_i,ax,colorbar_norm=norm,cmap='jet',levels=100,cbar=False)
            plot = self.showStress(x, y, f_i, ax, cmap='jet', levels=100, cbar=True, cbar_shrink=0.6)
            # cb = plt.colorbar(mappable=plot,cax = ax, ax=ax,location = 'bottom')

        # cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet',), ax=axs,location = 'bottom')

        # # 设置颜色条标签为科学计数法
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((-2,2))
        # cb.formatter = formatter
        # cb.update_ticks()

        if name is None:
            plt.show()
        else:
            plt.savefig(name + '.jpg', dpi=300)


class soft_multidomain_Bi_muti(Elasticity2D.DEM_bimaterial_muti):
    def __init__(self, sub_domains: list[Elasticity2D.DEM_bimaterial]):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sub_domains = sub_domains
        # self.sub_domain_num = len(sub_domains)
        self.model = nn.ModuleList([domain.model for domain in self.sub_domains])
        self.models_name = ['model_' + str(i + 1) for i in range(len(sub_domains))]
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.history = []
        self.params = self.model.parameters()
        self.domain_num = len(self.models_name)

    def pred_uv(self, xy):
        uv = [domain.pred_uv(xy) for domain in self.sub_domains]
        u = sum([uv_i[0] for uv_i in uv])
        v = sum([uv_i[1] for uv_i in uv])
        return u, v

    def setMaterial(self, E1, E2, nu1 = 0.3, nu2 = 0.3, type='plane stress'):
        super().setMaterial_Bi(E1, E2, nu1=0.3, nu2=0.3, type='plane stress')
        for i in range(self.domain_num):
            self.sub_domains[i].setMaterial_Bi(E1, E2, nu1 = 0.3, nu2 = 0.3, type='plane stress')

    # def soft_BC_loss(self) -> torch.Tensor:
    #     loss = list(map(lambda x:x.soft_BC_loss() ,self.sub_domains))
    #     return torch.cat(loss)

    def eval(self):
        loss_array, loss_sum = self.get_loss()
        print(self.iter, loss_sum.cpu().detach().numpy())
        self.EarlyStopping(loss_sum, self.model)  # 判断是否需要提前结束训练
        self.history.append(loss_sum.cpu().detach().numpy())
        if self.iter % 1000 == 0:
            for loss in self.losses:
                print(loss.__name__, ':',
                      loss().cpu().detach().numpy())

        if self.iter % 10000 == 0:
            self.save(self.path + str(self.iter))

    def train(self, epochs=100000, patience=100, path='test', eval_sep=100):
        self.path = path
        # self.optimizer = torch.optim.Adam(params=self.params, lr= 0.01)
        self.EarlyStopping = MultiEarlyStopping(patience=patience, verbose=True,
                                                paths=[self.path + model + '.pth' for model in self.models_name])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5000, 10000, 15000], gamma=0.5)
        for i in range(epochs):

            self.train_step()
            if self.iter % eval_sep == 0:
                self.eval()
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break

        self.save(path + '_final')

    def save(self, name):
        for i, domain in enumerate(self.sub_domains):
            domain.save(name + self.models_name[i])
        self.save_hist(name)

    def load(self, path, loadtype='state'):
        for i, domain in enumerate(self.sub_domains):
            domain.load(path + self.models_name[i], loadtype)

    def plot_branch(self, index, name=None):
        x = self.labeled_x.cpu().detach().numpy()
        y = self.labeled_y.cpu().detach().numpy()
        xy = self.labeled_xy

        f = self.infer(xy)[index].cpu().detach().numpy()
        norm = colors.Normalize(vmin=np.min(f), vmax=np.max(f))

        fig, axs = plt.subplots(1, self.domain_num + 1, figsize=(10, 4))
        # plot = self.showStress(x,y,f,axs[0],colorbar_norm=norm,cmap='jet',levels=100,cbar=False)
        plot = self.showStress(x, y, f, axs[0], cmap='jet', levels=100, cbar=True, cbar_shrink=0.6)
        # cb = plt.colorbar(mappable=plot,cax = axs[0], ax=axs[0],location = 'bottom')

        for i in range(self.domain_num):
            ax = axs[i + 1]
            f_i = self.sub_domains[i].infer(xy)[index].cpu().detach().numpy()
            # plot = self.showStress(x,y,f_i,ax,colorbar_norm=norm,cmap='jet',levels=100,cbar=False)
            plot = self.showStress(x, y, f_i, ax, cmap='jet', levels=100, cbar=True, cbar_shrink=0.6)
            # cb = plt.colorbar(mappable=plot,cax = ax, ax=ax,location = 'bottom')

        # cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet',), ax=axs,location = 'bottom')

        # # 设置颜色条标签为科学计数法
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((-2,2))
        # cb.formatter = formatter
        # cb.update_ticks()

        if name is None:
            plt.show()
        else:
            plt.savefig(name + '.jpg', dpi=300)

        
class soft_multidomain_Bi_muti_interfaces(Elasticity2D.DEM_bimaterial_muti_interfaces):
    def __init__(self, sub_domains: list[Elasticity2D.DEM_bimaterial]):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sub_domains = sub_domains
        # self.sub_domain_num = len(sub_domains)
        self.model = nn.ModuleList([domain.model for domain in self.sub_domains])
        self.models_name = ['model_' + str(i + 1) for i in range(len(sub_domains))]
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.history = []
        self.params = self.model.parameters()
        self.domain_num = len(self.models_name)

    def pred_uv(self, xy):
        uv = [domain.pred_uv(xy) for domain in self.sub_domains]
        # print(uv.size())
        u = sum([uv_i[0] for uv_i in uv])
        v = sum([uv_i[1] for uv_i in uv])
        # print(u.size())
        # x = xy[:,0];y = xy[:,1]
        # u = u*y
        # v = v*y
        return u, v

    def setMaterial(self, E1, E2, nu1 = 0.3, nu2 = 0.3, type='plane stress'):
        super().setMaterial_Bi(E1, E2, nu1=0.3, nu2=0.3, type='plane stress')
        for i in range(self.domain_num):
            self.sub_domains[i].setMaterial_Bi(E1, E2, nu1 = 0.3, nu2 = 0.3, type='plane stress')

    # def soft_BC_loss(self) -> torch.Tensor:
    #     loss = list(map(lambda x:x.soft_BC_loss() ,self.sub_domains))
    #     return torch.cat(loss)


    def eval(self):
        loss_array, loss_sum = self.get_loss()
        print(self.iter, loss_sum.cpu().detach().numpy())
        # Equilibrium_loss_history ,soft_BC_loss_history= self.set_loss_history_eq()
        # self.print_loss()
        self.EarlyStopping(loss_sum, self.model)  # 判断是否需要提前结束训练
        self.history.append(loss_sum.cpu().detach().numpy())
        # if self.iter % 10 == 0:
        #     self.soft_BC_loss_history_xy1.append\
        #         (self.soft_BC_loss()[0].cpu().detach())
        #     self.soft_BC_loss_history_xy_up.append\
        #         (self.soft_BC_loss()[1].cpu().detach())
        #     self.soft_BC_loss_history_yy.append\
        #         (self.soft_BC_loss()[2].cpu().detach())
        #
        #     self.Equilibrium_loss_history_u.append\
        #         (self.Equilibrium_loss_strongform()[0].cpu().detach())
        #     self.Equilibrium_loss_history_v.append\
        #         (self.Equilibrium_loss_strongform()[1].cpu().detach())
        #     # print(self.Equilibrium_loss_history_u)

        if self.iter % 1000 == 0:
            for loss in self.losses:
                print(loss.__name__, ':',
                      loss().cpu().detach().numpy())

        if self.iter % 10000 == 0:
            self.save(self.path + str(self.iter))



    def train(self, epochs=100000, patience=100, path='test', eval_sep=100,numx = 201,numy = 201):
        self.path = path
        # self.optimizer = torch.optim.Adam(params=self.params, lr= 0.01)
        self.EarlyStopping = MultiEarlyStopping(patience=patience, verbose=True,
                                                paths=[self.path + model + '.pth' for model in self.models_name])
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10000, 30000,50000], gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[6500,15000,25000,30000], gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[6000, 12500, 18500, 24500,30000,50000],
        #                                                  gamma=0.5)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        milestones=[ 6500, 12500,18500, 24500,30000,50000,120000],gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        # milestones=[ 10500 ,20500,30000,50000,120000],gamma=0.5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
        # milestones=[6500, 12500, 22500, 30000,40000],gamma=0.5)
        self.Equilibrium_loss_history_u = []
        self.Equilibrium_loss_history_v = []
        self.soft_BC_loss_history_xy1 = []
        self.soft_BC_loss_history_xy_up = []
        self.soft_BC_loss_history_yy = []
        self.rMSE_calculate = []
        for i in range(epochs):

            self.train_step()
            if hasattr(self, 'labeled_x') and self.labeled_x is not None and self.labeled_x.numel() > 0:

                if self.iter % eval_sep == 0:
                    vm_rMSE, sx_rMSE, sy_rMSE, sxy_rMSE = self.rMSE_stress(numx, numy)
                    self.rMSE_calculate.append([
                        vm_rMSE.item(),
                        sx_rMSE.item(),
                        sy_rMSE.item(),
                        sxy_rMSE.item()
                    ])

                if self.iter % 500 == 0 and self.rMSE_calculate:
                    self.save_rMSE_to_txt(self.rMSE_calculate, self.path, self.iter)
            # 判断 self.x_strong 是否存在且非空，才执行
            if hasattr(self, 'x_strong') and self.x_strong is not None and self.x_strong.numel() > 0:

                if self.iter % eval_sep == 0:
                    self.soft_BC_loss_history_xy1.append(self.soft_BC_loss()[0].cpu().detach())
                    self.soft_BC_loss_history_xy_up.append(self.soft_BC_loss()[1].cpu().detach())
                    self.soft_BC_loss_history_yy.append(self.soft_BC_loss()[2].cpu().detach())

                    self.Equilibrium_loss_history_u.append(self.Equilibrium_loss_strongform()[0].cpu().detach())
                    self.Equilibrium_loss_history_v.append(self.Equilibrium_loss_strongform()[1].cpu().detach())

                if self.iter % 5000 == 0 and self.Equilibrium_loss_history_u:
                    self.save_losses_to_txt(self.path, self.iter)

            if self.iter % eval_sep == 0:
                self.eval()
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break

        self.save(path + '_final')

    def save(self, name):
        for i, domain in enumerate(self.sub_domains):
            domain.save(name + self.models_name[i])
        self.save_hist(name)

    def load(self, path, loadtype='state'):
        for i, domain in enumerate(self.sub_domains):
            domain.load(path + self.models_name[i], loadtype)

    def plot_branch(self, index, name=None):
        x = self.labeled_x.cpu().detach().numpy()
        y = self.labeled_y.cpu().detach().numpy()
        xy = self.labeled_xy

        f = self.infer(xy)[index].cpu().detach().numpy()
        norm = colors.Normalize(vmin=np.min(f), vmax=np.max(f))

        fig, axs = plt.subplots(1, self.domain_num + 1, figsize=(10, 4))
        # plot = self.showStress(x,y,f,axs[0],colorbar_norm=norm,cmap='jet',levels=100,cbar=False)
        plot = self.showStress(x, y, f, axs[0], cmap='jet', levels=100, cbar=True, cbar_shrink=0.6)
        # cb = plt.colorbar(mappable=plot,cax = axs[0], ax=axs[0],location = 'bottom')

        for i in range(self.domain_num):
            ax = axs[i + 1]
            f_i = self.sub_domains[i].infer(xy)[index].cpu().detach().numpy()
            # plot = self.showStress(x,y,f_i,ax,colorbar_norm=norm,cmap='jet',levels=100,cbar=False)
            plot = self.showStress(x, y, f_i, ax, cmap='jet', levels=100, cbar=True, cbar_shrink=0.6)
            # cb = plt.colorbar(mappable=plot,cax = ax, ax=ax,location = 'bottom')

        # cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet',), ax=axs,location = 'bottom')

        # # 设置颜色条标签为科学计数法
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((-2,2))
        # cb.formatter = formatter
        # cb.update_ticks()

        if name is None:
            plt.show()
        else:
            plt.savefig(name + '.jpg', dpi=300)


class soft_multidomain_Bi_varlinear(Elasticity2D.DEM_E_varLinear):
    def __init__(self, sub_domains: list[Elasticity2D.DEM_E_varLinear]):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.sub_domains = sub_domains
        # self.sub_domain_num = len(sub_domains)
        self.model = nn.ModuleList([domain.model for domain in self.sub_domains])
        self.models_name = ['model_' + str(i + 1) for i in range(len(sub_domains))]
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.history = []
        self.params = self.model.parameters()
        self.domain_num = len(self.models_name)

    def pred_uv(self, xy):
        x = xy[..., 0];
        y = xy[..., 1];
        uv = [domain.pred_uv(xy) for domain in self.sub_domains]
        u = sum([uv_i[0] for uv_i in uv])
        v = sum([uv_i[1] for uv_i in uv])

        # umax = 0.0762
        # u = 10*umax * torch.tanh(u) * y
        #     # return u * x * (1 - x)
        # vmax = 0.51
        # v = 10*vmax * torch.tanh(v) * y
        # u1 = u*(1-self.mask_d(xy))
        # umax = 2e-3
        # u2 = self.mask_d(xy)*torch.tanh(u)*umax
        # u = (u1+u2)*y
        # v1 = v*(1-self.mask_d(xy))
        # vmax = 0.215
        # v2 = self.mask_d(xy)*torch.tanh(v)*vmax
        # v = (v1+v2)*y
        return u, v

    def setMaterial(self, E1, E2, nu1 = 0.3, nu2 = 0.3, type='plane stress'):
        super().setMaterial_Bi(E1, E2, nu1=0.3, nu2=0.3, type='plane stress')
        for i in range(self.domain_num):
            self.sub_domains[i].setMaterial_Bi(E1, E2, nu1 = 0.3, nu2 = 0.3, type='plane stress')




    def eval(self):
        loss_array, loss_sum = self.get_loss()
        print(self.iter, loss_sum.cpu().detach().numpy())
        self.EarlyStopping(loss_sum, self.model)  # 判断是否需要提前结束训练
        self.history.append(loss_sum.cpu().detach().numpy())
        if self.iter % 1000 == 0:
            for loss in self.losses:
                print(loss.__name__, ':',
                      loss().cpu().detach().numpy())

        if self.iter % 10000 == 0:
            self.save(self.path + str(self.iter))

    def train(self, epochs=100000, patience=100, path='test', eval_sep=100):
        self.path = path
        # self.optimizer = torch.optim.Adam(params=self.params, lr= 0.01)
        self.EarlyStopping = MultiEarlyStopping(patience=patience, verbose=True,
                                                paths=[self.path + model + '.pth' for model in self.models_name])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                    milestones=[5000, 10000, 15000,20000,25000], gamma=0.5)

        for i in range(epochs):

            self.train_step()
            if self.iter % eval_sep == 0:
                self.eval()
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break

        self.save(path + '_final')

    def save(self, name):
        for i, domain in enumerate(self.sub_domains):
            domain.save(name + self.models_name[i])
        self.save_hist(name)

    def load(self, path, loadtype='state'):
        for i, domain in enumerate(self.sub_domains):
            domain.load(path + self.models_name[i], loadtype)

    def plot_branch(self, index, name=None):
        x = self.labeled_x.cpu().detach().numpy()
        y = self.labeled_y.cpu().detach().numpy()
        xy = self.labeled_xy

        f = self.infer(xy)[index].cpu().detach().numpy()
        norm = colors.Normalize(vmin=np.min(f), vmax=np.max(f))

        fig, axs = plt.subplots(1, self.domain_num + 1, figsize=(10, 4))
        # plot = self.showStress(x,y,f,axs[0],colorbar_norm=norm,cmap='jet',levels=100,cbar=False)
        plot = self.showStress(x, y, f, axs[0], cmap='jet', levels=100, cbar=True, cbar_shrink=0.6)
        # cb = plt.colorbar(mappable=plot,cax = axs[0], ax=axs[0],location = 'bottom')

        for i in range(self.domain_num):
            ax = axs[i + 1]
            f_i = self.sub_domains[i].infer(xy)[index].cpu().detach().numpy()
            # plot = self.showStress(x,y,f_i,ax,colorbar_norm=norm,cmap='jet',levels=100,cbar=False)
            plot = self.showStress(x, y, f_i, ax, cmap='jet', levels=100, cbar=True, cbar_shrink=0.6)
            # cb = plt.colorbar(mappable=plot,cax = ax, ax=ax,location = 'bottom')

        # cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet',), ax=axs,location = 'bottom')

        # # 设置颜色条标签为科学计数法
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((-2,2))
        # cb.formatter = formatter
        # cb.update_ticks()

        if name is None:
            plt.show()
        else:
            plt.savefig(name + '.jpg', dpi=300)