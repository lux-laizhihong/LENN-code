import torch.nn as nn
import torch
from Elasticity2D import PINN2D,DEM_bimaterial
import torch.nn.functional as F
import numpy as np
import pandas as pd
from EarlyStopping import MultiEarlyStopping
from itertools import chain
from NodesGenerater import AcceptanceSampling2D,genMeshNodes2D,genRandomNodes2D
from stats import Stats2D
import stats
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPoint

from get_grad import get_grad
from Geometry import Geometry1D

class LSNet(nn.Module):
    '''位移富集项'''
    def __init__(self,uv_net:nn.Module,
                 geometry:Geometry1D):
        super().__init__()
        self.uv_net = uv_net
        self.levelset = geometry.levelset

    def forward(self,xy):
        x = xy[...,0]; y = xy[...,1]
        ls = self.levelset(x,y)
        ls = torch.where(ls>0,ls,-ls)
        axis = torch.stack((x,y,ls),dim=1)
        return self.uv_net(axis)
    
    def infer(self,axis):
        return self.uv_net(axis)

class LS_interface:
    '''两个域的界面'''
    def __init__(self,domain:DEM_bimaterial) -> None:
        self.device = domain.device
        self.domain = domain
        self.geometry = domain.material_interface
        self.criterion = domain.criterion


    def variable(self,x:torch.Tensor):
        return x.float().requires_grad_().to(device=self.device)
    
    def set_interface_points(self,x:torch.Tensor,y:torch.Tensor):
        one = torch.ones_like(x)
        x = x.repeat(2)
        y = y.repeat(2)
        self.x = self.variable(x)
        self.y = self.variable(y)
        
        
        '''
        注意:
        xy通过x y创建
        ls通过xy创建
        axis通过xy和ls创建
        这样保证了梯度可以正确继承
        '''
        self.xy = torch.stack([self.x,self.y],dim=1)
        # self.xy = xy.repeat(2,1)
        # x = self.xy[...,0]; y = self.xy[...,1]
        # self.ls = self.geometry.levelset(x,y)
        # self.ls[self.num:] = -self.ls[self.num:]       
        self.d11 = torch.cat((self.domain.d11[0] * one,self.domain.d11[1] * one)).to(self.device)
        self.d12 = torch.cat((self.domain.d12[0] * one,self.domain.d12[1] * one)).to(self.device)
        self.G = torch.cat((self.domain.G[0] * one,self.domain.G[1] * one)).to(self.device)

        
        l_x , l_y = self.geometry.get_direction_cosine(self.xy[...,0],self.xy[...,1])
        self.l_x , self.l_y = l_x.to(self.device) , l_y.to(self.device)

    def pred_uv(self):
        x = self.xy[...,0]; y = self.xy[...,1]
        ls = self.geometry.levelset(x,y)
        '''切记！直线左边<0,需要取负！'''
        ls[:self.num] = -ls[:self.num] 
        # uv = self.domain.model(self.xy)
        axis = torch.stack((x,y,ls),dim=1)
        uv = self.domain.model.uv_net(axis)
        # uv = self.domain.model.infer(self.axis)
        u,v = self.domain.hard_u(uv[0].squeeze(-1),self.xy[...,0],self.xy[...,1]) , self.domain.hard_v(uv[1].squeeze(-1),self.xy[...,0],self.xy[...,1])
        return u,v  
    

    def generate_points(self,num):
        self.num = num
        x,y = self.geometry.generate_random_points(num=num)
        self.set_interface_points(x,y)


    def interface_surface_force(self,sx,sy,sxy):
        px = self.l_x * sx + self.l_y * sxy
        py = self.l_x * sxy + self.l_y * sy
        return px , py

    
    def F_loss(self):
        u,v = self.pred_uv()
        exx,eyy,exy = self.domain.compute_Strain(u,v,self.xy)
        sx,sy,sxy = self.domain.constitutive(exx,eyy,exy,
                                             self.d11,self.d12,self.G)
        px,py = self.interface_surface_force(sx,sy,sxy)  

        px_upper = px[:self.num]  
        py_upper = py[:self.num]  

        px_lower = px[self.num:]
        py_lower = py[self.num:]


        px_loss = self.criterion(px_upper,px_lower)
        py_loss = self.criterion(py_upper,py_lower)
        return torch.stack([px_loss,py_loss])   
    

class interface:
    '''两个域的界面'''
    def __init__(self,domain1:PINN2D,domain2:PINN2D,geometry:Geometry1D) -> None:
        self.device = domain1.device
        self.domain1 = domain1
        self.domain2 = domain2
        self.geometry = geometry
        self.criterion = domain1.criterion
        self.stress_scalar = 0.01

    def variable(self,x:torch.Tensor):
        return x.float().requires_grad_().to(device=self.device)
    
    # def set_interface_points(self,x:torch.Tensor,y:torch.Tensor):
    #     '''xy之间的创建需要没有联系'''
    #     self.x = self.variable(x)
    #     self.y = self.variable(y)
    #     self.xy = torch.stack([self.x,self.y],dim=1)
    #
    #     if (any(self.domain1.is_in_domain(self.xy)) == False) or (any(self.domain2.is_in_domain(self.xy)) == False):
    #         raise Exception('please check domain!')
    #     l_x , l_y = self.geometry.get_direction_cosine(self.xy[...,0],self.xy[...,1])
    #     self.l_x , self.l_y = l_x.to(self.device) , l_y.to(self.device)

    def set_interface_points(self, x: torch.Tensor, y: torch.Tensor):
        '''xy之间的创建需要没有联系'''
        self.x = self.variable(x)
        self.y = self.variable(y)
        self.xy = torch.stack([self.x, self.y], dim=1)
        print(self.xy.size())
        if (any(self.domain1.is_in_domain(self.xy)) == False) or (any(self.domain2.is_in_domain(self.xy)) == False):
            raise Exception('please check domain!')
        l_x, l_y = self.geometry.get_direction_cosine(self.xy[..., 0], self.xy[..., 1])
        self.l_x, self.l_y = l_x.to(self.device), l_y.to(self.device)
    # def generate_points(self,num):
    #     x,y = self.geometry.generate_random_points(num=num)
    #     self.set_interface_points(x,y)
    def generate_points(self,num):
        x,y = self.geometry.generate_random_points(num=num)
        self.set_interface_points(x,y)

    def add_points(self,x,y):
        interface_x , interface_y = self.variable(torch.tensor(x)),self.variable(torch.tensor(y))
        x = torch.cat((self.x,interface_x))
        y = torch.cat((self.y,interface_y))
        self.set_interface_points(x,y)




    def interface_surface_force(self,sx,sy,sxy):
        # print(sx.size)
        # print(self.l_x.size())
        px = self.l_x * sx + self.l_y * sxy
        py = self.l_x * sxy + self.l_y * sy
        return px , py
    # def interface_surface_force(self,xy,sx,sy,sxy):
    #     l_x , l_y = self.geometry.get_direction_cosine(xy[...,0],xy[...,1])
    #     px = l_x * sx + l_y * sxy
    #     py = l_x * sxy + l_y * sy
    #     return px , py
    
    def u_loss(self):
        u_1,v_1 = self.domain1.pred_uv(self.xy)

        u_2,v_2 = self.domain2.pred_uv(self.xy)

        u_loss =  (self.criterion(u_1,u_2))
        v_loss =  (self.criterion(v_1,v_2))

        return torch.stack([u_loss,v_loss])
    
    def F_loss(self):

        sx_1,sy_1,sxy_1 = self.domain1.pred_stress(self.xy)
        px_1,py_1 = self.interface_surface_force(sx_1,sy_1,sxy_1)

        sx_2,sy_2,sxy_2 = self.domain2.pred_stress(self.xy)
        px_2,py_2 = self.interface_surface_force(sx_2,sy_2,sxy_2)

        px_loss = self.criterion(px_1,px_2) * self.stress_scalar
        py_loss = self.criterion(py_1,py_2) * self.stress_scalar
        return torch.stack([px_loss,py_loss])    

    def loss(self):
        u_1,v_1,sx_1,sy_1,sxy_1 = self.domain1.infer(self.xy)
        px_1,py_1 = self.interface_surface_force(sx_1,sy_1,sxy_1)

        u_2,v_2,sx_2,sy_2,sxy_2 = self.domain2.infer(self.xy)
        px_2,py_2 = self.interface_surface_force(sx_2,sy_2,sxy_2)

        u_loss =  self.criterion(u_1,u_2)
        v_loss =  self.criterion(v_1,v_2)
        px_loss = self.criterion(px_1,px_2) * self.stress_scalar
        py_loss = self.criterion(py_1,py_2) * self.stress_scalar
        return torch.stack([u_loss,v_loss,px_loss,py_loss])


class muti_interface:
    '''两个域的界面'''

    def __init__(self, domain1: PINN2D, domain2: PINN2D, geometry1: Geometry1D,geometry2: Geometry1D) -> None:
        self.device = domain1.device
        self.domain1 = domain1
        self.domain2 = domain2
        self.geometry1 = geometry1
        self.geometry2 = geometry2
        self.criterion = domain1.criterion
        self.stress_scalar = 0.01

    def variable(self, x: torch.Tensor):
        return x.float().requires_grad_().to(device=self.device)

    # def set_interface_points(self,x:torch.Tensor,y:torch.Tensor):
    #     '''xy之间的创建需要没有联系'''
    #     self.x = self.variable(x)
    #     self.y = self.variable(y)
    #     self.xy = torch.stack([self.x,self.y],dim=1)
    #
    #     if (any(self.domain1.is_in_domain(self.xy)) == False) or (any(self.domain2.is_in_domain(self.xy)) == False):
    #         raise Exception('please check domain!')
    #     l_x , l_y = self.geometry.get_direction_cosine(self.xy[...,0],self.xy[...,1])
    #     self.l_x , self.l_y = l_x.to(self.device) , l_y.to(self.device)

    def set_interface_points(self, x: torch.Tensor, y: torch.Tensor):
        '''xy之间的创建需要没有联系'''
        self.x = self.variable(x)
        self.y = self.variable(y)
        self.xy = torch.stack([self.x, self.y], dim=1)
        # print(self.xy.size())
        # if (any(self.domain1.is_in_domain(self.xy)) == False) or (any(self.domain2.is_in_domain(self.xy)) == False):
        #     raise Exception('please check domain!')
        # l_x, l_y = self.geometry.get_direction_cosine(self.xy[..., 0], self.xy[..., 1])
        # self.l_x, self.l_y = l_x.to(self.device), l_y.to(self.device)

    # def generate_points(self,num):
    #     x,y = self.geometry.generate_random_points(num=num)
    #     self.set_interface_points(x,y)
    def generate_points(self, num):
        x1, y1 = self.geometry1.generate_random_points(num=num)
        x2, y2 = self.geometry2.generate_random_points(num=num)
        x = torch.cat([x1, x2],dim=0)
        y = torch.cat([y1, y2], dim=0)
        # print(x.size())
        self.set_interface_points(x, y)

    def add_points(self, x, y):
        interface_x, interface_y = self.variable(torch.tensor(x)), self.variable(torch.tensor(y))
        x = torch.cat((self.x, interface_x))
        y = torch.cat((self.y, interface_y))
        self.set_interface_points(x, y)

    def interface_surface_force(self, sx, sy, sxy):
        px = self.l_x * sx + self.l_y * sxy
        py = self.l_x * sxy + self.l_y * sy
        return px, py

    # def interface_surface_force(self,xy,sx,sy,sxy):
    #     l_x , l_y = self.geometry.get_direction_cosine(xy[...,0],xy[...,1])
    #     px = l_x * sx + l_y * sxy
    #     py = l_x * sxy + l_y * sy
    #     return px , py

    def u_loss(self):
        u_1, v_1 = self.domain1.pred_uv(self.xy)

        u_2, v_2 = self.domain2.pred_uv(self.xy)

        u_loss = (self.criterion(u_1, u_2))
        v_loss = (self.criterion(v_1, v_2))

        return torch.stack([u_loss, v_loss])

    def F_loss(self):
        sx_1, sy_1, sxy_1 = self.domain1.pred_stress(self.xy)
        px_1, py_1 = self.interface_surface_force(sx_1, sy_1, sxy_1)

        sx_2, sy_2, sxy_2 = self.domain2.pred_stress(self.xy)
        px_2, py_2 = self.interface_surface_force(sx_2, sy_2, sxy_2)

        px_loss = self.criterion(px_1, px_2) * self.stress_scalar
        py_loss = self.criterion(py_1, py_2) * self.stress_scalar
        return torch.stack([px_loss, py_loss])

    def loss(self):
        u_1, v_1, sx_1, sy_1, sxy_1 = self.domain1.infer(self.xy)
        px_1, py_1 = self.interface_surface_force(sx_1, sy_1, sxy_1)

        u_2, v_2, sx_2, sy_2, sxy_2 = self.domain2.infer(self.xy)
        px_2, py_2 = self.interface_surface_force(sx_2, sy_2, sxy_2)

        u_loss = self.criterion(u_1, u_2)
        v_loss = self.criterion(v_1, v_2)
        px_loss = self.criterion(px_1, px_2) * self.stress_scalar
        py_loss = self.criterion(py_1, py_2) * self.stress_scalar
        return torch.stack([u_loss, v_loss, px_loss, py_loss])





class multidomain(PINN2D):
    def __init__(self, domains:list[PINN2D]):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.domains = domains
        self.models = nn.ModuleList([domain.model for domain in self.domains])
        self.models_name = ['domain_'+str(i+1) for i in range(len(self.domains))]
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.history = []
        self.params = self.models.parameters()
        #[{'params': x.parameters()} for x in self.models] 
        #chain(model.parameters() for model in self.models)
        #self.params = self.models[0].parameters()
        self.area = sum([domain.polygon.area for domain in domains])
        self.domain_num = len(self.domains)

    def set_inner_points(self, internal_points, internal_points_pdf):
        
        for domain in self.domains:
            index = domain.is_in_domain(internal_points)
            weight = self.area / domain.polygon.area
            points = internal_points[index,:]
            pdf = internal_points_pdf[index] * weight
            domain.set_inner_points(points,pdf)

    def E_int(self) -> torch.Tensor:
        #return self.domains[0].E_int()
        #return torch.sum(torch.stack([domain.E_int() for domain in self.domains]))
        return torch.sum(torch.stack(list(map(lambda x: x.E_int(),self.domains))))
    
    def E_ext(self) -> torch.Tensor:
        #return torch.sum(torch.stack([domain.E_ext() for domain in self.domains]))
        return torch.sum(torch.stack(list(map(lambda x: x.E_ext(),self.domains))))

    def Equilibrium_loss(self) -> torch.Tensor:
        return torch.cat(list(map(lambda domain :domain.Equilibrium_loss() ,self.domains)))
    
    def set_interface_loss(self,interfaces:list[interface]):
        self.interfaces = interfaces



    def interface_loss(self) -> torch.Tensor:
        return torch.cat(list(map(lambda x :x.loss() ,self.interfaces)))

    def soft_BC_loss(self) -> torch.Tensor:
        loss = list(map(lambda x:x.soft_BC_loss() ,self.domains))
        return torch.cat(loss)

    def pred_uv(self, xy):
        u,v = torch.zeros_like(xy[:,0]) , torch.zeros_like(xy[:,0])
        for domain in self.domains:
            index = domain.is_in_domain(xy)
            u[index],v[index] = domain.pred_uv(xy[index])
        return u,v

    def pred_stress(self, xy):
        sx,sy,sxy = torch.zeros_like(xy[:,0]) , torch.zeros_like(xy[:,0]) , torch.zeros_like(xy[:,0])
        for domain in self.domains:
            index = domain.is_in_domain(xy)
            sx[index],sy[index],sxy[index] = domain.pred_stress(xy[index])
        return sx,sy,sxy
    
    def infer(self , xy):
        u,v = self.pred_uv(xy)
        sx,sy,sxy = self.pred_stress(xy)
        return u,v,sx,sy,sxy
    def rMSE_stress(self,numx,numy):
        u, v = self.pred_uv(self.labeled_xy)
        sx, sy, sxy = self.pred_stress(self.labeled_xy)
        sx_refer = self.labeled_sx
        sy_refer = self.labeled_sy
        sxy_refer = self.labeled_sxy
        labeled_yx = torch.stack([self.labeled_y,self.labeled_x],dim=1)
        von_mises = torch.sqrt((sx-sy)**2+3*(sxy**2))
        von_mises_refer = torch.sqrt((sx_refer-sy_refer)**2+3*(sxy_refer**2))
        von_mises_rMSE = self.rMSE(von_mises,von_mises_refer,labeled_yx,numx,numy)
        sx_rMSE = self.rMSE(sx, sx_refer, labeled_yx, numx, numy)
        sy_rMSE = self.rMSE(sy, sy_refer, labeled_yx, numx, numy)
        sxy_rMSE = self.rMSE(sxy, sxy_refer, labeled_yx, numx, numy)

        # print(error,denominator)
        return von_mises_rMSE,sx_rMSE,sy_rMSE,sxy_rMSE
        # u_refer = self.labeled_u
        # v_refer = self.labeled_v
        # u = u.cpu().detach().numpy()
        # v = v.cpu().detach().numpy()
        # x = self.labeled_x.cpu().detach().numpy()
        # n = x.size
        # y = self.labeled_y.cpu().detach().numpy()
    def save_rMSE_to_txt(self, rmse_list, filename_base, iter_num):
        import numpy as np
        import os
        rmse_array = np.array(rmse_list)
        base_dir = "C:/Users/15844/PycharmProjects/pythonProject4/DEM/rMSE"
        task_dir = os.path.join(base_dir, filename_base)
        os.makedirs(task_dir, exist_ok=True)

        if rmse_array.ndim == 2 and rmse_array.shape[1] == 4:
            # 如果是4列 → 统一保存到 rMSE/self.path/self.path_iter_rMSE.txt
            filename = os.path.join(task_dir, f"{filename_base}_{iter_num}_rMSE.txt")
            np.savetxt(filename, rmse_array)
        else:
            # 单列模式 → 保存到 rMSE/self.path/sx/ 等子文件夹
            names = ["von_mises", "sx", "sy", "sxy"]
            for i in range(rmse_array.shape[1]):
                sub_dir = os.path.join(task_dir, names[i])
                os.makedirs(sub_dir, exist_ok=True)
                filename = os.path.join(sub_dir, f"{filename_base}_{iter_num}_{names[i]}_rMSE.txt")
                np.savetxt(filename, rmse_array[:, i])
    def save_losses_to_txt(self, filename_base, iter_num):
        import numpy as np
        import os

        base_dir = "C:/Users/15844/PycharmProjects/pythonProject4/DEM/Loss"
        task_dir = os.path.join(base_dir, filename_base)
        os.makedirs(task_dir, exist_ok=True)

        # 保存每一项损失
        loss_dict = {
            "u_loss": self.Equilibrium_loss_history_u,
            "v_loss": self.Equilibrium_loss_history_v,
            "xy1_loss": self.soft_BC_loss_history_xy1,
            "xy_up_loss": self.soft_BC_loss_history_xy_up,
            "yy_loss": self.soft_BC_loss_history_yy,
        }

        for name, loss_list in loss_dict.items():
            filename = os.path.join(task_dir, f"{filename_base}_{iter_num}_{name}.txt")
            np.savetxt(filename, np.array(loss_list))
    def eval(self):
        loss_array,loss_sum = self.get_loss()
        print(self.iter, loss_sum.cpu().detach().numpy())
        self.EarlyStopping(loss_sum,self.models)      #判断是否需要提前结束训练
        self.history.append(self.Energy_loss().cpu().detach().numpy())
        if self.iter % 1000 == 0:
            for loss in self.losses:
                print(loss.__name__,':',
                      loss().cpu().detach().numpy())
            # for it,domain in enumerate(self.domains):
            #     print(self.models_name[it]+':')
            #     domain.print_loss() 
              
        if self.iter % 10000 == 0:
            self.save(self.path+str(self.iter)) 

    def train(self,epochs = 100000, patience=100 , path = 'test', eval_sep=100):
        self.path = path
        self.optimizer = torch.optim.Adam(params=self.params, lr= 0.008)
        self.EarlyStopping=MultiEarlyStopping(patience=patience,verbose=True,
                                              paths= [self.path + model + '.pth' for model in self.models_name])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                milestones=[10000,20000,30000,40000,50000], gamma = 0.5)
        self.Equilibrium_loss_history_u = []
        self.Equilibrium_loss_history_v = []
        self.soft_BC_loss_history_xy1 = []
        self.soft_BC_loss_history_xy_up = []
        self.soft_BC_loss_history_yy = []
        self.rMSE_calculate = []



        for i in range(epochs):
            
            self.train_step()
            numx = 201;numy = 201
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

            scheduler.step()
            if self.iter % eval_sep == 0: 
                self.eval()
            if (self.EarlyStopping.early_stop):
                break

            if self.iter % 50000 == 0:
                self.save(path+str(self.iter))

        self.save(path+'_final')

    def train_resampling(self, points_distribution: Stats2D, num, epochs=100000, patience=100, path='test',
                          left=0, right=1, bottom=0, top=1):
        self.path = path
        self.optimizer = torch.optim.Adam(params=self.params, lr= 0.01)
        self.EarlyStopping=MultiEarlyStopping(patience=patience,verbose=True,
                                              paths= [self.path + model + '.pth' for model in self.models_name])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2000,10000,30000], gamma = 0.2)
        for i in range(epochs):
            if i % 100 == 0:
                points = genRandomNodes2D(left, right, bottom, top ,num)
                points_pdf = points_distribution.pdf(points[:,0],points[:,1])
                self.set_inner_points(points,points_pdf)  
            self.train_step()
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break

        self.save(path+'_final')

    def save(self, name):
        for i,domain in enumerate(self.domains):
            domain.save(name+self.models_name[i])
        self.save_hist(name)

    def load(self, path, loadtype='state'):
        for i,domain in enumerate(self.domains):
            domain.load(path+self.models_name[i],loadtype)
    
    def showDomains(self):
        for domian in self.domains:
            plt.scatter(domian.X.detach().numpy(),domian.Y.detach().numpy(),s=1)
        for inter in self.interfaces:
            plt.scatter(inter.x.detach().numpy(),inter.y.detach().numpy(),s=5,c='black')
        # for domian in self.domains:
        #     plt.scatter(domian.bc_x.detach().numpy(),domian.bc_y.detach().numpy(),s=1)
        plt.axis('equal')
        plt.show()





class BiDomain(PINN2D):
    '''
    将全域划分为两个子域,使用两个模型分别预测
    第一个子域为levelset>0的区域,第二个子域为levelset<0的区域
    '''

    def __init__(self,PINN_domain1:PINN2D,PINN_domain2:PINN2D):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.PINN_domain1 = PINN_domain1
        self.PINN_domain2 = PINN_domain2
        self.models = [self.PINN_domain1.model , self.PINN_domain2.model]
        self.models_name = ['_1','_2']
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.history = []
        self.params = [{'params': x.parameters()} for x in self.models] 
        #self.params = chain(model.parameters() for model in self.models)

    def set_scalars(self,E,delta,interface_weight):
        super().set_scalars(E,delta)
        self.PINN_domain1.set_scalars(E,delta)
        self.PINN_domain2.set_scalars(E,delta)
        self.interface_weight = torch.tensor(interface_weight).to(self.device)

    
    def set_LevelSet(self,k,b):
        self.k = k
        self.b = b
        self.LevelSet = self.get_LevelSet()
        # 界面的外法线定义为levelset<0处的外法线
        self.outward_normal = np.arctan(k) + np.pi/2    
        self.l_x = torch.tensor(np.cos(self.outward_normal)).to(self.device)
        self.l_y = torch.tensor(np.cos(np.pi/2 -self.outward_normal)).to(self.device)
        

    def get_LevelSet(self):
        '''获取水平集函数'''

        def get_psi(x,y):
            '''获取法向水平集函数'''
            return y-(self.k*x+self.b)
        
        return get_psi    
    
    def pred_uv(self, xy):
        ls = self.LevelSet(xy[...,0],xy[...,1])
        u_1, v_1 = self.PINN_domain1.pred_uv(xy)
        u_2, v_2 = self.PINN_domain2.pred_uv(xy)
        u = torch.where(ls > 0, u_1, u_2)
        v = torch.where(ls > 0, v_1, v_2)
        return u, v


    def pred_stress(self, xy):
        ls = self.LevelSet(xy[...,0],xy[...,1])
        sx_1, sy_1, sxy_1 = self.PINN_domain1.pred_stress(xy)
        sx_2, sy_2, sxy_2 = self.PINN_domain2.pred_stress(xy)
        sx = torch.where(ls > 0, sx_1, sx_2)
        sy = torch.where(ls > 0, sy_1, sy_2)
        sxy = torch.where(ls > 0, sxy_1, sxy_2)
        return sx, sy, sxy
    
    def set_inner_points(self, internal_points: torch.Tensor, internal_points_pdf: torch.Tensor):
        '''该函数仅在采样点均匀分布,且面积均匀分割时有效'''

        x,y = internal_points[:,0] , internal_points[:,1]
        ls = self.LevelSet(x, y)
        '''查找分别属于两个区域的内部点，并过滤正好落在界面上的点'''
        points_1 , pdf_1 = internal_points[ls > 0,:] , internal_points_pdf[ls > 0] * 2
        points_2 , pdf_2 = internal_points[ls < 0,:] , internal_points_pdf[ls < 0] * 2
        self.PINN_domain1.set_inner_points(points_1 , pdf_1)
        self.PINN_domain2.set_inner_points(points_2 , pdf_2)
        
    def set_meshgrid_inner_points(self,xstart,xend,xnum,ystart,yend,ynum):
        if self.k != 0:
            raise Exception('不能用规则网格!')
        else:
            ynum_1 = int( ynum * (yend - self.b) / (yend - ystart) )
            ynum_2 = ynum - ynum_1
            self.PINN_domain1.set_meshgrid_inner_points(xstart,xend,xnum,self.b,yend,ynum_1)
            self.PINN_domain2.set_meshgrid_inner_points(xstart,xend,xnum,ystart,self.b,ynum_2)


    def setMaterial(self, E1, E2, mu1 = 0.3, mu2 = 0.3,type='plane stress'):
        self.PINN_domain1.setMaterial(E1,mu1,type)
        self.PINN_domain2.setMaterial(E2,mu2,type)

    def Equilibrium_loss(self) -> torch.Tensor:
        return torch.cat(self.PINN_domain1.Equilibrium_loss(),self.PINN_domain2.Equilibrium_loss())
    
    def E_int(self) -> torch.Tensor:
        return self.PINN_domain1.E_int() + self.PINN_domain2.E_int()  
    
    def E_ext(self) -> torch.Tensor:
        return self.PINN_domain1.E_ext() + self.PINN_domain2.E_ext()  
    
    def soft_BC_loss(self) -> torch.Tensor:
        return torch.cat([self.PINN_domain1.soft_BC_loss() , self.PINN_domain2.soft_BC_loss()])
    
    def eval(self):
        loss_array,loss_sum = self.get_loss()
        print(self.iter, loss_sum.cpu().detach().numpy())
        self.EarlyStopping(torch.sum(loss_array).item(),self.models)      #判断是否需要提前结束训练 
        if self.iter % 1000 == 0:

            print(self.Energy_loss.__name__,':',self.Energy_loss().cpu().detach().numpy())
            print(self.interface_loss.__name__,':',self.interface_loss().cpu().detach().numpy())
            print('domain_1:')
            self.PINN_domain1.print_loss()
            print('domain_2:')
            self.PINN_domain2.print_loss()        
            self.history.append(loss_sum.cpu().detach().numpy())
              
        if self.iter % 5000 == 0:
            self.save(self.path+str(self.iter)) 

    def save(self, name):
        name1 = name +'_1'
        name2 = name +'_2'
        self.PINN_domain1.save(name1)
        self.PINN_domain2.save(name2)
    
    def load(self, path, loadtype='state'):
        self.PINN_domain1.load(path+'_1', loadtype)
        self.PINN_domain2.load(path+'_2', loadtype)

    def interface_surface_force(self,sx,sy,sxy):
        px = self.l_x * sx + self.l_y * sxy
        py = self.l_x * sxy + self.l_y * sy
        return px , py

    def set_interface_points(self,x:torch.Tensor,y:torch.Tensor):
        self.interface_x , self.interface_y , self.interface_xy = self._set_points(x,y)
        self.interface_zero = torch.zeros_like(self.interface_x)

    def gen_interface_points(self,xstart = 0,xend = 1,num=500):
        x = np.random.rand(num) * (xend - xstart) + xstart
        y = self.k * x + self.b
        self.interface_x , self.interface_y , self.interface_xy = self._set_points(torch.tensor(x),torch.tensor(y))
        self.interface_zero = torch.zeros_like(self.interface_x)

    def add_interface_points(self,xstart = 0,xend = 1,num=500):
        x = np.random.rand(num) * (xend - xstart) + xstart
        y = self.k * x + self.b
        interface_x , interface_y = self.variable(torch.tensor(x)),self.variable(torch.tensor(y))
        x = torch.cat((self.interface_x,interface_x))
        y = torch.cat((self.interface_y,interface_y))
        self.interface_x , self.interface_y , self.interface_xy = self._set_points(x,y)
        self.interface_zero = torch.zeros_like(self.interface_x)

    def interface_u_loss(self):
        u_1,v_1 = self.PINN_domain1.pred_uv(self.interface_xy)

        u_2,v_2 = self.PINN_domain2.pred_uv(self.interface_xy)

        u_loss =  (self.criterion(u_1,u_2)) * self.u_scalar
        v_loss =  (self.criterion(v_1,v_2)) * self.u_scalar
        return torch.stack([u_loss,v_loss])
    
    def interface_F_loss(self):

        sx_1,sy_1,sxy_1 = self.PINN_domain1.pred_stress(self.interface_xy)
        px_1,py_1 = self.interface_surface_force(sx_1,sy_1,sxy_1)

        sx_2,sy_2,sxy_2 = self.PINN_domain2.pred_stress(self.interface_xy)
        px_2,py_2 = self.interface_surface_force(sx_2,sy_2,sxy_2)

        px_loss = self.criterion(px_1,px_2) * self.stress_scalar
        py_loss = self.criterion(py_1,py_2) * self.stress_scalar
        return torch.stack([px_loss,py_loss])

    def interface_loss(self):
        u_1,v_1 = self.PINN_domain1.pred_uv(self.interface_xy)
        eXX,eYY,eXY = self.compute_Strain(u_1,v_1,self.interface_xy)
        sx_1,sy_1,sxy_1 = self.PINN_domain1.constitutive(eXX,eYY,eXY)
        px_1,py_1 = self.interface_surface_force(sx_1,sy_1,sxy_1)

        u_2,v_2 = self.PINN_domain2.pred_uv(self.interface_xy)
        eXX,eYY,eXY = self.compute_Strain(u_2,v_2,self.interface_xy)
        sx_2,sy_2,sxy_2 = self.PINN_domain2.constitutive(eXX,eYY,eXY)
        px_2,py_2 = self.interface_surface_force(sx_2,sy_2,sxy_2)

        u_loss =  (self.criterion(u_1,u_2)) * self.u_scalar
        v_loss =  (self.criterion(v_1,v_2)) * self.u_scalar
        px_loss = self.criterion(px_1,px_2) * self.stress_scalar
        py_loss = self.criterion(py_1,py_2) * self.stress_scalar
        return torch.stack([u_loss,v_loss,px_loss,py_loss])

    
    def train(self,epochs = 100000, patience=100 , path = 'test'):
        self.path = path
        self.optimizer = torch.optim.Adam(params=self.params, lr= 0.01)
        self.EarlyStopping=MultiEarlyStopping(patience=patience,verbose=True,
                                              paths= [self.path + model + '.pth' for model in self.models_name])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2000,10000,30000], gamma = 0.2)
        for i in range(epochs):
            
            self.train_step()
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break

        self.save(path+'_final')

    def train_resampling(self,points_distribution:Stats2D,epochs = 100000, patience=100 ,points_num = 10000, path = 'test'):

        self.path = path
        self.optimizer = torch.optim.Adam(params=chain(self.PINN_domain1.model.parameters(), 
                                                        self.PINN_domain2.model.parameters()), lr= 0.005)
        self.EarlyStopping=MultiEarlyStopping(patience=patience,verbose=True,paths= [path+'_1.pth',path+'_2.pth'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=np.linspace(1000,epochs,4), gamma = 0.2)
        points_distribution = stats.Stats2D(stats.Uniform(0,1),stats.Uniform(0,1))
        for i in range(epochs):
            if self.iter % 500 == 0:
                points = AcceptanceSampling2D(0,1,0,1,points_num,points_distribution)
                points_pdf = points_distribution.pdf(points[:,0],points[:,1])
                self.set_inner_points(points,points_pdf)
            self.train_step()
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break



        self.save(path+'_final')
    




from WeightOptimizer import ReLoBRaLo

class BiDomain_ReLoBRaLo(BiDomain):
    def __init__(self, PINN_domain1: PINN2D, PINN_domain2: PINN2D):
        super().__init__(PINN_domain1, PINN_domain2)
        # self.device = torch.device("cpu")

    def loss_func(self) -> torch.Tensor:

        loss_array , loss_sum = self.get_loss()
        loss_sum.backward()
        self.weights = self.weight_optimizer.step(loss_array.detach())
        return loss_sum

    def train(self, epochs=100000, patience=1000, path='test'):
        self.optimizer = torch.optim.Adam(params=chain(self.PINN_domain1.model.parameters(), 
                                                self.PINN_domain2.model.parameters()), lr= 0.002)
        self.weight_optimizer = ReLoBRaLo(self.weights,self.get_loss_terms().detach())
        self.EarlyStopping=MultiEarlyStopping(patience=patience,verbose=True,paths= [path+'_1.pth',path+'_2.pth'])
        self.path = path
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2000,int(epochs*0.75)], gamma = 0.5)
        for i in range(epochs):
            self.train_step()
            scheduler.step()

        self.save(path+'_final')
    


    def eval(self):
        super().eval()
        
        if self.iter % 1000 == 0:
            print('weights:')
            print([loss.__name__ for loss in self.losses])
            print(self.weights.cpu().detach().numpy())