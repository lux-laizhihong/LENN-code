import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from get_grad import get_grad
from EarlyStopping import EarlyStopping,MultiEarlyStopping
from Integral import montecarlo,trapz2D,simps2D
from itertools import chain
from NodesGenerater import AcceptanceSampling2D,genMeshNodes2D,genRandomNodes2D,genHeteroTip2D,genDenseCirques

from stats import Stats2D
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt
from matplotlib import colors,cm
import stats
from shapely.geometry import Point, Polygon
from Geometry import Geometry1D
import Geometry
'''
请注意,所有domain对象中的域内点属性self.X,self.Y只能在被求导时使用。
千万不能用于推理！！！！！不然会导致求导出错！！！！！
'''

class Domain2D:
    def __init__(self) -> None:
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def set_geometry(self,polygon:Polygon):
        self.polygon = polygon

    def is_in_domain(self,xy:torch.Tensor):
        xy = xy.cpu().detach().numpy()
        return [self.polygon.covers(Point(xy[i,0],xy[i,1])) for i in range(xy.shape[0])]

    def setMaterial(self,E,nu,type='plane stress'):
        '''杨氏模量单位为MPa'''
        self.E = E
        self.mu = nu
        self.setD(self.E,self.mu,type=type)

    def setMaterial_Bi(self, E1, E2, nu1 = 0.3, nu2 = 0.3 ,type='plane stress'):
        '''1为线段左边,2为右边'''
        self.E = np.array([E1,E2])
        self.nu = np.array([nu1,nu2])
        self.setD(self.E , self.nu,type=type)

    def setD(self,E,mu,type='plane stress'):
        if type=='plane stress':
            self.d11 = torch.tensor(E/(1-mu*mu)).to(self.device)
            self.d12 = torch.tensor(E/(1-mu*mu)*mu).to(self.device)
            self.G = torch.tensor(E/(2*(1+mu))).to(self.device)
        elif type == 'plane strain':
            self.d11 = torch.tensor(E*(1-mu)/((1+mu)*(1-2*mu))).to(self.device)
            self.d12 = torch.tensor(E*mu/((1+mu)*(1-2*mu))).to(self.device)
            self.G = torch.tensor(E/(2*(1+mu))).to(self.device)

        else: raise Exception('error!')

    def variable(self,x:torch.Tensor):
        return x.float().requires_grad_().to(device=self.device)
    
    def _set_points(self,x,y)->torch.Tensor:
        x = self.variable(x)
        y = self.variable(y)
        xy = torch.stack([x,y],dim=1)
        return x , y , xy
    
    def gen_uniform_points(self,num):...


class Domain2D_rect(Domain2D):
    def __init__(self,left,right,bottom,top) -> None:
        super().__init__()
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.polygon = Polygon(([(left, bottom),
                                (left, top),
                                (right, top),
                                (right, bottom)]))
    
    def gen_uniform_points(self,num):
        xy = genRandomNodes2D(self.left,self.right,self.bottom,self.top,num)
        return xy, torch.ones_like(xy[...,0]) / self.polygon.area

#Pa,N,m
#模型输出位移是mm
#输出应力单位MPa
class PINN2D(Domain2D):
    def __init__(self,model:nn.Module):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = model
        self.model.to(self.device)
        self.criterion = torch.nn.MSELoss()
        self.iter = 1
        self.history = []
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)
        #self.stress_scalar = torch.tensor(100.0).to(self.device)
        # self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001)
        self.model.apply(self.weight_init)    #初始化权重
        self.integral = montecarlo
        self.E_int = self.E_int_montecarlo
        self.params = self.model.parameters()
        # self.LBFGS = torch.optim.LBFGS(self.model.parameters(), lr=1e-2)

    def set_inner_points(self,internal_points,internal_points_pdf: torch.Tensor,variable = True):
        '''设置域内的采样点'''
        '''单独的X和Y用于在单方向求导时节约计算量,
           推理时使用XY,便于同时对两个方向求导
           目前只有计算平衡方程损失时用到了单独的X与Y,请注意检查程序
        '''        

        if variable:
            self.X,self.Y,self.XY = self._set_points(internal_points[:,0] , internal_points[:,1])
        else:
            self.X,self.Y,self.XY = self._set_points(internal_points[0] , internal_points[1])

        if type(internal_points_pdf) is np.ndarray:
            self.points_pdf = torch.from_numpy(internal_points_pdf).to(device=self.device)
        elif torch.is_tensor(internal_points_pdf):
            self.points_pdf = internal_points_pdf.to(device=self.device)
        else:
            raise Exception('Type error!')
        self.zero=torch.zeros_like(self.X) 
        # self.integral = montecarlo
        # self.E_int = self.E_int_montecarlo     


    def set_cirques_inner_points(self,left, right, bottom, top, outer_num,eplison,circle_x,circle_y,
                    circles:list[Geometry.Circle],inner_num,variable = True):
        points,points_pdf = genDenseCirques(left, right, bottom, top, outer_num,eplison,circle_x,circle_y,circles,inner_num)

        if variable:
            self.X,self.Y,self.XY = self._set_points(points[:,0] , points[:,1])
        else:
            self.X,self.Y,self.XY = self._set_points(points[0] , points[1])
        if type(points_pdf) is np.ndarray:
            self.points_pdf = torch.from_numpy(points_pdf).to(device=self.device)
        elif torch.is_tensor(points_pdf):
            self.points_pdf = points_pdf.to(device=self.device)
        else:
            raise Exception('Type error!')

        # self.integral = montecarlo
        self.E_int = self.E_int_montecarlo

    def get_energy_density(self,xy):
        u,v,result = self.pred_uv(xy)
        eXX,eYY,eXY = self.compute_Strain(u,v,xy)
        sx,sy,sxy = self.constitutive(eXX,eYY,eXY)
        energy =  0.5 * (eXX * sx + eYY * sy + eXY * sxy)
        return energy

    def set_meshgrid_inner_points(self,xstart,xend,xnum,ystart,yend,ynum):
        '''生成规则网格排布点'''
        x,y = genMeshNodes2D(xstart,xend,xnum,ystart,yend,ynum)
        self.X, self.Y, self.XY = self._set_points(x, y)
        stat = Stats2D(stats.Uniform(xstart,xend),stats.Uniform(ystart,yend))
        self.points_pdf = torch.from_numpy(stat.pdf(self.X.detach().cpu(),self.Y.detach().cpu())).to(device=self.device)
        self.xshape = xnum
        self.yshape = ynum

        def E_int_trapz():
            energy = self.get_energy_density(self.XY)
            return trapz2D(energy , self.XY , [self.xshape,self.yshape])
        #
        self.E_int = E_int_trapz
        # self.E_int = self.E_int_montecarlo
        self.zero = torch.zeros_like(self.X) 

    # def set_meshgrid_inner_points_scalar(self,xstart,xend,xnum,ystart,yend,ynum,scalar,translate):
    #     '''生成规则网格排布点'''
    #     x,y = genMeshNodes2D(xstart,xend,xnum,ystart,yend,ynum)
    #     x = scalar*x+translate
    #     y = scalar*y+translate
    #     self.X, self.Y, self.XY = self._set_points(x, y)
    #     stat = Stats2D(stats.Uniform(xstart,xend),stats.Uniform(ystart,yend))
    #     self.points_pdf = torch.from_numpy(stat.pdf(self.X.detach().cpu(),self.Y.detach().cpu())).to(device=self.device)
    #     self.xshape = xnum
    #     self.yshape = ynum
    #
    #     def E_int_trapz():
    #         energy = self.get_energy_density(self.XY)
    #         return trapz2D(energy , self.XY , [self.xshape,self.yshape])
    #     #
    #     self.E_int = E_int_trapz
    #     # self.E_int = self.E_int_montecarlo
    #     self.zero = torch.zeros_like(self.X)


    def set_meshgrid_trapz_Tip_Dense(self,xstart,xend,
                    ystart,yend,
                    xTip,yTip,
                    x_dense_num,y_dense_num,
                    x_outer_num,y_outer_num,
                    x_inteval=0.2,y_inteval=0.2):
        '''生成规则网格排布点'''
        x,y = genHeteroTip2D(xstart,xend,
                            ystart,yend,
                            xTip,yTip,
                            x_dense_num,y_dense_num,
                            x_outer_num,y_outer_num,
                            x_inteval=x_inteval,
                            y_inteval=y_inteval)
        self.X,self.Y,self.XY = self._set_points(x,y)
        self.xshape = x_dense_num + x_outer_num
        self.yshape = y_dense_num + y_outer_num
        def E_int_trapz():
            energy = self.get_energy_density(self.XY)
            return trapz2D(energy , self.XY , [self.xshape,self.yshape])
        self.E_int = E_int_trapz
        self.zero = torch.zeros_like(self.X)

    def set_meshgrid_inner_points_formuti(self,xstart,xend,xnum,ystart,yend,ynum,X0,Y0):
        '''生成规则网格排布点'''
        x,y = genMeshNodes2D(xstart,xend,xnum,ystart,yend,ynum)

        self.X, self.Y, self.XY = self._set_points(x, y)
        stat = Stats2D(stats.Uniform(xstart,xend),stats.Uniform(ystart,yend))
        # self.points_pdf = torch.from_numpy(stat.pdf(self.X.detach().cpu(),self.Y.detach().cpu())).to(device=self.device)
        self.xshape = xnum
        self.yshape = ynum

        def E_int_trapz():
            energy = self.get_energy_density(self.XY)
            return trapz2D(energy , self.XY , [self.xshape,self.yshape])
        #
        self.E_int = E_int_trapz
        # self.E_int = self.E_int_montecarlo
        self.zero = torch.zeros_like(self.X)

    def set_meshgrid_simps_points(self,xstart,xend,x_interval,
                                  ystart,yend,y_interval):
        xnum = x_interval *2 -1
        ynum = y_interval *2 -1
        self.set_meshgrid_inner_points(xstart,xend,xnum,ystart,yend,ynum)
        def E_int_simps():
            energy = self.get_energy_density(self.XY)
            return simps2D(energy , self.XY , [self.xshape,self.yshape])
        self.E_int = E_int_simps

    def add_BCPoints(self):...

    def weight_init(self,m):
        '''初始化模型参数'''
        # net.modules()
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight)
 
    def hard_u(self,u,x,y):
        return u

    def hard_v(self,v,x,y):
        return v

    def mm_to_m(self,u,v):
        return u/1000,v/1000

    def pred_uv(self,xy):
        uv,result = self.model(xy)
        u,v = self.hard_u(uv[0].squeeze(-1),xy[...,0],xy[...,1]) , self.hard_v(uv[1].squeeze(-1),xy[...,0],xy[...,1])      
        return u,v,result
    
    def compute_Strain(self,u,v,xy):
        # uv = self.model(xy)
        # u,v = self.hard_u(uv[0].squeeze(-1),xy[...,0],xy[...,1]) , self.hard_v(uv[1].squeeze(-1),xy[...,0],xy[...,1])
        u,v = self.mm_to_m(u,v)
        # uv=torch.stack(uv).squeeze(-1)
        # duv_dxy = jacobian(uv,xy)
        du_dxy = get_grad(u,xy)
        dv_dxy = get_grad(v,xy)
        eXX = du_dxy[...,0]
        eYY = dv_dxy[...,1]
        eXY = dv_dxy[...,0] + du_dxy[...,1]
        return eXX,eYY,eXY

    def constitutive(self,eXX,eYY,eXY):
        sx = self.d11 * eXX + self.d12 * eYY
        sy = self.d12 * eXX + self.d11 * eYY
        sxy = self.G * eXY
        return sx , sy , sxy
    
    def Equilibrium_loss(self)->torch.Tensor:
        '''无体力的平衡方程'''
        sx,sy,sxy = self.pred_stress(self.XY)
        dsx_dx = get_grad(sx,self.X)
        dsy_dy = get_grad(sy,self.Y)
        dsxy_dxy = get_grad(sxy,self.XY)
        dsxy_dx , dsxy_dy = dsxy_dxy[...,0] , dsxy_dxy[...,1]
        fx_loss = self.criterion(dsx_dx + dsxy_dy,self.zero)
        fy_loss = self.criterion(dsy_dy + dsxy_dx,self.zero)
        # return fx_loss + fy_loss
        return torch.stack([fx_loss,fy_loss]) #*  self.stress_scalar      

    def E_int_montecarlo(self)->torch.Tensor:
        energy = self.get_energy_density(self.XY)
        return montecarlo(energy , self.points_pdf)

    def E_ext(self)->torch.Tensor:
        return torch.tensor(0.0,device=self.device)

    def Energy_loss(self)->torch.Tensor:
        return torch.stack([self.E_int() - self.E_ext()])

    def set_scalars(self,E,delta):
        self.Equilibrium_scalar = torch.tensor(1.0).to(self.device)
        self.stress_scalar = torch.tensor((1/delta)**2).to(self.device)
        self.u_scalar = torch.tensor((E/(delta**2))**2).to(self.device)

    def pred_stress(self,xy):...
    def infer(self,xy):...

    def pde_loss(self)->torch.Tensor:...

    def soft_BC_loss(self)->torch.Tensor:
        return torch.tensor(0.0,device=self.device)
    
    def set_loss_func(self,losses:list,weights = None):
        self.losses = losses
        if weights is None:
            self.weights = torch.Tensor([1.0]*self.get_loss_terms().shape[0]).to(self.device)
        else:
            self.weights = torch.Tensor(weights).to(self.device)

    def get_loss_terms(self)->torch.Tensor:
        return torch.cat(list(map(lambda x: x(),self.losses)))

    def get_loss(self) -> torch.Tensor:
        loss_array = self.get_loss_terms()
        loss_sum = torch.sum(self.weights * loss_array)
        return loss_array , loss_sum

    def loss_func(self) -> torch.Tensor:

        loss_array , loss_sum = self.get_loss()
        loss_sum.backward()
        return loss_sum

    def print_loss(self):
        print(  self.Equilibrium_loss.__name__,':',self.Equilibrium_loss().cpu().detach().numpy(),
                self.Energy_loss.__name__,':',self.Energy_loss().cpu().detach().numpy(),
                self.soft_BC_loss.__name__,':',self.soft_BC_loss().cpu().detach().numpy())   

    def eval(self):
        loss_array,loss_sum = self.get_loss()
        print(self.iter, loss_sum.cpu().detach().numpy())
        self.print_loss()
        self.history.append(loss_sum.cpu().detach().numpy())
        self.EarlyStopping(loss_sum.cpu().detach().numpy(),self.model)      #判断是否需要提前结束训练  
        # self.scheduler.step(loss_sum)
        if self.iter % 2000 == 0:
            print(loss_array.cpu().detach().numpy())             


    def train_step(self,eval_sep:int = 100):
        self.optimizer.zero_grad()
        self.optimizer.step(self.loss_func)
        self.iter = self.iter + 1 

        # if self.iter % 10000 == 0: 
        #     self.save(self.path+str(self.iter))
        #     self.save_hist(self.path)

    def set_Optimizer(self,lr):
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr= lr)
    
    def set_EarlyStopping(self,patience,verbose,path):
        self.EarlyStopping=EarlyStopping(patience,verbose=verbose,path=path+'.pth')

    def train(self,epochs = 100000, patience=1000 , path = 'test', lr= 0.005,eval_sep=100):
        self.iter = 0
        self.set_EarlyStopping(patience=patience,verbose=True,path= path)
        self.path = path
        self.set_Optimizer(lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[15000,60000,120000,200000], gamma = 0.5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[120000,200000], gamma = 0.5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5000,10000,15000,20000], gamma = 0.5)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[15000], gamma = 0.5)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5,cooldown=1,factor=0.5,verbose=True)
        for i in range(epochs):
            self.train_step(eval_sep=eval_sep)

            if self.iter % eval_sep == 0: 
                self.eval()

            # loss_array , loss_sum = self.get_loss()
            
            scheduler.step()
            # self.EarlyStopping(loss_sum.cpu().detach().numpy(),self.model)      #判断是否需要提前结束训练  
            if (self.EarlyStopping.early_stop):
                print('end epoch:'+str(self.iter))
                break
        self.save_hist(self.path)

        # self.save(path+'_final')

    def train_resampling(self,points_distribution:Stats2D,num,epochs = 100000, patience=100,path = 'test',
                         left=0, right=1, bottom=0, top=1):

        self.EarlyStopping=EarlyStopping(patience=patience,verbose=True,path= path+'.pth')
        self.path = path
        self.optimizer = torch.optim.Adam(params=self.params, lr= 0.001)
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
        # self.save(path+'_final')



    def save_hist(self,name):
        hist = pd.DataFrame(self.history)
        hist.to_csv(name+'.csv')

    def save(self,name):
        # torch.save(self.model,name+'.pth')
        self.save_hist(name)
        # torch.save(self.model.state_dict(),'state'+name+'.pth')

    def load(self,path,loadtype='state'):
        if loadtype=='state':
            self.model.load_state_dict(torch.load(path+'.pth',map_location=torch.device('cpu')))
        else:
            self.model=torch.load(path+'.pth',map_location=torch.device('cpu'))

    def showStress(self,X,Y,Z,ax = plt.subplot(1,1,1),vmax = None,show = False,colorbar_norm = None,cmap = 'jet', cbar=True,levels=200,cbar_shrink=1.0)->plt.Axes:

        nx = np.unique(X).size
        ny = np.unique(Y).size

        X_grid = X.reshape(nx, ny)
        Y_grid = Y.reshape(nx, ny)
        Z_grid = Z.reshape(nx, ny)

        # X,Y=np.meshgrid(X,Y)
        # cmap='jet'
        # cmap='bwr'
        if vmax:
            max = min([0.2,np.max(Z)])
            plot = ax.contourf(X_grid,Y_grid,Z_grid,cmap=cmap,levels=np.linspace(0,max,200),extend='both')
            # plot = ax.imshow(Z_grid,cmap=cmap,origin='lower',levels=np.linspace(0,max,20),extend='both')
            # cb = plt.colorbar(plot,ax=ax)
        elif colorbar_norm is not None:
            plot = ax.contourf(X_grid,Y_grid,Z_grid,levels,cmap=cmap,norm = colorbar_norm)
            # #plot = ax.pcolormesh(X_grid,Y_grid,Z_grid,cmap=cmap,norm = colorbar_norm)
            # plot = ax.imshow(Z_grid,cmap=cmap,origin='lower',norm=colorbar_norm)
            # cb = plt.colorbar(cm.ScalarMappable(norm=colorbar_norm, cmap=cmap), ax=ax)
        else:
            plot = ax.contourf(X_grid, Y_grid, Z_grid, levels=np.linspace(0,0.2,200), cmap=cmap,extend='both')
            plot = ax.contourf(X_grid, Y_grid, Z_grid, levels, cmap=cmap)
            # plot = ax.pcolormesh(X_grid,Y_grid,Z_grid,cmap=cmap)1
            # plot = ax.imshow(Z_grid,cmap=cmap,origin='lower')
            # cb = plt.colorbar(plot)
            # cb = plt.colorbar(plot,ax=ax)

        if cbar:
            cb = self.plot_cbar(ax=ax,plot=plot,colorbar_norm=colorbar_norm,cmap=cmap,shrink=cbar_shrink)

        ax.axis('equal')
        ax.autoscale(tight=True)
        ax.axis('off')
        if show:
            plt.show()
            return
        return plot

    def plot_cbar(self,ax,plot,cmap='jet',colorbar_norm=None,cax=None,shrink=1.0):
        '''
        画颜色图
        '''
        if colorbar_norm is not None:
            cb = plt.colorbar(cm.ScalarMappable(norm=colorbar_norm, cmap=cmap), ax=ax, cax=cax,shrink=shrink)
        else:
            cb = plt.colorbar(plot,ax=ax,cax=cax,shrink=shrink)

        # 设置颜色条标签为科学计数法
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2,2))
        cb.formatter = formatter
        cb.update_ticks()        

        return cb

    def readData(self,path):
        self.labeled=True
        df=pd.read_csv(path,skiprows=9,names=['x','y','u','v','sx','sy','sxy'],delim_whitespace=True)  #分割为空白字符
        # df = df.sample(9)
        self.labeled_x = torch.tensor(df['x'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_y = torch.tensor(df['y'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_xy = torch.stack([self.labeled_x,self.labeled_y],dim=1)
        self.labeled_u = torch.tensor(df['u'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_v = torch.tensor(df['v'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_sx = torch.tensor(df['sx'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_sy = torch.tensor(df['sy'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)       
        self.labeled_sxy = torch.tensor(df['sxy'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)

    def readData_sampling(self,path,num):
        self.labeled=True
        df=pd.read_csv(path,skiprows=9,names=['x','y','u','v','sx','sy','sxy'],delim_whitespace=True)  #分割为空白字符
        df = df.sample(num)
        self.labeled_x = torch.tensor(df['x'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_y = torch.tensor(df['y'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_xy = torch.stack([self.labeled_x,self.labeled_y],dim=1)
        self.labeled_u = torch.tensor(df['u'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_v = torch.tensor(df['v'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_sx = torch.tensor(df['sx'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_sy = torch.tensor(df['sy'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)       
        self.labeled_sxy = torch.tensor(df['sxy'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)

    def readDataStrain(self,path):
        self.labeled=True
        df=pd.read_csv(path,skiprows=9,names=['x','y','u','v','eXX','eYY','eXY'],delim_whitespace=True)  #分割为空白字符
        # df = df.sample(9)
        self.labeled_x = torch.tensor(df['x'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_y = torch.tensor(df['y'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_xy = torch.stack([self.labeled_x,self.labeled_y],dim=1)
        self.labeled_u = torch.tensor(df['u'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_v = torch.tensor(df['v'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_eXX = torch.tensor(df['eXX'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_eYY = torch.tensor(df['eYY'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)       
        self.labeled_eXY = torch.tensor(df['eXY'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)

    def readData_condition(self,path,x_condition,y_condition):
        self.labeled=True
        df=pd.read_csv(path,skiprows=9,names=['x','y','u','v','sx','sy','sxy'],delim_whitespace=True)  #分割为空白字符
        df = df[df["x"].map(x_condition)]
        df = df[df["y"].map(y_condition)]
        self.labeled_x = torch.tensor(df['x'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_y = torch.tensor(df['y'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_xy = torch.stack([self.labeled_x,self.labeled_y],dim=1)
        self.labeled_u = torch.tensor(df['u'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_v = torch.tensor(df['v'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_sx = torch.tensor(df['sx'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
        self.labeled_sy = torch.tensor(df['sy'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)       
        self.labeled_sxy = torch.tensor(df['sxy'].to_numpy(),dtype=torch.float,requires_grad=True).to(self.device)
    
    def set_label_xy(self,x,y):
        self.labeled_x , self.labeled_y, self.labeled_xy = self._set_points(x,y)

 
    def dataLoss_u_strain(self)->torch.Tensor:
        u,v = self.pred_uv(self.labeled_xy)
        eXX,eYY,eXY = self.compute_Strain(u,v,self.labeled_xy)
        return torch.stack([self.criterion(u,self.labeled_u),
                            self.criterion(v,self.labeled_v),
                            self.criterion(eXX,self.labeled_eXX),
                            self.criterion(eYY,self.labeled_eYY),
                            self.criterion(eXY,self.labeled_eXY)])


    def dataLoss_strain(self)->torch.Tensor:
        u,v = self.pred_uv(self.labeled_xy)
        eXX,eYY,eXY = self.compute_Strain(u,v,self.labeled_xy)
        return torch.stack([self.criterion(eXX,self.labeled_eXX),
                            self.criterion(eYY,self.labeled_eYY),
                            self.criterion(eXY,self.labeled_eXY)])
    
    def dataLoss_u(self)->torch.Tensor:
        u,v = self.pred_uv(self.labeled_xy)
        return torch.stack([self.criterion(u,self.labeled_u),self.criterion(v,self.labeled_v)])
    
    def dataLoss(self)->torch.Tensor:
        u,v,sx,sy,sxy = self.infer(self.labeled_xy)
        return torch.stack([self.criterion(u,self.labeled_u),
                            self.criterion(v,self.labeled_v),
                            self.criterion(sx,self.labeled_sx),
                            self.criterion(sy,self.labeled_sy),
                            self.criterion(sxy,self.labeled_sxy)])
    
    def plot_error(self,x,y,z,z_refer,title='output',name = None,levels=200):
        import matplotlib.pyplot as plt


        nan_indices = np.argwhere(np.isnan(z))
        # 替换 NaN 值
        for idx in nan_indices:
            if idx + 1 < len(z):
                z[idx] = z[idx + 1]
            else:
                print(f"NaN at position {idx} has no right neighbor to replace.")

        print("NaN values found at indices:", nan_indices)
        print(z[nan_indices], x[nan_indices], y[nan_indices])

        z_min_plot = np.min(z_refer)
        z_max_plot = np.max(z_refer)
        z = np.clip(z, z_min_plot, z_max_plot)

        abs_error = np.abs((z - z_refer))
        norm_abs_error = np.abs((z - z_refer)/np.max(z_refer))
        relative_error = np.abs(z - z_refer)/(np.abs(z_refer)+1e-3*np.max(np.abs(z_refer)))
        
        # fig = plt.figure()

        # vmin = min(np.min(z_refer), np.min(z))
        # vmax = max(np.max(z_refer), np.max(z))
        # norm = colors.Normalize(vmin=vmin, vmax=vmax)
        norm = colors.Normalize(vmin=np.min(z_refer), vmax=np.max(z_refer))
        #norm = colors.Normalize(vmin=np.min(z), vmax=np.max(z))
        # norm = colors.Normalize(vmin=(np.min(z_refer)+np.min(z))/2,vmax=(np.max(z_refer)+np.max(z))/2)
        # with np.printoptions(threshold=np.inf):
        #  print(z.reshape(101,101))

        print(np.max(z),np.min(z))
        print(np.argmax(z),np.argmin(z))
        print(z_refer[np.argmax(z)],z_refer[np.argmin(z)])
        print(x[np.argmax(z)],y[np.argmax(z)])
        print(x[np.argmin(z)], y[np.argmin(z)])


        print(np.max(z_refer), np.min(z_refer))
        print(np.argmax(z_refer),np.argmin(z_refer))
        print(z[np.argmax(z_refer)],z[np.argmin(z_refer)])
        print(x[np.argmax(z_refer)],y[np.argmax(z_refer)])
        print(x[np.argmin(z_refer)], y[np.argmin(z_refer)])

        print(np.max(norm_abs_error))
        print(np.max(np.abs((z - z_refer))))
        print(np.max(relative_error))

        ax = plt.subplot(2,2,2)
        plot = self.showStress(x,y,z_refer,ax,colorbar_norm=norm,cmap='jet',levels=levels)
        ax.set_title('FEM')

        bx = plt.subplot(2,2,1)
        plot = self.showStress(x,y,z,bx,colorbar_norm=norm,cmap='jet',levels=levels)
        # plot = self.showStress(x,y,z,bx,cmap='jet',levels=levels)
        # plot = self.showStress(x,y,z,ax)
        bx.set_title('NN') 

        
        # cb = fig.colorbar(plot,ax = [ax,bx])
        # # 设置颜色条标签为科学计数法
        # formatter = ScalarFormatter(useMathText=True)
        # formatter.set_powerlimits((-2,2))
        # cb.formatter = formatter
        # cb.update_ticks()        # ax = plt.subplot(2,2,2)
        # # plot = self.showStress(x,y,z_refer,ax,cmap='jet')
        # # ax.set_title('FEM')
               
        ax = plt.subplot(2,2,3)        
        # plot = self.showStress(x,y,abs_error,ax)
        vmax = True if np.max(norm_abs_error) > 0.5 else False
        plot = self.showStress(x,y,norm_abs_error,ax,vmax,levels=levels)   
        # plot = self.showStress(x,y,norm_abs_error,ax)
        # ax.set_title('abs error')
        ax.set_title('normed abs error')

        ax = plt.subplot(2,2,4)
        vmax = True if np.max(relative_error) > 0.5 else False
        plot = self.showStress(x,y,relative_error,ax,vmax,levels=levels)   
        ax.set_title('relative error') 
        plt.suptitle(title,fontsize = 15)  
        plt.tight_layout()  
        
        if name is None:
            plt.show()
        else:
            plt.savefig(name+title+'.jpg', dpi=300)


    def showResult(self,xy):
        u,v = self.pred_uv(xy)
        u=u.cpu().detach().numpy()
        v=v.cpu().detach().numpy()

        x=xy[...,0].cpu().detach().numpy()
        y=xy[...,1].cpu().detach().numpy()
        # plt.figure()
        self.showStress(x,y,u,show=True)
        plt.figure()
        self.showStress(x,y,v,show=True,ax=plt.subplot(1,1,1))
        sx,sy,sxy = self.pred_stress(xy)
        sx = sx.cpu().detach().numpy()
        sy = sy.cpu().detach().numpy()
        sxy = sxy.cpu().detach().numpy()
        plt.figure()
        self.showStress(x,y,sx,show=True,ax=plt.subplot(1,1,1))
        plt.figure()
        self.showStress(x,y,sy,show=True,ax=plt.subplot(1,1,1))
        plt.figure()
        self.showStress(x,y,sxy,show=True,ax=plt.subplot(1,1,1))

        # plt.draw()
        # plt.show()

    def evaluate(self,name = None,levels=200):
        u,v = self.pred_uv(self.labeled_xy)

        u=u.cpu().detach().numpy()
        v=v.cpu().detach().numpy()

        x=self.labeled_x.cpu().detach().numpy()
        y=self.labeled_y.cpu().detach().numpy()
        u_refer = self.labeled_u.cpu().detach().numpy()
        v_refer = self.labeled_v.cpu().detach().numpy()
        self.plot_error(x,y,u,u_refer,title='u',name=name,levels=levels)
        self.plot_error(x,y,v,v_refer,title='v',name=name,levels=levels)
        sx,sy,sxy = self.pred_stress(self.labeled_xy)
        sx = sx.cpu().detach().numpy()
        sy = sy.cpu().detach().numpy()
        sxy = sxy.cpu().detach().numpy()
        sx_refer = self.labeled_sx.cpu().detach().numpy()
        sy_refer = self.labeled_sy.cpu().detach().numpy()
        sxy_refer = self.labeled_sxy.cpu().detach().numpy()
        self.plot_error(x,y,sx,sx_refer,title='sx',name=name,levels=levels)
        self.plot_error(x,y,sy,sy_refer,title='sy',name=name,levels=levels)
        self.plot_error(x,y,sxy,sxy_refer,title='sxy',name=name,levels=levels)    


#Pa,N,m

    def l2_error(self,name = None,levels=200):
        u,v = self.pred_uv(self.labeled_xy)
        u=u.cpu().detach().numpy()
        v=v.cpu().detach().numpy()
        x =self.labeled_x.cpu().detach().numpy()
        n = x.size
        y=self.labeled_y.cpu().detach().numpy()
        u_refer = self.labeled_u.cpu().detach().numpy()
        v_refer = self.labeled_v.cpu().detach().numpy()

        sx,sy,sxy = self.pred_stress(self.labeled_xy)
        sx = sx.cpu().detach().numpy()
        sy = sy.cpu().detach().numpy()
        sxy = sxy.cpu().detach().numpy()
        # print(sx,sy)
        sx_refer = self.labeled_sx.cpu().detach().numpy()
        sy_refer = self.labeled_sy.cpu().detach().numpy()
        sxy_refer = self.labeled_sxy.cpu().detach().numpy()
        u_error = np.linalg.norm(u - u_refer) / n
        v_error = np.linalg.norm(v - v_refer) / n
        sx_error = np.linalg.norm(sx - sx_refer) / n
        sy_error = np.linalg.norm(sy - sy_refer) / n
        sxy_error = np.linalg.norm(sxy - sxy_refer) / n
        print(f"L2 error for u: {u_error}")
        print(f"L2 error for v: {v_error}")
        print(f"L2 error for sx: {sx_error}")
        print(f"L2 error for sy: {sy_error}")
        print(f"L2 error for sxy: {sxy_error}")

    def R2_squre(self,name = None,levels=200):
        u,v = self.pred_uv(self.labeled_xy)
        u=u.cpu().detach().numpy()
        v=v.cpu().detach().numpy()
        x =self.labeled_x.cpu().detach().numpy()
        n = x.size
        y=self.labeled_y.cpu().detach().numpy()
        u_refer = self.labeled_u.cpu().detach().numpy()
        v_refer = self.labeled_v.cpu().detach().numpy()

        sx,sy,sxy = self.pred_stress(self.labeled_xy)
        sx = sx.cpu().detach().numpy()
        sy = sy.cpu().detach().numpy()
        sxy = sxy.cpu().detach().numpy()
        # print(sx,sy)
        sx_refer = self.labeled_sx.cpu().detach().numpy()
        sy_refer = self.labeled_sy.cpu().detach().numpy()
        sxy_refer = self.labeled_sxy.cpu().detach().numpy()

        def r2_score(y_true, y_pred):
            # 计算真实值的均值
            y_mean = np.mean(y_true)
            # 计算总变异
            total_variance = np.sum((y_true - y_mean) ** 2)
            # 计算残差变异
            residual_variance = np.sum((y_true - y_pred) ** 2)
            # 计算 R²
            return 1 - (residual_variance / total_variance)

        u_r2 = r2_score(u_refer, u)
        v_r2 = r2_score(v_refer, v)
        sx_r2 = r2_score(sx_refer, sx)
        sy_r2 = r2_score(sy_refer, sy)
        sxy_r2 = r2_score(sxy_refer, sxy)
        print(f"R² for u: {u_r2}")
        print(f"R² for v: {v_r2}")
        print(f"R² for sx: {sx_r2}")
        print(f"R² for sy: {sy_r2}")
        print(f"R² for sxy: {sxy_r2}")

    def plot_result_NN(self):
     u_model, v_model = self.pred_uv(self.labeled_xy)
     u_model = u_model.cpu().detach().numpy()
     v_model = v_model.cpu().detach().numpy()
     x = self.labeled_x.cpu().detach().numpy()
     y = self.labeled_y.cpu().detach().numpy()
     u_refer = self.labeled_u.cpu().detach().numpy()
     v_refer = self.labeled_v.cpu().detach().numpy()
     sx_model, sy_model, sxy_model = self.pred_stress(self.labeled_xy)
     sx_model = sx_model.cpu().detach().numpy()
     sy_model = sy_model.cpu().detach().numpy()
     sxy_model = sxy_model.cpu().detach().numpy()
     sx_refer = self.labeled_sx.cpu().detach().numpy()
     sy_refer = self.labeled_sy.cpu().detach().numpy()
     sxy_refer = self.labeled_sxy.cpu().detach().numpy()

     u_min_plot = np.min(u_refer)
     u_max_plot = np.max(u_refer)
     v_min_plot = np.min(v_refer)
     v_max_plot = np.max(v_refer)
     sx_min_plot = np.min(sx_refer)
     sx_max_plot = np.max(sx_refer)
     sy_min_plot = np.min(sy_refer)
     sy_max_plot = np.max(sy_refer)
     sxy_min_plot = np.min(sxy_refer)
     sxy_max_plot = np.max(sxy_refer)
     # u_model = u_model+0.466
     u_model = np.clip(u_model, u_min_plot, u_max_plot)
     v_model = np.clip(v_model, v_min_plot, v_max_plot)
     sx_model = np.clip(sx_model, sx_min_plot, sx_max_plot)
     sy_model = np.clip(sy_model, sy_min_plot, sy_max_plot)
     sxy_model = np.clip(sxy_model, sxy_min_plot, sxy_max_plot)
     u_norm = colors.Normalize(vmin=np.min(u_refer), vmax=np.max(u_refer))
     v_norm = colors.Normalize(vmin=np.min(v_refer), vmax=np.max(v_refer))
     sx_norm = colors.Normalize(vmin=np.min(sx_refer), vmax=np.max(sx_refer))
     sy_norm = colors.Normalize(vmin=np.min(sy_refer), vmax=np.max(sy_refer))
     sxy_norm = colors.Normalize(vmin=np.min(sxy_refer), vmax=np.max(sxy_refer))


     plt.figure(figsize=(3, 16))
     ax = plt.subplot(5, 1, 1)
     plot = self.showStress(x, y, u_model, ax, colorbar_norm=u_norm, cmap='jet', levels=200)
     ax.set_title('u')
     ax = plt.subplot(5, 1, 2)
     plot = self.showStress(x, y, v_model, ax, colorbar_norm=v_norm, cmap='jet', levels=200)
     ax.set_title('v')
     ax = plt.subplot(5, 1, 3)
     plot = self.showStress(x, y, sx_model, ax, colorbar_norm=sx_norm, cmap='jet', levels=200)
     ax.set_title('sx')
     ax = plt.subplot(5, 1, 4)
     plot = self.showStress(x, y, sy_model, ax, colorbar_norm=sy_norm, cmap='jet', levels=200)
     ax.set_title('sy')
     ax = plt.subplot(5, 1, 5)
     plot = self.showStress(x, y, sxy_model, ax, colorbar_norm=sxy_norm, cmap='jet', levels=200)
     ax.set_title('sxy')

     plt.figure(figsize=(16, 3))
     ax = plt.subplot(1, 5, 1)
     plot = self.showStress(x, y, u_model, ax, colorbar_norm=u_norm, cmap='jet', levels=200)
     ax.set_title('u')
     ax = plt.subplot(1, 5, 2)
     plot = self.showStress(x, y, v_model, ax, colorbar_norm=v_norm, cmap='jet', levels=200)
     ax.set_title('v')
     ax = plt.subplot(1, 5, 3)
     plot = self.showStress(x, y, sx_model, ax, colorbar_norm=sx_norm, cmap='jet', levels=200)
     ax.set_title('sx')
     ax = plt.subplot(1, 5, 4)
     plot = self.showStress(x, y, sy_model, ax, colorbar_norm=sy_norm, cmap='jet', levels=200)
     ax.set_title('sy')
     ax = plt.subplot(1, 5, 5)
     plot = self.showStress(x, y, sxy_model, ax, colorbar_norm=sxy_norm, cmap='jet', levels=200)
     ax.set_title('sxy')


     plt.tight_layout()
     plt.show()

    def plot_result_FEM(self):
     u_model, v_model = self.pred_uv(self.labeled_xy)
     u_model = u_model.cpu().detach().numpy()
     v_model = v_model.cpu().detach().numpy()
     x = self.labeled_x.cpu().detach().numpy()
     y = self.labeled_y.cpu().detach().numpy()
     u_refer = self.labeled_u.cpu().detach().numpy()
     v_refer = self.labeled_v.cpu().detach().numpy()
     sx_model, sy_model, sxy_model = self.pred_stress(self.labeled_xy)
     sx_model = sx_model.cpu().detach().numpy()
     sy_model = sy_model.cpu().detach().numpy()
     sxy_model = sxy_model.cpu().detach().numpy()
     sx_refer = self.labeled_sx.cpu().detach().numpy()
     sy_refer = self.labeled_sy.cpu().detach().numpy()
     sxy_refer = self.labeled_sxy.cpu().detach().numpy()

     u_min_plot = np.min(u_refer)
     u_max_plot = np.max(u_refer)
     v_min_plot = np.min(v_refer)
     v_max_plot = np.max(v_refer)
     sx_min_plot = np.min(sx_refer)
     sx_max_plot = np.max(sx_refer)
     sy_min_plot = np.min(sy_refer)
     sy_max_plot = np.max(sy_refer)
     sxy_min_plot = np.min(sxy_refer)
     sxy_max_plot = np.max(sxy_refer)

     u_model = np.clip(u_model, u_min_plot, u_max_plot)
     v_model = np.clip(v_model, v_min_plot, v_max_plot)
     sx_model = np.clip(sx_model, sx_min_plot, sx_max_plot)
     sy_model = np.clip(sy_model, sy_min_plot, sy_max_plot)
     sxy_model = np.clip(sxy_model, sxy_min_plot, sxy_max_plot)
     u_norm = colors.Normalize(vmin=np.min(u_refer), vmax=np.max(u_refer))
     v_norm = colors.Normalize(vmin=np.min(v_refer), vmax=np.max(v_refer))
     sx_norm = colors.Normalize(vmin=np.min(sx_refer), vmax=np.max(sx_refer))
     sy_norm = colors.Normalize(vmin=np.min(sy_refer), vmax=np.max(sy_refer))
     sxy_norm = colors.Normalize(vmin=np.min(sxy_refer), vmax=np.max(sxy_refer))
     plt.figure(figsize=(3, 16))
     ax = plt.subplot(5, 1, 1)
     plot = self.showStress(x, y, u_refer, ax, colorbar_norm=u_norm, cmap='jet', levels=200)
     ax.set_title('u')
     ax = plt.subplot(5, 1, 2)
     plot = self.showStress(x, y, v_refer, ax, colorbar_norm=v_norm, cmap='jet', levels=200)
     ax.set_title('v')
     ax = plt.subplot(5, 1, 3)
     plot = self.showStress(x, y, sx_refer, ax, colorbar_norm=sx_norm, cmap='jet', levels=200)
     ax.set_title('sx')
     ax = plt.subplot(5, 1, 4)
     plot = self.showStress(x, y, sy_refer, ax, colorbar_norm=sy_norm, cmap='jet', levels=200)
     ax.set_title('sy')
     ax = plt.subplot(5, 1, 5)
     plot = self.showStress(x, y, sxy_refer, ax, colorbar_norm=sxy_norm, cmap='jet', levels=200)
     ax.set_title('sxy')


     # plt.figure(figsize=(16, 3))
     # ax = plt.subplot(1, 5, 1)
     # plot = self.showStress(x, y, u_refer, ax, colorbar_norm=u_norm, cmap='jet', levels=200)
     # ax.set_title('u')
     # ax = plt.subplot(1, 5, 2)
     # plot = self.showStress(x, y, v_refer, ax, colorbar_norm=v_norm, cmap='jet', levels=200)
     # ax.set_title('v')
     # ax = plt.subplot(1, 5, 3)
     # plot = self.showStress(x, y, sx_refer, ax, colorbar_norm=sx_norm, cmap='jet', levels=200)
     # ax.set_title('sx')
     # ax = plt.subplot(1, 5, 4)
     # plot = self.showStress(x, y, sy_refer, ax, colorbar_norm=sy_norm, cmap='jet', levels=200)
     # ax.set_title('sy')
     # ax = plt.subplot(1, 5, 5)
     # plot = self.showStress(x, y, sxy_refer, ax, colorbar_norm=sxy_norm, cmap='jet', levels=200)
     # ax.set_title('sxy')
     #
     # plt.tight_layout()
     plt.show()
     import os
     def to_col(x):
         return x.reshape(-1, 1) if x.ndim == 1 else x

     output_dir = r'C:\Users\15844\Desktop'  # 你的目标目录
     columns = ['x', 'y', 'u', 'v', 'sigmaxx', 'sigmayy', 'sigmaxy']
     # data_model = np.hstack([x, y, u_model, v_model, sx_model, sy_model, sxy_model])
     # data_refer = np.hstack([x, y, u_refer, v_refer, sx_refer, sy_refer, sxy_refer])
     data_model = np.hstack([
         to_col(x),
         to_col(y),
         to_col(u_model),
         to_col(v_model),
         to_col(sx_model),
         to_col(sy_model),
         to_col(sxy_model)
     ])

     data_refer = np.hstack([
         to_col(x),
         to_col(y),
         to_col(u_refer),
         to_col(v_refer),
         to_col(sx_refer),
         to_col(sy_refer),
         to_col(sxy_refer)
     ])
     df_model = pd.DataFrame(data_model, columns=columns)
     df_refer = pd.DataFrame(data_refer, columns=columns)

     df_model.to_csv(os.path.join(output_dir, 'model_result.csv'), index=False, sep=',', encoding='utf-8-sig')
     df_refer.to_csv(os.path.join(output_dir, 'refer_result.csv'), index=False, sep=',', encoding='utf-8-sig')
     abs_error_data = np.hstack([
         to_col(x), to_col(y),
         np.abs(to_col(u_model - u_refer)),
         np.abs(to_col(v_model - v_refer)),
         np.abs(to_col(sx_model - sx_refer)),
         np.abs(to_col(sy_model - sy_refer)),
         np.abs(to_col(sxy_model - sxy_refer))
     ])
     df_abs_error = pd.DataFrame(abs_error_data, columns=columns)
     df_abs_error.to_csv(os.path.join(output_dir, 'abs_error_result.csv'), index=False, sep=',', encoding='utf-8-sig')
     print(f"✅ 绝对误差结果已保存至: {os.path.join(output_dir, 'abs_error_result.csv')}")
     print(f"模型预测结果已保存到: {os.path.join(output_dir, 'model_result.csv')}")
     print(f"FEM参考结果已保存到: {os.path.join(output_dir, 'refer_result.csv')}")


    def plot_abs_error(self):
     u_model, v_model = self.pred_uv(self.labeled_xy)
     u_model = u_model.cpu().detach().numpy()
     v_model = v_model.cpu().detach().numpy()
     x = self.labeled_x.cpu().detach().numpy()
     y = self.labeled_y.cpu().detach().numpy()
     u_refer = self.labeled_u.cpu().detach().numpy()
     v_refer = self.labeled_v.cpu().detach().numpy()
     sx_model, sy_model, sxy_model = self.pred_stress(self.labeled_xy)
     sx_model = sx_model.cpu().detach().numpy()
     sy_model = sy_model.cpu().detach().numpy()
     sxy_model = sxy_model.cpu().detach().numpy()
     sx_refer = self.labeled_sx.cpu().detach().numpy()
     sy_refer = self.labeled_sy.cpu().detach().numpy()
     sxy_refer = self.labeled_sxy.cpu().detach().numpy()

     u_min_plot = np.min(u_refer)
     u_max_plot = np.max(u_refer)
     v_min_plot = np.min(v_refer)
     v_max_plot = np.max(v_refer)
     sx_min_plot = np.min(sx_refer)
     sx_max_plot = np.max(sx_refer)
     sy_min_plot = np.min(sy_refer)
     sy_max_plot = np.max(sy_refer)
     sxy_min_plot = np.min(sxy_refer)
     sxy_max_plot = np.max(sxy_refer)

     u_model = np.clip(u_model, u_min_plot, u_max_plot)
     v_model = np.clip(v_model, v_min_plot, v_max_plot)
     sx_model = np.clip(sx_model, sx_min_plot, sx_max_plot)
     sy_model = np.clip(sy_model, sy_min_plot, sy_max_plot)
     sxy_model = np.clip(sxy_model, sxy_min_plot, sxy_max_plot)
     u_norm = colors.Normalize(vmin=np.min(u_refer), vmax=np.max(u_refer))
     v_norm = colors.Normalize(vmin=np.min(v_refer), vmax=np.max(v_refer))
     sx_norm = colors.Normalize(vmin=np.min(sx_refer), vmax=np.max(sx_refer))
     sy_norm = colors.Normalize(vmin=np.min(sy_refer), vmax=np.max(sy_refer))
     sxy_norm = colors.Normalize(vmin=np.min(sxy_refer), vmax=np.max(sxy_refer))
     u_norm_abs_error = np.abs((u_model - u_refer) / np.max(u_refer))
     u_relative_error = np.abs(u_model - u_refer) / (np.abs(u_refer) + 1e-3 * np.max(np.abs(u_refer)))

     # 计算 v 的绝对误差和相对误差
     v_norm_abs_error = np.abs((v_model - v_refer) / np.max(v_refer))
     v_relative_error = np.abs(v_model - v_refer) / (np.abs(v_refer) + 1e-3 * np.max(np.abs(v_refer)))

     # 计算 sx 的绝对误差和相对误差
     sx_norm_abs_error = np.abs((sx_model - sx_refer) / np.max(sx_refer))
     sx_relative_error = np.abs(sx_model - sx_refer) / (np.abs(sx_refer) + 1e-3 * np.max(np.abs(sx_refer)))

     # 计算 sy 的绝对误差和相对误差
     sy_norm_abs_error = np.abs((sy_model - sy_refer) / np.max(sy_refer))
     sy_relative_error = np.abs(sy_model - sy_refer) / (np.abs(sy_refer) + 1e-3 * np.max(np.abs(sy_refer)))

     # 计算 sxy 的绝对误差和相对误差
     sxy_norm_abs_error = np.abs((sxy_model - sxy_refer) / np.max(sxy_refer))
     sxy_relative_error = np.abs(sxy_model - sxy_refer) / (np.abs(sxy_refer) + 1e-3 * np.max(np.abs(sxy_refer)))

     u_max = True if np.max(u_norm_abs_error) > 0.5 else False
     v_max = True if np.max(v_norm_abs_error) > 0.5 else False
     sx_max = True if np.max(sx_norm_abs_error) > 0.5 else False
     sy_max = True if np.max(sy_norm_abs_error) > 0.5 else False
     sxy_max = True if np.max(sxy_norm_abs_error) > 0.5 else False

     plt.figure(figsize=(3, 16))
     ax = plt.subplot(5, 1, 1)
     plot = self.showStress(x, y, u_norm_abs_error, ax, u_max, levels=200)
     ax.set_title('u')
     ax = plt.subplot(5, 1, 2)
     plot = self.showStress(x, y, v_norm_abs_error, ax, v_max, levels=200)
     ax.set_title('v')
     ax = plt.subplot(5, 1, 3)
     plot = self.showStress(x, y, sx_norm_abs_error, ax, sx_max, levels=200)
     ax.set_title('sx')
     ax = plt.subplot(5, 1, 4)
     plot = self.showStress(x, y, sy_norm_abs_error, ax, sy_max, levels=200)
     ax.set_title('sy')
     ax = plt.subplot(5, 1, 5)
     plot = self.showStress(x, y, sxy_norm_abs_error, ax, sxy_max, levels=200)
     ax.set_title('sxy')

     plt.figure(figsize=(16, 3))
     ax = plt.subplot(1, 5, 1)
     plot = self.showStress(x, y, u_norm_abs_error, ax, u_max, levels=200)
     ax.set_title('u')
     ax = plt.subplot(1, 5, 2)
     plot = self.showStress(x, y, v_norm_abs_error, ax, v_max, levels=200)
     ax.set_title('v')
     ax = plt.subplot(1, 5, 3)
     plot = self.showStress(x, y, sx_norm_abs_error, ax, sx_max, levels=200)
     ax.set_title('sx')
     ax = plt.subplot(1, 5, 4)
     plot = self.showStress(x, y, sy_norm_abs_error, ax, sy_max, levels=200)
     ax.set_title('sy')
     ax = plt.subplot(1, 5, 5)
     plot = self.showStress(x, y, sxy_norm_abs_error, ax, sxy_max, levels=200)
     ax.set_title('sxy')

     plt.tight_layout()
     plt.show()

    def plot_relative_error(self):
     u_model, v_model = self.pred_uv(self.labeled_xy)
     u_model = u_model.cpu().detach().numpy()
     v_model = v_model.cpu().detach().numpy()
     x = self.labeled_x.cpu().detach().numpy()
     y = self.labeled_y.cpu().detach().numpy()
     u_refer = self.labeled_u.cpu().detach().numpy()
     v_refer = self.labeled_v.cpu().detach().numpy()
     sx_model, sy_model, sxy_model = self.pred_stress(self.labeled_xy)
     sx_model = sx_model.cpu().detach().numpy()
     sy_model = sy_model.cpu().detach().numpy()
     sxy_model = sxy_model.cpu().detach().numpy()
     sx_refer = self.labeled_sx.cpu().detach().numpy()
     sy_refer = self.labeled_sy.cpu().detach().numpy()
     sxy_refer = self.labeled_sxy.cpu().detach().numpy()

     u_min_plot = np.min(u_refer)
     u_max_plot = np.max(u_refer)
     v_min_plot = np.min(v_refer)
     v_max_plot = np.max(v_refer)
     sx_min_plot = np.min(sx_refer)
     sx_max_plot = np.max(sx_refer)
     sy_min_plot = np.min(sy_refer)
     sy_max_plot = np.max(sy_refer)
     sxy_min_plot = np.min(sxy_refer)
     sxy_max_plot = np.max(sxy_refer)

     u_model = np.clip(u_model, u_min_plot, u_max_plot)
     v_model = np.clip(v_model, v_min_plot, v_max_plot)
     sx_model = np.clip(sx_model, sx_min_plot, sx_max_plot)
     sy_model = np.clip(sy_model, sy_min_plot, sy_max_plot)
     sxy_model = np.clip(sxy_model, sxy_min_plot, sxy_max_plot)
     u_norm = colors.Normalize(vmin=np.min(u_refer), vmax=np.max(u_refer))
     v_norm = colors.Normalize(vmin=np.min(v_refer), vmax=np.max(v_refer))
     sx_norm = colors.Normalize(vmin=np.min(sx_refer), vmax=np.max(sx_refer))
     sy_norm = colors.Normalize(vmin=np.min(sy_refer), vmax=np.max(sy_refer))
     sxy_norm = colors.Normalize(vmin=np.min(sxy_refer), vmax=np.max(sxy_refer))
     u_norm_abs_error = np.abs((u_model - u_refer) / np.max(u_refer))
     u_relative_error = np.abs(u_model - u_refer) / (np.abs(u_refer) + 1e-3 * np.max(np.abs(u_refer)))

     # 计算 v 的绝对误差和相对误差
     v_norm_abs_error = np.abs((v_model - v_refer) / np.max(v_refer))
     v_relative_error = np.abs(v_model - v_refer) / (np.abs(v_refer) + 1e-3 * np.max(np.abs(v_refer)))

     # 计算 sx 的绝对误差和相对误差
     sx_norm_abs_error = np.abs((sx_model - sx_refer) / np.max(sx_refer))
     sx_relative_error = np.abs(sx_model - sx_refer) / (np.abs(sx_refer) + 1e-3 * np.max(np.abs(sx_refer)))

     # 计算 sy 的绝对误差和相对误差
     sy_norm_abs_error = np.abs((sy_model - sy_refer) / np.max(sy_refer))
     sy_relative_error = np.abs(sy_model - sy_refer) / (np.abs(sy_refer) + 1e-3 * np.max(np.abs(sy_refer)))

     # 计算 sxy 的绝对误差和相对误差
     sxy_norm_abs_error = np.abs((sxy_model - sxy_refer) / np.max(sxy_refer))
     sxy_relative_error = np.abs(sxy_model - sxy_refer) / (np.abs(sxy_refer) + 1e-3 * np.max(np.abs(sxy_refer)))

     u_max = True if np.max(u_relative_error) > 0.5 else False
     v_max = True if np.max(v_relative_error) > 0.5 else False
     sx_max = True if np.max(sx_relative_error) > 0.5 else False
     sy_max = True if np.max(sy_relative_error) > 0.5 else False
     sxy_max = True if np.max(sxy_relative_error) > 0.5 else False

     plt.figure(figsize=(3, 16))
     ax = plt.subplot(5, 1, 1)
     plot = self.showStress(x, y, u_relative_error, ax, u_max, levels=200)
     ax.set_title('u')
     ax = plt.subplot(5, 1, 2)
     plot = self.showStress(x, y, v_relative_error, ax, v_max, levels=200)
     ax.set_title('v')
     ax = plt.subplot(5, 1, 3)
     plot = self.showStress(x, y, sx_relative_error, ax, sx_max, levels=200)
     ax.set_title('sx')
     ax = plt.subplot(5, 1, 4)
     plot = self.showStress(x, y, sy_relative_error, ax, sy_max, levels=200)
     ax.set_title('sy')
     ax = plt.subplot(5, 1, 5)
     plot = self.showStress(x, y, sxy_relative_error, ax, sxy_max, levels=200)
     ax.set_title('sxy')
     plt.tight_layout()
     plt.show()
#模型输出位移是mm
#输出应力单位MPa
class DEM2D_2(PINN2D):
    def __init__(self, model: nn.Module):
        super().__init__(model)  
    
    def pred_stress(self,xy):
        u,v = self.pred_uv(xy) 
        eXX,eYY,eXY = self.compute_Strain(u,v,xy)
        sx , sy , sxy = self.constitutive(eXX,eYY,eXY)   
        return sx,sy,sxy    

    def infer(self,xy):
        u,v = self.pred_uv(xy) 
        eXX,eYY,eXY = self.compute_Strain(u,v,xy)
        sx , sy , sxy = self.constitutive(eXX,eYY,eXY)   
        return u,v,sx,sy,sxy





class DEM_bimaterial(PINN2D):

    def set_LevelSet(self,line:Geometry1D):
        self.material_interface = line

    def setMaterial_Bi(self, E1, E2, nu1 = 0.3, nu2 = 0.3 ,type='plane stress'):
        '''1为线段左边,2为右边'''
        self.E = np.array([E1,E2])
        self.nu = np.array([nu1,nu2])
        self.setD(self.E , self.nu,type=type)
    def get_material(self,xy):
        # ls = torch.from_numpy(self.Line.is_on_left(xy.cpu().detach().numpy())).to(self.device)
        # ls = self.Line.is_on_left(xy)
        ls  = self.material_interface.levelset(xy[...,0],xy[...,1])
        ind = torch.where(ls < 0 , 0, 1)
        d11 = self.d11[ind]
        d12 = self.d12[ind]
        G   = self.G[ind]
        # d11 =  torch.where(ls , self.d11[0] ,self.d11[1])
        # d12 = torch.where(ls , self.d12[0] ,self.d12[1])
        # G = torch.where(ls , self.G[0] ,self.G[1])
        return d11 , d12 , G    
        # return self.d11[0],self.d12[0],self.G[0]
    
    def pred_stress(self, xy):
        u,v= self.pred_uv(xy)
        eXX,eYY,eXY = self.compute_Strain(u,v,xy)
        d11 , d12 , G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY , d11 , d12 , G)
        return sx, sy, sxy
    
    def infer(self, xy):
        u,v = self.pred_uv(xy)
        eXX,eYY,eXY = self.compute_Strain(u,v,xy)
        d11 , d12 , G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY , d11 , d12 , G)
        return u,v,sx, sy, sxy
    
    def constitutive(self, eXX, eYY, eXY , d11 , d12 , G):
        sx = d11 * eXX + d12 * eYY
        sy = d12 * eXX + d11 * eYY
        sxy = G * eXY
        return sx , sy , sxy

    def set_material_domain(self,domains:list[Domain2D]):
        self.domains = domains
        self.setMaterial_Bi(E1 = domains[0].E,nu1=domains[0].mu,
                         E2 = domains[1].E,nu2=domains[1].mu,)
    
    def gen_inner_points(self, num):
        '''保证材料不会判断错'''
        # num = int(num/len(self.domains))
        points_list = []
        pdf_list = []
        d11_list = []
        d12_list = []
        G_list = []
        for domain in self.domains:
            points,pdf = domain.gen_uniform_points(num)
            one = torch.ones_like(points[...,0])
            d11_list.append(one * domain.d11)
            d12_list.append(one * domain.d12)
            G_list.append(one * domain.G)
            points_list.append(points)
            pdf_list.append(pdf * domain.polygon.area)
        points = torch.cat(points_list)
        pdf = np.concatenate(pdf_list)
        self.set_inner_points(points,pdf)
        self.inner_d11 = torch.cat(d11_list).to(self.device)
        self.inner_d12 = torch.cat(d12_list).to(self.device)
        self.inner_G = torch.cat(d11_list).to(self.device)
    
    def set_inner_materials(self):
        d11 , d12 , G = self.get_material(self.XY)
        self.inner_d11 = d11
        self.inner_d12 = d12
        self.inner_G = G

    def train_resampling(self,num,epochs = 100000, patience=100,path = 'test'):

        self.EarlyStopping=EarlyStopping(patience=patience,verbose=True,path= path+'.pth')
        self.path = path
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr= 0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2000,10000,30000], gamma = 0.2)
        for i in range(epochs):
            if i % 10 == 0:
                self.gen_inner_points(num)                   
            self.train_step()
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break
        self.save(path+'_final')

    def get_energy_density(self,xy):
            u,v = self.pred_uv(xy)
            eXX,eYY,eXY = self.compute_Strain(u,v,xy)
            # sx,sy,sxy = self.constitutive(eXX,eYY,eXY,
            #                             self.inner_d11 ,self.inner_d12 ,self.inner_G)
            d11,d12,G = self.get_material(xy)
            sx,sy,sxy = self.constitutive(eXX,eYY,eXY,d11,d12,G)
            energy = 0.5 * (eXX * sx + eYY * sy + eXY * sxy)
            return energy

    def set_meshgrid_inner_points(self,xstart,xend,xnum,ystart,yend,ynum):
        '''生成规则网格排布点'''
        super().set_meshgrid_inner_points(xstart,xend,xnum,ystart,yend,ynum)
        self.set_inner_materials()

    def set_meshgrid_trapz_Tip_Dense(self, xstart, xend, ystart, yend, xTip, yTip, x_dense_num, y_dense_num, x_outer_num, y_outer_num, x_inteval=0.2,y_inteval=0.2):
        super().set_meshgrid_trapz_Tip_Dense(xstart, xend, ystart, yend, xTip, yTip, x_dense_num, y_dense_num, x_outer_num, y_outer_num, x_inteval , y_inteval)
        self.set_inner_materials()


class DEM_bimaterial_muti(PINN2D):

    def set_LevelSet(self, lines:list[Geometry1D]):
        self.material_surfaces = lines
    # def set_LevelSet(self, line1: Geometry1D,line2: Geometry1D,line3: Geometry1D):
    #     self.material_interface1 = line1
    #     self.material_interface2 = line2
    #     self.material_interface3 = line3

    def setMaterial_Bi(self, E1, E2, nu1=0.3, nu2=0.3, type='plane stress'):
        '''1为线段左边,2为右边'''
        self.E = np.array([E1, E2])
        self.nu = np.array([nu1, nu2])
        self.setD(self.E, self.nu, type=type)

    def get_material(self, xy):
        # ls = torch.from_numpy(self.Line.is_on_left(xy.cpu().detach().numpy())).to(self.device)
        # ls = self.Line.is_on_left(xy)
        ls = [line.levelset(xy[..., 0], xy[..., 1]) for line in self.material_surfaces]
        ind = torch.where(torch.all(torch.stack(ls) < 0, dim=0), 0, 1)
        ind = torch.where(torch.any(torch.stack(ls) < -1e-4, dim=0), 0, 1)
        # ls1 = self.material_interface1.levelset(xy[..., 0], xy[..., 1])
        # ls2 = self.material_interface2.levelset(xy[..., 0], xy[..., 1])
        # ls3 = self.material_interface3.levelset(xy[..., 0], xy[..., 1])
        # ind = torch.where((ls1 > 0) & (ls2 > 0) & (ls3 > 0), 1, 0)
        d11 = self.d11[ind]
        d12 = self.d12[ind]
        G = self.G[ind]
        # d11 =  torch.where(ls , self.d11[0] ,self.d11[1])
        # d12 = torch.where(ls , self.d12[0] ,self.d12[1])
        # G = torch.where(ls , self.G[0] ,self.G[1])
        return d11, d12, G
        # return self.d11[0],self.d12[0],self.G[0]

    def pred_stress(self, xy):
        u, v= self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        return sx, sy, sxy

    def infer(self, xy):
        u, v= self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        return u, v, sx, sy, sxy

    def constitutive(self, eXX, eYY, eXY, d11, d12, G):
        sx = d11 * eXX + d12 * eYY
        sy = d12 * eXX + d11 * eYY
        sxy = G * eXY
        return sx, sy, sxy

    def set_material_domain(self, domains: list[Domain2D]):
        self.domains = domains
        self.setMaterial_Bi(E1=domains[0].E, nu1=domains[0].mu,
                            E2=domains[1].E, nu2=domains[1].mu, )

    def gen_inner_points(self, num):
        '''保证材料不会判断错'''
        # num = int(num/len(self.domains))
        points_list = []
        pdf_list = []
        d11_list = []
        d12_list = []
        G_list = []
        for domain in self.domains:
            points, pdf = domain.gen_uniform_points(num)
            one = torch.ones_like(points[..., 0])
            d11_list.append(one * domain.d11)
            d12_list.append(one * domain.d12)
            G_list.append(one * domain.G)
            points_list.append(points)
            pdf_list.append(pdf * domain.polygon.area)
        points = torch.cat(points_list)
        pdf = np.concatenate(pdf_list)
        self.set_inner_points(points, pdf)
        self.inner_d11 = torch.cat(d11_list).to(self.device)
        self.inner_d12 = torch.cat(d12_list).to(self.device)
        self.inner_G = torch.cat(d11_list).to(self.device)

    def set_inner_materials(self):
        d11, d12, G = self.get_material(self.XY)
        self.inner_d11 = d11
        self.inner_d12 = d12
        self.inner_G = G

    def train_resampling(self, num, epochs=100000, patience=100, path='test'):

        self.EarlyStopping = EarlyStopping(patience=patience, verbose=True, path=path + '.pth')
        self.path = path
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2000, 10000, 30000], gamma=0.2)
        for i in range(epochs):
            if i % 10 == 0:
                self.gen_inner_points(num)
            self.train_step()
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break
        self.save(path + '_final')

    def get_energy_density(self, xy):
        u, v = self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        # sx,sy,sxy = self.constitutive(eXX,eYY,eXY,
        #                             self.inner_d11 ,self.inner_d12 ,self.inner_G)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        energy = 0.5 * (eXX * sx + eYY * sy + eXY * sxy)
        return energy

    def set_meshgrid_inner_points(self, xstart, xend, xnum, ystart, yend, ynum):
        '''生成规则网格排布点'''
        super().set_meshgrid_inner_points(xstart, xend, xnum, ystart, yend, ynum)
        self.set_inner_materials()

    def set_meshgrid_trapz_Tip_Dense(self, xstart, xend, ystart, yend, xTip, yTip, x_dense_num, y_dense_num,
                                     x_outer_num, y_outer_num, x_inteval=0.2, y_inteval=0.2):
        super().set_meshgrid_trapz_Tip_Dense(xstart, xend, ystart, yend, xTip, yTip, x_dense_num, y_dense_num,
                                             x_outer_num, y_outer_num, x_inteval, y_inteval)
        self.set_inner_materials()

class DEM_bimaterial_muti_four(PINN2D):

    def set_LevelSet(self, line1: Geometry1D,line2: Geometry1D,line3: Geometry1D,line4: Geometry1D):
        self.material_interface1 = line1
        self.material_interface2 = line2
        self.material_interface3 = line3
        self.material_interface4 = line4
    def setMaterial_Bi(self, E1, E2, nu1=0.3, nu2=0.3, type='plane stress'):
        '''1为线段左边,2为右边'''
        self.E = np.array([E1, E2])
        self.nu = np.array([nu1, nu2])
        self.setD(self.E, self.nu, type=type)

    def get_material(self, xy):
        # ls = torch.from_numpy(self.Line.is_on_left(xy.cpu().detach().numpy())).to(self.device)
        # ls = self.Line.is_on_left(xy)
        ls1 = self.material_interface1.levelset(xy[..., 0], xy[..., 1])
        ls2 = self.material_interface2.levelset(xy[..., 0], xy[..., 1])
        ls3 = self.material_interface3.levelset(xy[..., 0], xy[..., 1])
        ls4 = self.material_interface4.levelset(xy[..., 0], xy[..., 1])
        ind = torch.where((ls1 > 0) & (ls2 > 0) & (ls3 > 0) & (ls4 > 0), 1, 0)
        d11 = self.d11[ind]
        d12 = self.d12[ind]
        G = self.G[ind]
        # d11 =  torch.where(ls , self.d11[0] ,self.d11[1])
        # d12 = torch.where(ls , self.d12[0] ,self.d12[1])
        # G = torch.where(ls , self.G[0] ,self.G[1])
        return d11, d12, G
        # return self.d11[0],self.d12[0],self.G[0]

    def pred_stress(self, xy):
        u, v= self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        return sx, sy, sxy

    def infer(self, xy):
        u, v= self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        return u, v, sx, sy, sxy

    def constitutive(self, eXX, eYY, eXY, d11, d12, G):
        sx = d11 * eXX + d12 * eYY
        sy = d12 * eXX + d11 * eYY
        sxy = G * eXY
        return sx, sy, sxy

    def set_material_domain(self, domains: list[Domain2D]):
        self.domains = domains
        self.setMaterial_Bi(E1=domains[0].E, nu1=domains[0].mu,
                            E2=domains[1].E, nu2=domains[1].mu, )

    def gen_inner_points(self, num):
        '''保证材料不会判断错'''
        # num = int(num/len(self.domains))
        points_list = []
        pdf_list = []
        d11_list = []
        d12_list = []
        G_list = []
        for domain in self.domains:
            points, pdf = domain.gen_uniform_points(num)
            one = torch.ones_like(points[..., 0])
            d11_list.append(one * domain.d11)
            d12_list.append(one * domain.d12)
            G_list.append(one * domain.G)
            points_list.append(points)
            pdf_list.append(pdf * domain.polygon.area)
        points = torch.cat(points_list)
        pdf = np.concatenate(pdf_list)
        self.set_inner_points(points, pdf)
        self.inner_d11 = torch.cat(d11_list).to(self.device)
        self.inner_d12 = torch.cat(d12_list).to(self.device)
        self.inner_G = torch.cat(d11_list).to(self.device)

    def set_inner_materials(self):
        d11, d12, G = self.get_material(self.XY)
        self.inner_d11 = d11
        self.inner_d12 = d12
        self.inner_G = G

    def train_resampling(self, num, epochs=100000, patience=100, path='test'):

        self.EarlyStopping = EarlyStopping(patience=patience, verbose=True, path=path + '.pth')
        self.path = path
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2000, 10000, 30000], gamma=0.2)
        for i in range(epochs):
            if i % 10 == 0:
                self.gen_inner_points(num)
            self.train_step()
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break
        self.save(path + '_final')

    def get_energy_density(self, xy):
        u, v = self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        # sx,sy,sxy = self.constitutive(eXX,eYY,eXY,
        #                             self.inner_d11 ,self.inner_d12 ,self.inner_G)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        energy = 0.5 * (eXX * sx + eYY * sy + eXY * sxy)
        return energy

    def set_meshgrid_inner_points(self, xstart, xend, xnum, ystart, yend, ynum):
        '''生成规则网格排布点'''
        super().set_meshgrid_inner_points(xstart, xend, xnum, ystart, yend, ynum)
        self.set_inner_materials()

    def set_meshgrid_trapz_Tip_Dense(self, xstart, xend, ystart, yend, xTip, yTip, x_dense_num, y_dense_num,
                                     x_outer_num, y_outer_num, x_inteval=0.2, y_inteval=0.2):
        super().set_meshgrid_trapz_Tip_Dense(xstart, xend, ystart, yend, xTip, yTip, x_dense_num, y_dense_num,
                                             x_outer_num, y_outer_num, x_inteval, y_inteval)
        self.set_inner_materials()

    
class DEM_bimaterial_muti_nine(PINN2D):

    def set_LevelSet(self, line1: Geometry1D,line2: Geometry1D,line3: Geometry1D,line4: Geometry1D,line5: Geometry1D,line6: Geometry1D,line7: Geometry1D,line8: Geometry1D,line9: Geometry1D):
        self.material_interface1 = line1
        self.material_interface2 = line2
        self.material_interface3 = line3
        self.material_interface4 = line4
        self.material_interface5 = line5
        self.material_interface6 = line6
        self.material_interface7 = line7
        self.material_interface8 = line8
        self.material_interface9 = line9

    def setMaterial_Bi(self, E1, E2, nu1=0.3, nu2=0.3, type='plane stress'):
        '''1为线段左边,2为右边'''
        self.E = np.array([E1, E2])
        self.nu = np.array([nu1, nu2])
        self.setD(self.E, self.nu, type=type)

    def get_material(self, xy):
        # ls = torch.from_numpy(self.Line.is_on_left(xy.cpu().detach().numpy())).to(self.device)
        # ls = self.Line.is_on_left(xy)
        ls1 = self.material_interface1.levelset(xy[..., 0], xy[..., 1])
        ls2 = self.material_interface2.levelset(xy[..., 0], xy[..., 1])
        ls3 = self.material_interface3.levelset(xy[..., 0], xy[..., 1])
        ls4 = self.material_interface4.levelset(xy[..., 0], xy[..., 1])
        ls5 = self.material_interface5.levelset(xy[..., 0], xy[..., 1])
        ls6 = self.material_interface6.levelset(xy[..., 0], xy[..., 1])
        ls7 = self.material_interface7.levelset(xy[..., 0], xy[..., 1])
        ls8 = self.material_interface8.levelset(xy[..., 0], xy[..., 1])
        ls9 = self.material_interface9.levelset(xy[..., 0], xy[..., 1])

        ind = torch.where((ls1 > 0) & (ls2 > 0) & (ls3 > 0) & (ls4 > 0)& (ls5 > 0)& (ls6 > 0)& (ls7 > 0)& (ls8 > 0)& (ls9 > 0), 1, 0)
        d11 = self.d11[ind]
        d12 = self.d12[ind]
        G = self.G[ind]
        # d11 =  torch.where(ls , self.d11[0] ,self.d11[1])
        # d12 = torch.where(ls , self.d12[0] ,self.d12[1])
        # G = torch.where(ls , self.G[0] ,self.G[1])
        return d11, d12, G
        # return self.d11[0],self.d12[0],self.G[0]

    def pred_stress(self, xy):
        u, v= self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        return sx, sy, sxy

    def infer(self, xy):
        u, v= self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        return u, v, sx, sy, sxy

    def constitutive(self, eXX, eYY, eXY, d11, d12, G):
        sx = d11 * eXX + d12 * eYY
        sy = d12 * eXX + d11 * eYY
        sxy = G * eXY
        return sx, sy, sxy

    def set_material_domain(self, domains: list[Domain2D]):
        self.domains = domains
        self.setMaterial_Bi(E1=domains[0].E, nu1=domains[0].mu,
                            E2=domains[1].E, nu2=domains[1].mu, )

    def gen_inner_points(self, num):
        '''保证材料不会判断错'''
        # num = int(num/len(self.domains))
        points_list = []
        pdf_list = []
        d11_list = []
        d12_list = []
        G_list = []
        for domain in self.domains:
            points, pdf = domain.gen_uniform_points(num)
            one = torch.ones_like(points[..., 0])
            d11_list.append(one * domain.d11)
            d12_list.append(one * domain.d12)
            G_list.append(one * domain.G)
            points_list.append(points)
            pdf_list.append(pdf * domain.polygon.area)
        points = torch.cat(points_list)
        pdf = np.concatenate(pdf_list)
        self.set_inner_points(points, pdf)
        self.inner_d11 = torch.cat(d11_list).to(self.device)
        self.inner_d12 = torch.cat(d12_list).to(self.device)
        self.inner_G = torch.cat(d11_list).to(self.device)

    def set_inner_materials(self):
        d11, d12, G = self.get_material(self.XY)
        self.inner_d11 = d11
        self.inner_d12 = d12
        self.inner_G = G

    def train_resampling(self, num, epochs=100000, patience=100, path='test'):

        self.EarlyStopping = EarlyStopping(patience=patience, verbose=True, path=path + '.pth')
        self.path = path
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2000, 10000, 30000], gamma=0.2)
        for i in range(epochs):
            if i % 10 == 0:
                self.gen_inner_points(num)
            self.train_step()
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break
        self.save(path + '_final')

    def get_energy_density(self, xy):
        u, v = self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        # sx,sy,sxy = self.constitutive(eXX,eYY,eXY,
        #                             self.inner_d11 ,self.inner_d12 ,self.inner_G)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        energy = 0.5 * (eXX * sx + eYY * sy + eXY * sxy)
        return energy

    def set_meshgrid_inner_points(self, xstart, xend, xnum, ystart, yend, ynum):
        '''生成规则网格排布点'''
        super().set_meshgrid_inner_points(xstart, xend, xnum, ystart, yend, ynum)
        self.set_inner_materials()

    def set_meshgrid_trapz_Tip_Dense(self, xstart, xend, ystart, yend, xTip, yTip, x_dense_num, y_dense_num,
                                     x_outer_num, y_outer_num, x_inteval=0.2, y_inteval=0.2):
        super().set_meshgrid_trapz_Tip_Dense(xstart, xend, ystart, yend, xTip, yTip, x_dense_num, y_dense_num,
                                             x_outer_num, y_outer_num, x_inteval, y_inteval)
        self.set_inner_materials()

class DEM_bimaterial_muti_six(PINN2D):

    def set_LevelSet(self, line1: Geometry1D,line2: Geometry1D,line3: Geometry1D,line4: Geometry1D,line5: Geometry1D,line6: Geometry1D):
        self.material_interface1 = line1
        self.material_interface2 = line2
        self.material_interface3 = line3
        self.material_interface4 = line4
        self.material_interface5 = line5
        self.material_interface6 = line6

    def setMaterial_Bi(self, E1, E2, nu1=0.3, nu2=0.3, type='plane stress'):
        '''1为线段左边,2为右边'''
        self.E = np.array([E1, E2])
        self.nu = np.array([nu1, nu2])
        self.setD(self.E, self.nu, type=type)

    def get_material(self, xy):
        # ls = torch.from_numpy(self.Line.is_on_left(xy.cpu().detach().numpy())).to(self.device)
        # ls = self.Line.is_on_left(xy)
        ls1 = self.material_interface1.levelset(xy[..., 0], xy[..., 1])
        ls2 = self.material_interface2.levelset(xy[..., 0], xy[..., 1])
        ls3 = self.material_interface3.levelset(xy[..., 0], xy[..., 1])
        ls4 = self.material_interface4.levelset(xy[..., 0], xy[..., 1])
        ls5 = self.material_interface5.levelset(xy[..., 0], xy[..., 1])
        ls6 = self.material_interface6.levelset(xy[..., 0], xy[..., 1])

        ind = torch.where((ls1 > 0) & (ls2 > 0) & (ls3 > 0) & (ls4 > 0)& (ls5 > 0)& (ls6 > 0), 1, 0)
        d11 = self.d11[ind]
        d12 = self.d12[ind]
        G = self.G[ind]
        # d11 =  torch.where(ls , self.d11[0] ,self.d11[1])
        # d12 = torch.where(ls , self.d12[0] ,self.d12[1])
        # G = torch.where(ls , self.G[0] ,self.G[1])
        return d11, d12, G
        # return self.d11[0],self.d12[0],self.G[0]

    def pred_stress(self, xy):
        u, v= self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        return sx, sy, sxy

    def infer(self, xy):
        u, v= self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        return u, v, sx, sy, sxy

    def constitutive(self, eXX, eYY, eXY, d11, d12, G):
        sx = d11 * eXX + d12 * eYY
        sy = d12 * eXX + d11 * eYY
        sxy = G * eXY
        return sx, sy, sxy

    def set_material_domain(self, domains: list[Domain2D]):
        self.domains = domains
        self.setMaterial_Bi(E1=domains[0].E, nu1=domains[0].mu,
                            E2=domains[1].E, nu2=domains[1].mu, )

    def gen_inner_points(self, num):
        '''保证材料不会判断错'''
        # num = int(num/len(self.domains))
        points_list = []
        pdf_list = []
        d11_list = []
        d12_list = []
        G_list = []
        for domain in self.domains:
            points, pdf = domain.gen_uniform_points(num)
            one = torch.ones_like(points[..., 0])
            d11_list.append(one * domain.d11)
            d12_list.append(one * domain.d12)
            G_list.append(one * domain.G)
            points_list.append(points)
            pdf_list.append(pdf * domain.polygon.area)
        points = torch.cat(points_list)
        pdf = np.concatenate(pdf_list)
        self.set_inner_points(points, pdf)
        self.inner_d11 = torch.cat(d11_list).to(self.device)
        self.inner_d12 = torch.cat(d12_list).to(self.device)
        self.inner_G = torch.cat(d11_list).to(self.device)

    def set_inner_materials(self):
        d11, d12, G = self.get_material(self.XY)
        self.inner_d11 = d11
        self.inner_d12 = d12
        self.inner_G = G

    def train_resampling(self, num, epochs=100000, patience=100, path='test'):

        self.EarlyStopping = EarlyStopping(patience=patience, verbose=True, path=path + '.pth')
        self.path = path
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2000, 10000, 30000], gamma=0.2)
        for i in range(epochs):
            if i % 10 == 0:
                self.gen_inner_points(num)
            self.train_step()
            scheduler.step()
            if (self.EarlyStopping.early_stop):
                break
        self.save(path + '_final')

    def get_energy_density(self, xy):
        u, v = self.pred_uv(xy)
        eXX, eYY, eXY = self.compute_Strain(u, v, xy)
        # sx,sy,sxy = self.constitutive(eXX,eYY,eXY,
        #                             self.inner_d11 ,self.inner_d12 ,self.inner_G)
        d11, d12, G = self.get_material(xy)
        sx, sy, sxy = self.constitutive(eXX, eYY, eXY, d11, d12, G)
        energy = 0.5 * (eXX * sx + eYY * sy + eXY * sxy)
        return energy

    def set_meshgrid_inner_points(self, xstart, xend, xnum, ystart, yend, ynum):
        '''生成规则网格排布点'''
        super().set_meshgrid_inner_points(xstart, xend, xnum, ystart, yend, ynum)
        self.set_inner_materials()

    def set_meshgrid_trapz_Tip_Dense(self, xstart, xend, ystart, yend, xTip, yTip, x_dense_num, y_dense_num,
                                     x_outer_num, y_outer_num, x_inteval=0.2, y_inteval=0.2):
        super().set_meshgrid_trapz_Tip_Dense(xstart, xend, ystart, yend, xTip, yTip, x_dense_num, y_dense_num,
                                             x_outer_num, y_outer_num, x_inteval, y_inteval)
        self.set_inner_materials()




class DEM2D_5(PINN2D):
    def __init__(self, model: nn.Module):
        super().__init__(model)

    def hard_sx(self,sx,x,y):
        return sx
    def hard_sy(self,sy,x,y):
        return sy
    def hard_sxy(self,sxy,x,y):
        return sxy

    
    def infer(self,xy):
        uv = self.model(xy)
        x = xy[...,0]
        y = xy[...,1]
        u,v = self.hard_u(uv[0].squeeze(-1),x,y) , self.hard_v(uv[1].squeeze(-1),x,y)
        sx, sy, sxy = self.hard_sx(uv[2].squeeze(-1),x,y) , self.hard_sy(uv[3].squeeze(-1),x,y) , self.hard_sxy(uv[4].squeeze(-1),x,y)
        return u,v,sx,sy,sxy

    def pred_uv(self, xy):
        u,v,sx,sy,sxy = self.infer(xy)
        return u,v
    
    def pred_stress(self, xy):
        u,v,sx,sy,sxy = self.infer(xy)
        return sx,sy,sxy
    
    def constitutive_loss(self)->torch.Tensor:
        u , v, sx , sy , sxy = self.infer(self.XY)
        eXX,eYY,eXY = self.compute_Strain(u,v,self.XY)
        sx_constitutive , sy_constitutive , sxy_constitutive = self.constitutive(eXX,eYY,eXY)
        return torch.stack([self.criterion(sx,sx_constitutive) ,
                            self.criterion(sy,sy_constitutive) , 
                            self.criterion(sxy,sxy_constitutive)])

    def print_loss(self):
        super().print_loss()
        print(self.constitutive_loss.__name__,':',self.constitutive_loss().item())        



if __name__ == '__main__':

    k = -1/np.sqrt(3)
    b = 0.5
    print(np.arctan(k))
    outward_normal = np.arctan(k) + np.pi/2
    print(outward_normal*180/np.pi)
    l_x = np.cos(outward_normal)
    l_y = np.cos(outward_normal - np.pi/2)
    print(l_x,l_y)
