from typing import Iterator
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import Elasticity2D
from torch.nn import functional as F
from itertools import chain
import NN
import multidomain
from Geometry import Geometry1D,LineSegement,LocalAxis
import numpy as np
from get_grad import get_grad

class EnrichBasis:
    def __init__(self,
                 HeavisideZero = 0):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.set_Heaviside(HeavisideZero)
        


    def getBasis(self,xy)->torch.Tensor:... 

    def set_no_grad(self):

        self.getBasis_grad = self.getBasis

        def getBasis_detach(xy):
            return self.getBasis_grad(xy).detach()
        
        self.getBasis = getBasis_detach

    def set_Heaviside(self,HeavisideZero):
        '''
        该函数只在两种情况下被使用：
        1.初始化;2.计算应力强度因子
        '''

        def HeavisideZero0(x):
            # return 0.5 * (torch.sign(x) + 1)
            # return torch.heaviside(x, self.HeavisideZero)
            return torch.sign(torch.relu(x))

        def HeavisideZero1(x):
            return 1 - torch.sign(torch.relu(-x))
        
        Heavisides = [HeavisideZero0,HeavisideZero1]
        self.Heaviside = Heavisides[HeavisideZero]

    # torch.sign不一样
    def sign(self,x):
        return (self.Heaviside(x) - 0.5 ) * 2
    def zero(self,x,y):
        return torch.zeros_like(x)

    def one(self,x,y):
        return torch.ones_like(x)
    
    def neg_one(self,x,y):
        return - self.one(x,y)
    
    def getgrad2(self,x:torch.Tensor,y:torch.Tensor):
        x.requires_grad_();y.requires_grad_()
        phi = self.getBasis(torch.stack((x,y),dim=1))
        dxx = torch.abs(get_grad(get_grad(phi,x),x).detach())
        dxy = torch.abs(get_grad(get_grad(phi,x),y).detach())
        dyy = torch.abs(get_grad(get_grad(phi,y),y).detach())
        return dxx + 2 * dxy + dyy
    

class EnrichBasisPolar:
    def __init__(self):
        pass  

    def getBasis(self,r,theta):... 


class multiBasis(EnrichBasis):
    '''拼接多个enrich'''
    def __init__(self,BasisList:list[EnrichBasis]):
        super().__init__()
        self.BasisList = BasisList

    def getBasis(self,xy):
        basis = torch.cat(list(map(lambda x:x.getBasis(xy) ,self.BasisList)),dim=1)
        return basis


class EnrichNet(nn.Module):
    '''位移富集项'''
    def __init__(self,u_net:nn.Module,v_net:nn.Module,
                 EnrichBasis:EnrichBasisPolar,
                 local_axis:LocalAxis,
                 input_axis):
        super().__init__()
        self.u_net = u_net
        self.v_net = v_net
        self.EnrichBasis = EnrichBasis
        self.local_axis = local_axis
        if input_axis == 'xy':
            self.get_axis = self._get_axis_xy
        elif input_axis == 'polar':
            self.get_axis = self._get_axis_polar
        elif input_axis == 'theta':
            self.get_axis = self._get_axis_theta
        elif input_axis == 'polar_normalized':
            self.get_axis = self._get_axis_polar_normalized

    def get_axis(self,xy,r,theta):...
    def _get_axis_xy(self,xy,r,theta):
        return xy
    def _get_axis_polar(self,xy,r,theta):
        return torch.stack((r,theta),dim=1)
    def _get_axis_theta(self,xy,r,theta):
        return theta.unsqueeze(-1)
    def _get_axis_polar_normalized(self,xy,r,theta):
        theta /= torch.pi
        return torch.stack((r,theta),dim=1)
#    这里u_net输入是什么
    def forward(self,xy):
        r,theta = self.local_axis.cartesianToPolar(xy[...,0],xy[...,1])
        basis = self.EnrichBasis.getBasis(r,theta)
        axis = self.get_axis(xy,r,theta)
        u_enrich = torch.sum(self.u_net(axis) * basis,dim=1)
        v_enrich = torch.sum(self.v_net(axis) * basis,dim=1)
        return [u_enrich,v_enrich]


class extendAxisNet(nn.Module):
    def __init__(self,net:nn.Module,
                 extendAxis:EnrichBasis) -> None:
        super().__init__()
        self.net = net
        self.extendAxis = extendAxis

    def forward(self,xy):
        basis = self.extendAxis.getBasis(xy)
        basis = basis
        axis = torch.cat((xy,basis),dim = 1)

        k = axis[:,2:3]

        # print(get_grad(k,xy)[...,0])
        # print(self.net(axis))
        net = self.net(axis)[0]

        # print(f"u1_net:{net}")
        # print(f"axis:{axis}")
        #
        # print(f"dgamma_dx:{get_grad(k, xy)[..., 0]}")
        # print(f"du1_dx:{get_grad(net,axis)[...,0]}")
        # print(f"du1_dgamma:{get_grad(net, axis)[..., 2]}")



        # print(xy)
        # print(get_grad(k,xy))
        # print(xy)
        # k= k.reshape(21,21)
        # print(k)
        # print(axis[:,2:3])
        return self.net(axis)

    def infer(self,axis):
        return self.net(axis)    
    
    def set_extend_axis(self,extendAxis:EnrichBasis):
        self.extendAxis = extendAxis




class extendAxisNet_muti(nn.Module):
    def __init__(self, net: nn.Module,
                 extendAxis1: EnrichBasis,extendAxis2: EnrichBasis) -> None:
        super().__init__()
        self.net = net
        self.extendAxis1 = extendAxis1
        self.extendAxis2 = extendAxis2
        # self.extendAxis3 = extendAxis3

    def forward(self, xy):
        basis1 = self.extendAxis1.getBasis(xy)
        basis2 = self.extendAxis2.getBasis(xy)
        # basis3 = self.extendAxis3.getBasis(xy)

        axis = torch.cat((xy, basis1), dim=1)
        axis = torch.cat((axis, basis2), dim=1)
        # axis = torch.cat((axis, basis3), dim=1)
        return self.net(axis)

    def infer(self, axis):
        return self.net(axis)

    def set_extend_axis(self, extendAxis1: EnrichBasis,extendAxis2: EnrichBasis):
        self.extendAxis1 = extendAxis1
        self.extendAxis2 = extendAxis2
        # self.extendAxis3 = extendAxis3


class extendOutputNet(nn.Module):
    '''位移富集项'''
    def __init__(self,u_net:nn.Module,v_net:nn.Module,
                 EnrichBasis:EnrichBasis):
        super().__init__()
        self.u_net = u_net
        self.v_net = v_net
        self.EnrichBasis = EnrichBasis

    def forward(self,xy):
        basis = self.EnrichBasis.getBasis(xy)
        u_enrich = torch.sum(self.u_net(xy) * basis,dim=1)
        v_enrich = torch.sum(self.v_net(xy) * basis,dim=1)
        return [u_enrich,v_enrich]





class PositionalEncoder(EnrichBasis):
    def __init__(self,enrichment:EnrichBasis,order=2) -> None:
        '''参考来源于nerf的positional encoding
            舍弃cos项减少输入维度'''
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.EnrichBasis = enrichment
        self.order = order
        self.out_num = order + 1
        self.freq = torch.tensor([2 ** i for i in range(order)]).to(self.device) * torch.pi


    
    def getBasis(self, xy):
        basis = self.EnrichBasis.getBasis(xy)
        sin_encoder = torch.sin (basis * self.freq)
        return torch.cat([sin_encoder,basis],dim=-1)

class CrackLinePolar(EnrichBasis):
    def __init__(self,xy0,xy1,right_beta,tip):
        '''使用sin平滑化阶跃函数'''
        super().__init__()

        left_beta = right_beta + np.pi

        self.left_tip = LocalAxis(xy0[0],xy0[1],beta=left_beta)
        self.left_psi_line = LineSegement.init_theta(xy0,left_beta-np.pi/2)

        self.right_tip = LocalAxis(xy1[0],xy1[1],beta=right_beta)
        self.right_psi_line = LineSegement.init_theta(xy1,right_beta+np.pi/2)

        self.line = LineSegement(xy0,xy1)
        self.a = self.line.length

        self.setTipBasis(tip)
        self.getH = self.getH_standard


    def setTipBasis(self,tip):
        self.left_basis = self.zero
        self.right_basis =  self.zero 
        if tip == 'left':
            self.left_basis = self.left_sin
        elif tip == 'right':
            self.right_basis =  self.right_sin
        elif tip == 'both':
            self.left_basis = self.left_sin
            self.right_basis =  self.right_sin 
    
    def left_sin(self,x,y):
        theta = self.left_tip.getLocalTheta(x,y)
        return - torch.sin(theta)
    
    def right_sin(self,x,y):
        theta = self.right_tip.getLocalTheta(x,y)
        return torch.sin(theta)
    
    def getH_standard(self,xy):
        x = xy[...,0]; y = xy[...,1]
        return self.sign(-self.line.levelset(x,y))
    
    def getBasis(self, xy):
        x = xy[...,0]; y = xy[...,1]

        H_left = self.Heaviside(self.left_psi_line.levelset(x,y))
        H_right = self.Heaviside(-self.right_psi_line.levelset(x,y))

        crack_surface = self.getH(xy) * H_left * H_right
        left_tip = self.left_basis(x,y) * (1 - H_left) 
        right_tip = self.right_basis(x,y) * (1 - H_right)
        
        basis = crack_surface + left_tip + right_tip
        return basis.unsqueeze(-1)
    
    def set_ls(self,ls:float):
        '''便于求应力强度因子设置ls为定值'''

        def constant(xy):
            x = xy[...,0]; y = xy[...,1]
            return self.one(x,y) * ls

        # self.getH_standard = self.getH
        self.getH = constant

    def restore_ls(self):
        self.getH = self.getH_standard


class CrackStepABSBasis(EnrichBasis):
    def __init__(self,local_axis : LocalAxis,a):
        '''a:边裂纹时是长度,中心裂纹时是一半长度'''
        super().__init__()
        self.local_axis = local_axis
        self.a = torch.tensor(a)

    def H(self,local_y):
        return torch.where(local_y >0,1,-1)
        # return torch.where(local_y >0,1,0)
    
    def psi(self,local_x):
        return F.relu(1-torch.abs(local_x/self.a))
        # return torch.where(torch.abs(local_x)>self.a,0,(1-(local_x/self.a)**2)**2)
    
    
    def getBasis(self,xy):
        x = xy[...,0]; y = xy[...,1]
        local_x , local_y = self.local_axis.cartesianToLocal(x,y)
        H = self.H(local_y)
        psi = self.psi(local_x)
        phi = H * psi
        return phi.unsqueeze(-1)


class CrackStepBasis(EnrichBasis):
    def __init__(self,local_axis : LocalAxis,a):
        '''a:边裂纹时是长度,中心裂纹时是一半长度'''
        super().__init__()
        self.local_axis = local_axis
        self.a = torch.tensor(a)

    def H(self,local_y):
        return self.sign(local_y)
        # return torch.where(local_y >0,1,-1)
        # return torch.where(local_y >0,1,0)
    
    def psi(self,local_x):
        return F.relu(1-(local_x/self.a)**2)**2
        # return torch.where(torch.abs(local_x)>self.a,0,(1-(local_x/self.a)**2)**2)
    
    
    def getBasis(self,xy):
        x = xy[...,0]; y = xy[...,1]
        local_x , local_y = self.local_axis.cartesianToLocal(x,y)
        H = self.H(local_y)
        psi = self.psi(local_x)
        phi = H * psi
        return phi.unsqueeze(-1)


class CrackStepLevelSet(CrackStepBasis):
    def __init__(self, local_axis: LocalAxis, a , levelset):
        super().__init__(local_axis, a)
        self.levelset = levelset

    def getBasis(self,xy):
        x = xy[...,0]; y = xy[...,1]
        local_x , local_y = self.local_axis.cartesianToLocal(x,y)
        ls = self.levelset(x,y)
        H = self.H(-ls)
        psi = self.psi(local_x)
        phi = H * psi
        return phi.unsqueeze(-1)  

    def getH(self,xy):
        x = xy[...,0]; y = xy[...,1]
        ls = self.levelset(x,y)
        return self.H(-ls)


class CrackStepDecay(CrackStepBasis):
    def __init__(self,local_axis : LocalAxis,a,r0_ratio = 1):
        '''在超过裂纹长度3倍时Enrichment衰减为0.5倍'''
        super().__init__(local_axis,a)
        self.r0 = r0_ratio * a
    
    def getBasis(self, xy):
        phi = super().getBasis(xy)
        
        r,_ = self.local_axis.cartesianToPolar(xy[...,0],xy[...,1])
        weight = F.sigmoid( 4 * (self.r0 - r)).unsqueeze(-1)
        return weight * phi


class InnerCrackBasis(CrackStepLevelSet):

    def __init__(self, local_axis: LocalAxis, a , levelset):
        ''' 没有裂尖，在裂纹边界处不连续
            a:边裂纹时是长度,中心裂纹时是一半长度'''        
        super().__init__(local_axis, a , levelset)
        '''
        此处Heaviside用于判断点是否在线段内
        以及是否在裂纹面左侧
        改变Heaviside符号会同时改变裂纹面上点的符号以及线段端点是否属于线段
        '''
        self.set_Heaviside(1)
    
    def psi(self,local_x):
        return self.Heaviside(self.a - torch.abs(local_x))
        # var = self.a - torch.abs(local_x)
        # print(var)
        # tmp = self.Heaviside(var)
        # print(tmp)
        # return tmp
        # return 0.5 * (self.sign() + 1)


class EdgeCrackBasis(CrackStepLevelSet):

    def __init__(self, local_axis: LocalAxis, a , levelset, tip = 'right'):
        ''' 只有一个裂尖，在裂纹边界处不连续
            a:边裂纹时是长度,中心裂纹时是一半长度'''        
        super().__init__(local_axis, a , levelset)
        if tip == 'right':
            self.direction = -1
        elif tip == 'left':
            self.direction = 1
    
    def psi(self,local_x):
        local_x_new = 0.5 * (self.sign(self.direction * local_x)+1) + F.relu(- self.direction * local_x)
        return super().psi(local_x_new)


class multiCrackStep(EnrichBasis):
    def __init__(self,cracks:list[CrackStepBasis],normalized = True):
        super().__init__()
        self.cracks = cracks
        self.num = len(cracks)
        self.a = torch.stack([self.cracks[i].a for i in range(self.num)]).float().to(device=self.device)
        self.weights = self.a / torch.sum(self.a)

        if normalized:  self.getBasis = self.getBasis_normalized

        else: self.getBasis = self.getBasis_unnormalized

    
    def getBasis_normalized(self, xy):
        basis = torch.sum(self.weights * torch.cat(list(map(lambda x:x.getBasis(xy) ,self.cracks)),dim=1),dim=1).unsqueeze(-1)
        return basis
    
    def getBasis_unnormalized(self, xy):
        basis = torch.sum(torch.cat(list(map(lambda x:x.getBasis(xy) ,self.cracks)),dim=1),dim=1).unsqueeze(-1)
        return basis


class multiInnerSurfaces(EnrichBasis):
    def __init__(self,points:list[list]):
        
        self.surfaces = []
        for i in range(len(points)-1):
            x0 = points[i][0] ; y0 = points[i][1]
            x1 = points[i+1][0] ; y1 = points[i+1][1]
            # line =
            self.surfaces.append(InnerCrackBasis(
                local_axis=LocalAxis(x0 = (x0 + x1)/2,
                                        y0 = (y0 + y1)/2,
                                        beta=0.0),
                a = abs(x1 - x0)/2,
                levelset=LineSegement([x0,y0],[x1,y1]).levelset))
            
        # super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.a = torch.sum(torch.stack([self.surfaces[i].a 
                                        for i in range(len(self.surfaces))]).float().to(device=self.device))
        
        self.set_Heaviside(1)
        
    def set_Heaviside(self, HeavisideZero):
        for surface in self.surfaces:
            surface.set_Heaviside(HeavisideZero)
        super().set_Heaviside(HeavisideZero)
    

    def getBasis(self,xy):
        '''
        torch.sign用来过滤两条边上的重合点,>0为1,=0为0,<0为-1
        比如一维线段(0,1)和(1,2)组成的线段组,在x=1处会导致多值
        '''
        return torch.sign(torch.sum(torch.cat(list(map(lambda x:x.getBasis(xy) ,self.surfaces)),dim=1),dim=1))





class PolylineCrack(EnrichBasis):
    def __init__(self,points:list[list],tip='both'):
        super().__init__()
        self.surfaces=[]

        def genLeftEdge():
            x0 = points[0][0] ; y0 = points[0][1]
            x1 = points[1][0] ; y1 = points[1][1]
            self.surfaces.append(EdgeCrackBasis(
                local_axis=LocalAxis(x1,y1,0.0),
                a = abs(x1 - x0),
                levelset=LineSegement([x0,y0],[x1,y1]).levelset,
                tip = 'left'
            ))

        def genRightEdge():
            x0 = points[-2][0] ; y0 = points[-2][1]
            x1 = points[-1][0] ; y1 = points[-1][1]
            self.surfaces.append(EdgeCrackBasis(
                local_axis=LocalAxis(x0,y0,0.0),
                a = abs(x1 - x0),
                levelset=LineSegement([x0,y0],[x1,y1]).levelset,
                tip = 'right'
            ))
        
        self.x0 = points[0][0] ; self.y0 = points[0][1]
        self.x1 = points[-1][0] ; self.y1 = points[-1][1]


        if tip == 'left':
            # genLeftEdge()
            # self.genInnerSurfaces(points=points[1:])
            self.x_center = self.x1
            self.y_center = self.y1
            self.xlen =  self.x_center - self.x0
        elif tip == 'right':
            # self.genInnerSurfaces(points=points[:-1])
            # genRightEdge()
            self.x_center = self.x0
            self.y_center = self.y0
            self.xlen =  self.x1 - self.x_center
        elif tip == 'both':
            # genLeftEdge()
            # self.genInnerSurfaces(points=points[1:-1])    
            # genRightEdge()   
            self.x_center = (self.x0 + self.x1)/2
            self.y_center = (self.y0 + self.y1)/2
            self.xlen = self.x_center - self.x0
        elif tip == None :
            # self.genInnerSurfaces(points=points)
            pass   
        else:
            raise Exception() 
        
        # self.a = torch.sum(torch.stack([self.surfaces[i].a 
        #                                 for i in range(len(self.surfaces))]).float().to(device=self.device))

        self.genInnerSurfaces(points)
        self.a = torch.sum(torch.stack([self.surfaces[i].a 
                                        for i in range(len(self.surfaces))]).float().to(device=self.device))

        self.psi_basis = CrackStepBasis(local_axis=LocalAxis(self.x_center,self.y_center,0.0),
                                        a = self.xlen)


    def genInnerSurfaces(self,points:list[list]):
        for i in range(len(points)-1):
            x0 = points[i][0] ; y0 = points[i][1]
            x1 = points[i+1][0] ; y1 = points[i+1][1]
            self.surfaces.append(InnerCrackBasis(
                local_axis=LocalAxis(x0 = (x0 + x1)/2,
                                     y0 = (y0 + y1)/2,
                                     beta=0.0),
                a = abs(x1 - x0)/2,
                levelset=LineSegement([x0,y0],[x1,y1]).levelset))
    
    def getH(self,xy):
        return torch.sum(torch.cat(list(map(lambda x:x.getBasis(xy) ,self.surfaces)),dim=1),dim=1)

    def getBasis(self, xy):
        # basis = torch.sum(torch.cat(list(map(lambda x:x.getBasis(xy) ,self.surfaces)),dim=1),dim=1).unsqueeze(-1)
        # H = torch.sum(torch.cat(list(map(lambda x:x.getH(xy) ,self.surfaces)),dim=1),dim=1).unsqueeze(-1)
        # H_list = torch.stack(list(map(lambda x:x.getH(xy) ,self.surfaces)))
        x = xy[...,0]; y = xy[...,1]
        local_x , local_y = self.psi_basis.local_axis.cartesianToLocal(x,y)
        psi = self.psi_basis.psi(local_x)
        H = self.getH(xy)
        
        phi = H * psi
        return phi.unsqueeze(-1)


class CrackLinePolar(EnrichBasis):
    def __init__(self, xy0, xy1, right_beta, tip):
        '''使用sin平滑化阶跃函数'''
        super().__init__()

        left_beta = right_beta + np.pi

        self.left_tip = LocalAxis(xy0[0], xy0[1], beta=left_beta)
        self.left_psi_line = LineSegement.init_theta(xy0, left_beta - np.pi / 2)

        self.right_tip = LocalAxis(xy1[0], xy1[1], beta=right_beta)
        self.right_psi_line = LineSegement.init_theta(xy1, right_beta + np.pi / 2)

        self.line = LineSegement(xy0, xy1)
        self.a = self.line.length

        self.setTipBasis(tip)
        self.getH = self.getH_standard

    def setTipBasis(self, tip):
        self.left_basis = self.zero
        self.right_basis = self.zero
        if tip == 'left':
            self.left_basis = self.left_sin
        elif tip == 'right':
            self.right_basis = self.right_sin
        elif tip == 'both':
            self.left_basis = self.left_sin
            self.right_basis = self.right_sin

    def left_sin(self, x, y):
        theta = self.left_tip.getLocalTheta(x, y)
        return - torch.sin(theta)

    def right_sin(self, x, y):
        theta = self.right_tip.getLocalTheta(x, y)
        return torch.sin(theta)

    def getH_standard(self, xy):
        x = xy[..., 0];
        y = xy[..., 1]
        return self.sign(-self.line.levelset(x, y))

    def getBasis(self, xy):
        x = xy[..., 0];
        y = xy[..., 1]

        H_left = self.Heaviside(self.left_psi_line.levelset(x, y))

        H_right = self.Heaviside(-self.right_psi_line.levelset(x, y))
        crack_surface = self.getH(xy) * H_left * H_right

        left_tip = self.left_basis(x, y) * (1 - H_left)
        right_tip = self.right_basis(x, y) * (1 - H_right)

        basis = crack_surface + left_tip + right_tip
        return basis.unsqueeze(-1)

    def set_ls(self, ls: float):
        '''便于求应力强度因子设置ls为定值'''

        def constant(xy):
            x = xy[..., 0];
            y = xy[..., 1]
            return self.one(x, y) * ls

        # self.getH_standard = self.getH
        self.getH = constant

    def restore_ls(self):
        self.getH = self.getH_standard


class PolylinePolar(CrackLinePolar):
    def __init__(self,points:list[list],tip='both'):
        # super().__init__(points=points,tip=None)

        self.surfaces = multiInnerSurfaces(points)
        self.a = self.surfaces.a
        self.getH = self.surfaces.getBasis

        # self.left_line = LineSegement(xy0=points[1],xy1=points[0])
        # self.left_beta = self.left_line.tangent_theta
        # self.left_tip = LocalAxis(points[0][0],points[0][1],beta=self.left_beta)
        # self.left_psi = LineSegement.init_theta(points[0],self.left_beta-np.pi/2)
        self.left_tip = LocalAxis(points[0][0],points[0][1],beta=np.pi)
        self.left_psi_line = LineSegement.init_theta(points[0],np.pi/2)

        # self.right_line = LineSegement(xy0=points[-2],xy1=points[-1])
        # self.right_beta = self.right_line.tangent_theta
        # self.right_tip = LocalAxis(points[-1][0],points[-1][1],beta=self.right_beta)
        # self.right_psi = LineSegement.init_theta(points[-1],self.right_beta+np.pi/2)
        self.right_tip = LocalAxis(points[-1][0],points[-1][1],beta=0.0)
        self.right_psi_line = LineSegement.init_theta(points[-1],np.pi/2)

        self.setTipBasis(tip)
        self.set_Heaviside(1)

    def set_Heaviside(self, HeavisideZero):
        self.surfaces.set_Heaviside(HeavisideZero)
        super().set_Heaviside(HeavisideZero)






class BimaterialLSBasis(EnrichBasis):
    def __init__(self,geometry:Geometry1D):
        super().__init__()
        self.levelset = geometry.levelset

    def levelset_relu(self,xy):
        x = xy[..., 0];
        y = xy[..., 1];
        ls = torch.relu(self.levelset(x, y))
        return ls
    def levelset_abs(self,xy):
        x = xy[..., 0];
        y = xy[..., 1];

        ls = torch.abs(self.levelset(x, y))
        return ls

    # def levelset_abs(self, xy):
    #     x = xy[..., 0]
    #     y = xy[..., 1]
    #     ls = self.levelset(x, y)
    #
    #     # 创建副本，避免原数据被修改
    #     ls_scaled = ls.clone()
    #
    #     # 取出正负掩码
    #     pos_mask = ls > 0
    #     neg_mask = ls < 0
    #
    #     # 分别归一化
    #     if pos_mask.any():
    #         ls_scaled[pos_mask] = ls[pos_mask] / torch.max(ls[pos_mask])
    #     if neg_mask.any():
    #         ls_scaled[neg_mask] = ls[neg_mask] / torch.min(ls[neg_mask])  # 注意 min 是负数
    #
    #     # 如果你需要绝对值结果，就在最后取 abs
    #     ls_scaled = torch.abs(ls_scaled)
    #
    #     return ls_scaled

    def getBasis(self,xy):
        ls = self.levelset_abs(xy)
        return ls.unsqueeze(-1)

    
class BimaterialLSBasis_relu(EnrichBasis):
    def __init__(self,geometry:Geometry1D):
        super().__init__()
        self.levelset = geometry.levelset

    def levelset_relu(self,xy):
        x = xy[..., 0];
        y = xy[..., 1];
        ls = torch.relu(self.levelset(x, y))
        return ls

    def getBasis(self,xy):
        ls = self.levelset_relu(xy)
        return ls.unsqueeze(-1)

class CustomAbs(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # 保存输入供反向传播使用
            ctx.save_for_backward(input)
            return torch.abs(input)

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input > 0] = grad_output[input > 0]
            grad_input[input < 0] = -grad_output[input < 0]
            grad_input[input == 0] = 0 #自定义梯度值，可改为 -1 或其他值
            return grad_input


class BimaterialLSBasis_CustomAbs(EnrichBasis):
    def __init__(self,geometry:Geometry1D):
        super().__init__()
        self.levelset = geometry.levelset
    def levelset_abs(self,xy):
        x = xy[..., 0];
        y = xy[..., 1];

        ls = self.levelset(x, y)
        ls = CustomAbs.apply(ls)
        return ls
    def getBasis(self,xy):
        ls = self.levelset_abs(xy)
        return ls.unsqueeze(-1)

class MultiEnrichment(nn.Module):
    def __init__(self, enrichments:list):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.enrichments = nn.ModuleList(enrichments)

    def forward(self, xy):
        u = torch.zeros_like(xy[...,0]); v = torch.zeros_like(xy[...,0])
        for enrichment in self.enrichments:
            uv = enrichment(xy)
            u += uv[0]
            v += uv[1]
        return [u,v]


class EnrichedNN(nn.Module):
    '''富集函数+本身的位移假设'''
    def __init__(self, uv_net:nn.Module,enrichment:nn.Module):
        '''富集函数+本身的位移假设'''
        super().__init__()
        self.uv_net = uv_net
        self.enrichment = enrichment

    def forward(self, xy):
        uv = self.uv_net(xy)
        uv_enriched = self.enrichment(xy)
        # u = uv[0].squeeze(-1) + uv_enriched[0]
        # v = uv[1].squeeze(-1) + uv_enriched[1]  
        u = uv[0].squeeze(-1) + uv_enriched[0].squeeze(-1)
        v = uv[1].squeeze(-1) + uv_enriched[1].squeeze(-1)  
        return [u,v]  





class TransferNN(EnrichedNN):

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.enrichment.parameters(recurse)
    
    def load_uvnet(self,path,loadtype='state'):
        if loadtype=='state':
            self.uv_net.load_state_dict(torch.load(path+'.pth',map_location=torch.device('cpu')))
        else:
            self.uv_net=torch.load(path+'.pth',map_location=torch.device('cpu'))


class SQRTBasis(EnrichBasis):

    def __init__(self,x0,y0):
        super().__init__()
        self.x0 = x0
        self.y0 = y0

    def getBasis(self,xy):
        x = xy[...,0];y = xy[...,1]
        r = torch.sqrt( (x-self.x0) ** 2 + (y-self.y0) ** 2 )
        r_sqrt = torch.sqrt(r)
        return r_sqrt.unsqueeze(-1)


class HomoXFEMEnrichBasis(EnrichBasisPolar):
    
    def getBasis(self,r,theta):
        sin = torch.sin(theta)
        cos_half = torch.cos(theta/2)
        sin_half = torch.sin(theta/2)
        # r = torch.sqrt( (xy[...,0]-self.x0) ** 2 + (xy[...,1]-self.y0) ** 2 )
        r_sqrt = torch.sqrt(r)
        basis = (r_sqrt * torch.stack([sin_half,cos_half,
                                        sin_half * sin,
                                        cos_half * sin]))
        return basis.T


class SQRTBasisPolar(EnrichBasisPolar):
    
    def getBasis(self,r,theta):
        r_sqrt = torch.sqrt(r)
        return r_sqrt.unsqueeze(-1)


class SQRTSINBasis(EnrichBasisPolar):
    
    def getBasis(self,r,theta):
        r_sqrt = torch.sqrt(r)
        sin = torch.sin(theta/2)
        return (r_sqrt * sin).unsqueeze(-1)


class OneBasis(EnrichBasis):
    '''相当于不Enrich'''
    
    def getBasis(self,xy):
        one = torch.ones_like(xy[...,0])
        return one.unsqueeze(-1)


class BimaterialXFEMBasis(EnrichBasisPolar):

    def __init__(self,epsilon):
        super().__init__()
        self.epsilon = epsilon
    
    def getBasis(self,r,theta):
        sin = torch.sin(theta)
        cos_half = torch.cos(theta/2)
        sin_half = torch.sin(theta/2)
        # r = torch.sqrt( (xy[...,0]-self.x0) ** 2 + (xy[...,1]-self.y0) ** 2 )
        r_sqrt = torch.sqrt(r)

        eps_lnr = self.epsilon * torch.log(r)
        cos_eps_lnr = torch.cos(eps_lnr)
        sin_eps_lnr = torch.sin(eps_lnr)

        eps_theta = self.epsilon * theta
        exp_eps_theta = torch.exp(eps_theta)
        exp_neg_eps_theta = 1 / exp_eps_theta
        

        basis = (r_sqrt * torch.stack([cos_eps_lnr * exp_neg_eps_theta * cos_half,
                                       cos_eps_lnr * exp_neg_eps_theta * sin_half,
                                       cos_eps_lnr * exp_eps_theta     * cos_half,
                                       cos_eps_lnr * exp_eps_theta     * sin_half,
                                       sin_eps_lnr * exp_neg_eps_theta * cos_half,
                                       sin_eps_lnr * exp_neg_eps_theta * sin_half,
                                       sin_eps_lnr * exp_eps_theta     * cos_half,
                                       sin_eps_lnr * exp_eps_theta     * sin_half,

                                       cos_eps_lnr * exp_eps_theta     * cos_half * sin,
                                       cos_eps_lnr * exp_eps_theta     * sin_half * sin,
                                       sin_eps_lnr * exp_eps_theta     * cos_half * sin,
                                       sin_eps_lnr * exp_eps_theta     * sin_half * sin]))
        return basis.T


class BimaterialOscillation(EnrichBasisPolar):

    def __init__(self,epsilon):
        super().__init__()
        self.epsilon = epsilon
    
    def getBasis(self,r,theta):

        r_sqrt = torch.sqrt(r)

        eps_lnr = self.epsilon * torch.log(r)
        cos_eps_lnr = torch.cos(eps_lnr)
        sin_eps_lnr = torch.sin(eps_lnr)
        

        basis = (r_sqrt * torch.stack([r_sqrt * cos_eps_lnr,
                                       r_sqrt * sin_eps_lnr]))
        return basis.T


class HomoXFEMEnrichNet(nn.Module):

    def __init__(self,x0,y0,u_net:nn.Module,v_net:nn.Module,beta=0.0):
        '''使用XFEM的enrichment'''

        super().__init__()
        self.u_net = u_net
        self.v_net = v_net
        self.local_axis = LocalAxis(x0,y0,beta)  
        self.EnrichBasis = HomoXFEMEnrichBasis()


    def forward(self,xy):
        r,theta = self.local_axis.cartesianToPolar(xy[...,0],xy[...,1])
        basis = self.EnrichBasis.getBasis(r,theta)
        u_enrich = torch.sum(self.u_net(xy) * basis,dim=1)
        v_enrich = torch.sum(self.v_net(xy) * basis,dim=1)
        #uv = self.net(torch.stack((r,theta),dim=1))

        return [u_enrich,v_enrich]


class EnrichSQRTNN(nn.Module):

    def __init__(self,x0,y0,net:nn.Module,beta=0.0):

        super().__init__()

        self.local_axis = LocalAxis(x0,y0,beta)
        self.net = net

    def forward(self,xy):
        # r = torch.sqrt( (xy[...,0]-self.x0) ** 2 + (xy[...,1]-self.y0) ** 2 )
        r,theta = self.local_axis.cartesianToPolar(xy[...,0],xy[...,1])
        r_sqrt = torch.sqrt(r)

        uv = self.net(torch.stack((r,theta),dim=1))

        return [uv[0].squeeze(-1) * r_sqrt , uv[1].squeeze(-1) * r_sqrt]


class EnrichHomoCrackNN(nn.Module):

    def __init__(self,kappa,x0,y0,beta, K1_init=1.0,K2_init=1.0):
        '''
        ..inputs:
            K1_init,K2_init:K1,K2的初始值,默认为0.001
            kappa:材料属性
            x0,y0:裂纹尖端坐标
            beta:裂纹与x轴正向的夹角
        ..注意:
            归一化K的单位是sqrt(m)
        '''
        super(EnrichHomoCrackNN, self).__init__()

        self.K1 = nn.Parameter(torch.tensor(K1_init,requires_grad=True))
        # self.K1_scalar = K1_init
        self.K2 = nn.Parameter(torch.tensor(K2_init,requires_grad=True))
        # self.K2_scalar = K2_init
        self.kappa = torch.tensor(kappa)
        self.local_axis = LocalAxis(x0,y0,beta)


    def forward(self,xy):
        K1 = self.K1
        K2 = self.K2
        r,theta = self.local_axis.cartesianToPolar(xy[...,0],xy[...,1])
        r_sqrt = torch.sqrt(r)
        cos = torch.cos(theta)
        cos_half = torch.cos(theta/2)
        sin_half = torch.sin(theta/2)

        K1_u = r_sqrt * (self.kappa - cos) * cos_half
        K1_v = r_sqrt * (self.kappa - cos) * sin_half

        K2_u = r_sqrt * (self.kappa + 2 + cos) * sin_half
        K2_v = r_sqrt * (self.kappa - 2 + cos) * cos_half

        
        u = K1 * K1_u + K2 * K2_u
        v = K1 * K1_v + K2 * K2_v

        return [u , v]
    

class EnrichHomoCrackNN_local(EnrichHomoCrackNN):
    def __init__(self, kappa, x0, y0, beta, K1_init=1, K2_init=1, r0 = 0.5):
        super().__init__(kappa, x0, y0, beta, K1_init, K2_init)
        self.length = r0

    def forward(self, xy):
        r,theta = self.local_axis.cartesianToPolar(xy[...,0],xy[...,1])

        uv_enriched = super().forward(xy)
        weight =  F.sigmoid(5 * (1 - 2 * r/self.length))
        return [uv_enriched[0] * weight , uv_enriched[1] * weight]


class PINN_enriched(Elasticity2D.DEM2D_2):
    def set_enrichment(self,enrichment:nn.Module):
          self.enrichment = enrichment

    def pred_uv(self, xy):
        uv = self.model(xy)
        uv_enriched = self.enrichment(xy)
        u = self.hard_u(uv[0].squeeze(-1),xy[...,0],xy[...,1]) + uv_enriched[0]
        v = self.hard_v(uv[1].squeeze(-1),xy[...,0],xy[...,1]) + uv_enriched[1]  
        return u , v  


class Bidomain_enriched(multidomain.BiDomain):

    def __init__(self, PINN_domain1: PINN_enriched, PINN_domain2: PINN_enriched,enrichment_net:nn.Module):
        super().__init__(PINN_domain1, PINN_domain2)
        self.enrichment_net = enrichment_net.to(self.device)
        self.params = chain(self.PINN_domain1.model.parameters(), 
                            self.PINN_domain2.model.parameters(),
                            enrichment_net.parameters())
        self.PINN_domain1.set_enrichment(self.enrichment_net)
        self.PINN_domain2.set_enrichment(self.enrichment_net)
        self.models.append(self.enrichment_net)
        self.models_name.append('_enrichment')

    def save(self, name):
        torch.save(self.enrichment_net,name+'_enrichment.pth')
        return super().save(name)
    
    def load(self, path, loadtype='state'):

        if loadtype=='state':
            self.enrichment_net.load_state_dict(torch.load(path+'_enrichment.pth',map_location=torch.device('cpu')))
        else:
            self.enrichment_net=torch.load(path+'_enrichment.pth',map_location=torch.device('cpu'))
        self.PINN_domain1.set_enrichment(self.enrichment_net)
        self.PINN_domain2.set_enrichment(self.enrichment_net)
        return super().load(path, loadtype)
    
    def print_loss(self):
        print('K1:',self.enrichment_net.K1.cpu().detach().numpy(),
              'K2:',self.enrichment_net.K2.cpu().detach().numpy())
        return super().print_loss()  
    
    def eval(self):
        print('K1:',self.enrichment_net.K1.cpu().detach().numpy(),
              'K2:',self.enrichment_net.K2.cpu().detach().numpy())
        return super().eval()

    



if __name__ == '__main__':
        # x0 = torch.tensor(0.5); y0 = torch.tensor(0.5)
        # x = torch.tensor(0.0); y = torch.tensor(0.0)
        # beta = torch.tensor(0)
        # cos = torch.cos(beta)
        # sin = torch.sin(beta)
        # r = torch.sqrt( (x-x0) ** 2 + (y-y0) ** 2 )
        # '''相对于裂纹尖端方向的xy坐标'''
        # local_x = cos * (x-x0) + sin * (y-y0)
        # local_y = - sin * (x-x0) + cos * (y-y0)
        # theta=torch.arctan2(local_y,local_x)
        # print(r,torch.sqrt( local_x ** 2 + local_y ** 2 ),local_x,local_y,theta * 180 / torch.pi)
        pass