from torch._tensor import Tensor
from torch.nn import functional as F
import NN
from Geometry import LineSegement
from Geometry import LocalAxis
from Geometry import Circle
import numpy as np
from Enrichment import EnrichBasis,multiInnerSurfaces
import torch


class RELU2PSI(EnrichBasis):
    def __init__(self, levelset,
                 xy0, xy1,
                 tip,
                 left_beta,
                 right_beta):
        '''和论文中表达式一样
            未对扩充坐标进行归一化'''
        super().__init__()
        self.levelset = levelset

        self.left_beta = left_beta
        self.right_beta = right_beta

        self.left_psi = LineSegement.init_theta(xy0, left_beta - np.pi / 2).levelset_dist
        self.right_psi = LineSegement.init_theta(xy1, right_beta + np.pi / 2).levelset_dist

        self.a = LineSegement(xy0, xy1).length
        self.norm = self.a ** 2
        # self.left_psi = LineSegement.init_theta(xy0,left_beta-np.pi/2).levelset
        # self.right_psi =  LineSegement.init_theta(xy1,right_beta+np.pi/2).levelset
        if tip == 'left':
            self.right_psi = self.one
        elif tip == 'right':
            self.left_psi = self.one
        elif tip == 'both':
            # self.norm /= 4

            self.norm = (self.norm / 4) ** 2
        else:
            raise Exception()

        self.getH = self.getH_standard

    def getPSI(self, xy):
        x = xy[..., 0];
        y = xy[..., 1]
        return F.relu(- self.left_psi(x, y) * self.right_psi(x, y)) ** 2

    def getH_standard(self, xy):
        x = xy[..., 0];
        y = xy[..., 1]
        ls = self.levelset(x, y)
        return self.get_crack_sign(ls)

    def get_crack_sign(self, ls):
        return self.sign(-ls)

    def getBasis(self, xy):
        H = self.getH(xy)
        PSI = self.getPSI(xy)
        # print(self.norm)
        # print(self.a)
        # return ( PSI).unsqueeze(-1) / self.norm
        return (H * PSI).unsqueeze(-1) / self.norm

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








class RELU2PSILine(RELU2PSI):

        def  __init__(self, xy0, xy1, tip):
            self.Line = LineSegement(xy0, xy1)
            super().__init__(levelset=self.Line.levelset_dist,
                             xy0=xy0, xy1=xy1, tip=tip,
                             left_beta=self.Line.tangent_theta + np.pi,
                             right_beta=self.Line.tangent_theta)




class RELU2PSI_circle(EnrichBasis):
    def __init__(self,levelset,
                 x0, y0,
                 r):
        super().__init__()
        self.levelset = levelset
        self.local_axis = LocalAxis(x0, y0)
    def dist(self, x, y):
        return self.local_axis.getR(x, y)

    def norm_dist(self, x, y):
        return self.dist(x, y) / self.r

    def getBasis(self, xy):
        x = xy[..., 0];
        y = xy[..., 1]
        return abs(self.levelset(x,y)).unsqueeze(-1)
class RELU2PSICircle(RELU2PSI_circle):

    def __init__(self, x0, y0, r):
        self.circle = Circle(x0, y0,r)
        super().__init__(levelset=self.circle.dist,
                         x0=x0, y0=y0, r=r)
class RELU2PSI_circle_crack(EnrichBasis):
    def __init__(self, levelset,local_area,
                 x0, y0,r,
                 tip,
                 theta):
        '''和论文中表达式一样
            未对扩充坐标进行归一化'''
        super().__init__()
        self.levelset = levelset
        self.r = r
        self.theta = theta
        self.circle_x = x0
        self.circle_y = y0
        self.local_area = local_area

        self.left_x = x0 - r * np.sin(self.theta/2)
        self.left_y = y0 + r * np.cos(self.theta/2)
        self.right_x = x0 + r * np.sin(self.theta/2)
        self.right_y = y0 + r * np.cos(self.theta/2)

        self.xy0 = [self.left_x,self.left_y]
        self.xy1 = [self.right_x, self.right_y]
        self.left_psi = LineSegement.init_theta(self.xy0, self.theta/2 + np.pi / 2).levelset_dist
        self.right_psi = LineSegement.init_theta(self.xy1, -self.theta/2 + np.pi / 2).levelset_dist

        self.c = (self.circle_x**2+self.circle_y**2)*(np.cos(self.theta/2)*np.cos(self.theta/2)-np.sin(self.theta/2)*np.sin(self.theta/2))
        self.a = LineSegement(self.xy0, self.xy1).length
        self.norm = self.a ** 2
        # self.left_psi = LineSegement.init_theta(xy0,left_beta-np.pi/2).levelset
        # self.right_psi =  LineSegement.init_theta(xy1,right_beta+np.pi/2).levelset
        if tip == 'left':
            self.right_psi = self.one
        elif tip == 'right':
            self.left_psi = self.one
        elif tip == 'both':
            self.norm = 0
            # self.norm  = (self.norm)**2
            # self.norm = (self.norm / 4) ** 2
        else:
            raise Exception()

        self.getH = self.getH_standard


    def getPSI(self, xy):
        x = xy[..., 0];
        y = xy[..., 1];
        return F.relu(- F.relu(self.left_psi(x, y)) * self.right_psi(x, y))**2
    def getH_standard(self, xy):
        x = xy[..., 0];
        y = xy[..., 1]
        ls = -self.levelset(x, y)
        return self.get_crack_sign(ls)

    def get_crack_sign(self, ls):
        return self.sign(-ls)

    def getBasis(self, xy):
        y = xy[:,1]
        H = self.getH(xy)
        PSI = self.getPSI(xy)
        # ymax = max(y)
        # print(ymax)
        # ymax = self.local_area+self.circle_y+self.r*np.cos(self.theta/2)
        ymax = self.local_area+self.circle_y
        norm1 = (self.r**2)*(np.sin(self.theta/2)*np.sin(self.theta/2))
        norm2 = ((ymax-self.circle_y)**2)*(np.sin(self.theta/2)*np.sin(self.theta/2))
        k = norm2/norm1
        norm = norm1*norm2
        # norm_2 = np.sqrt(norm)
        basis = (H*PSI).unsqueeze(-1)/norm
        basis[basis < 0] *=k
        basis[basis > 0] /=k
        # return (H * PSI).unsqueeze(-1) / self.norm
        return basis

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

class RELU2PSIcircle_crack(RELU2PSI_circle_crack):

        def __init__(self, x0, y0, r,tip,theta,local_area):
            self.circle = Circle(x0, y0,r)
            super().__init__(levelset=self.circle.levelset_dist,
                             x0=x0, y0=y0,r = r, tip=tip,theta = theta,local_area = local_area)



class RELU2PSIPolyLine_x(RELU2PSI):
    def __init__(self,points:list[list],tip='both'):
        '''不能沿着垂直方向扩展,这个目前有点慢,需要改进'''
        self.surfaces=[]
        
        self.xy0 = points[0]
        self.xy1 = points[-1]

        self.surfaces = multiInnerSurfaces(points)
        self.a = self.surfaces.a

        super().__init__(levelset = self.surfaces.getBasis,
                         xy0=self.xy0, xy1=self.xy1, tip=tip,
                         left_beta = np.pi,
                         right_beta = 0.0)
        
        self.set_Heaviside(1)

    def getH_standard(self,xy):
       '''
       sign操作在InnerSurfaces类中,
       便于日后在多裂纹中求单独一条裂纹面的张开位移
       '''
       return self.levelset(xy)

#    HeavisideZero这是什么值
    def set_Heaviside(self, HeavisideZero):
        '''
        注意在折线段交界处有问题，不过目前碰不到(计算SIF点不能正好取在线段端点上,可能导致错误)
        改变Heaviside符号会同时改变裂纹面上点的符号以及线段端点是否属于线段
        '''
        self.surfaces.set_Heaviside(HeavisideZero)
        # super().set_Heaviside(HeavisideZero)






class RELU2PSIPolyLine(EnrichBasis):
    def __init__(self,points:list[list],tip='both'):
        '''与上一个的不同点:仅在最前端的裂尖有1-0的变化,
            由于偷懒写法，因此使用前请注意板的尺寸！'''
        # self.surfaces=[]
        self.xy0 = points[0]
        self.xy1 = points[-1]
        # print(self.xy0,self.xy1)
        # left_beta = LineSegement(points[1],points[0]).tangent_theta
        # right_beta = LineSegement(points[-2],points[-1]).tangent_theta

        if tip == 'right':
            # 面内Embedding
            inner_points = points+[[2.0,points[-1][1]]] #偷懒，直接把域最右边的点视为裂纹面的最后一个点，避免斜裂尖的一些处理麻烦
            self.inner_surfaces = multiInnerSurfaces(inner_points)
            # 裂尖Embedding
            # self.right_tip_line = LineSegement(points[-2],points[-1])
            # print(points[-2],points[-1])
            self.right_tip = RELU2PSILine(points[-2],points[-1],tip='right')
            # 裂尖与裂纹面的分割线
            self.split_line = LineSegement.init_theta(
                points[-2],self.right_tip.Line.tangent_theta+np.pi/2)


            # self.right_tip_left_psi = LineSegement.init_theta(points[-2],
            #                                 #加pi减pi/2#
            #                                     self.right_tip_line.tangent_theta+np.pi/2).levelset
        elif tip == 'left':
            # 面内Embedding
            inner_points = points+[[-2.0,points[-1][1]]] #偷懒，直接把域最右边的点视为裂纹面的最后一个点，避免斜裂尖的一些处理麻烦
            self.inner_surfaces = multiInnerSurfaces(inner_points)
            # 裂尖Embedding
            # self.right_tip_line = LineSegement(points[-2],points[-1])
            # print(points[-2],points[-1])
            self.right_tip = RELU2PSILine(points[-2],points[-1],tip='right')
            # 裂尖与裂纹面的分割线
            self.split_line = LineSegement.init_theta(points[-2],self.right_tip.Line.tangent_theta+np.pi/2)
            # print(self.right_tip.Line.tangent_theta)

            # self.right_tip_left_psi = LineSegement.init_theta(points[-2],
            #                                 #加pi减pi/2#
            #                                     self.right_tip_line.tangent_theta+np.pi/2).levelset
        else:
            raise Exception('还未实现!')
        # self.a = self.inner_surfaces.a + self.right_tip_line.length


        self.set_Heaviside(0)

    # def right_tip_basis(self , xy) -> Tensor:
    #     left_sign = self.Heaviside(self.right_tip_left_psi(xy[...,0],xy[...,1])).unsqueeze(-1)
    #     tip_basis = self.right_tip.getBasis(xy)
    #     return left_sign * tip_basis

        
    def getBasis(self, xy) -> Tensor:
        '''判断是否位于折裂纹的裂尖部位'''
        left_H = self.Heaviside(self.split_line.levelset(xy[...,0],xy[...,1])).unsqueeze(-1)
        inner_basis = self.inner_surfaces.getBasis(xy).unsqueeze(-1) * (1-left_H)
        right_tip = self.right_tip.getBasis(xy) *left_H
        self.inner_basis = self.inner_surfaces.getBasis(xy).unsqueeze(-1)
        # self.inner_basis = self.inner_surfaces.getBasis(xy).unsqueeze(-1)
        # self.right_tip1 = right_tip
        # self.right_tip_noleftH = self.right_tip.getBasis(xy)
        return (inner_basis + right_tip)*1

    
    def set_ls(self,ls:float):
        '''便于求应力强度因子设置ls为定值'''

        self.right_tip.set_ls(ls)

    def restore_ls(self):
        self.right_tip.restore_ls()


class RELU2PSIPolyLine_global(EnrichBasis):
    def __init__(self, points: list[list], tip='both'):
        '''与上一个的不同点:仅在最前端的裂尖有1-0的变化,
            由于偷懒写法，因此使用前请注意板的尺寸！'''
        # self.surfaces=[]
        self.xy0 = points[0]
        self.xy1 = points[-1]
        # print(self.xy0,self.xy1)
        # left_beta = LineSegement(points[1],points[0]).tangent_theta
        # right_beta = LineSegement(points[-2],points[-1]).tangent_theta

        if tip == 'right':
            # 面内Embedding
            inner_points = points + [[2.0, points[-1][1]]]  # 偷懒，直接把域最右边的点视为裂纹面的最后一个点，避免斜裂尖的一些处理麻烦
            self.inner_surfaces = multiInnerSurfaces(inner_points)
            # 裂尖Embedding
            # self.right_tip_line = LineSegement(points[-2],points[-1])
            # print(points[-2],points[-1])
            self.right_tip = RELU2PSILine(points[-2], points[-1], tip='right')
            # 裂尖与裂纹面的分割线
            self.split_line = LineSegement.init_theta(points[-2], self.right_tip.Line.tangent_theta + np.pi / 2)

            # self.right_tip_left_psi = LineSegement.init_theta(points[-2],
            #                                 #加pi减pi/2#
            #                                     self.right_tip_line.tangent_theta+np.pi/2).levelset
        else:
            raise Exception('还未实现!')
        # self.a = self.inner_surfaces.a + self.right_tip_line.length

        self.set_Heaviside(0)

    # def right_tip_basis(self , xy) -> Tensor:
    #     left_sign = self.Heaviside(self.right_tip_left_psi(xy[...,0],xy[...,1])).unsqueeze(-1)
    #     tip_basis = self.right_tip.getBasis(xy)
    #     return left_sign * tip_basis

    def getBasis(self, xy) -> Tensor:
        '''判断是否位于折裂纹的裂尖部位'''
        left_H = self.Heaviside(self.split_line.levelset(xy[..., 0], xy[..., 1])).unsqueeze(-1)
        inner_basis = self.inner_surfaces.getBasis(xy).unsqueeze(-1) * (1 - left_H)
        right_tip = self.right_tip.getBasis(xy) * left_H
        # self.left_H = left_H
        # self.inner_basis = self.inner_surfaces.getBasis(xy).unsqueeze(-1)
        # self.right_tip1 = right_tip
        # self.right_tip_noleftH = self.right_tip.getBasis(xy)
        return inner_basis

    def set_ls(self, ls: float):
        '''便于求应力强度因子设置ls为定值'''

        self.right_tip.set_ls(ls)

    def restore_ls(self):
        self.right_tip.restore_ls()

class RELU2PSIPolyLine_local(EnrichBasis):
    def __init__(self, points: list[list], tip='both'):
        '''与上一个的不同点:仅在最前端的裂尖有1-0的变化,
            由于偷懒写法，因此使用前请注意板的尺寸！'''
        # self.surfaces=[]
        self.xy0 = points[0]
        self.xy1 = points[-1]
        # print(self.xy0,self.xy1)
        # left_beta = LineSegement(points[1],points[0]).tangent_theta
        # right_beta = LineSegement(points[-2],points[-1]).tangent_theta

        if tip == 'right':
            # 面内Embedding
            inner_points = points + [[2.0, points[-1][1]]]  # 偷懒，直接把域最右边的点视为裂纹面的最后一个点，避免斜裂尖的一些处理麻烦
            self.inner_surfaces = multiInnerSurfaces(inner_points)
            # 裂尖Embedding
            # self.right_tip_line = LineSegement(points[-2],points[-1])
            # print(points[-2],points[-1])
            self.right_tip = RELU2PSILine(points[-2], points[-1], tip='right')
            # 裂尖与裂纹面的分割线
            self.split_line = LineSegement.init_theta(points[-2], self.right_tip.Line.tangent_theta + np.pi / 2)

            # self.right_tip_left_psi = LineSegement.init_theta(points[-2],
            #                                 #加pi减pi/2#
            #                                     self.right_tip_line.tangent_theta+np.pi/2).levelset
        else:
            raise Exception('还未实现!')
        # self.a = self.inner_surfaces.a + self.right_tip_line.length

        self.set_Heaviside(0)

    # def right_tip_basis(self , xy) -> Tensor:
    #     left_sign = self.Heaviside(self.right_tip_left_psi(xy[...,0],xy[...,1])).unsqueeze(-1)
    #     tip_basis = self.right_tip.getBasis(xy)
    #     return left_sign * tip_basis

    def getBasis(self, xy) -> Tensor:
        '''判断是否位于折裂纹的裂尖部位'''
        left_H = self.Heaviside(self.split_line.levelset(xy[..., 0], xy[..., 1])).unsqueeze(-1)
        inner_basis = self.inner_surfaces.getBasis(xy).unsqueeze(-1) * (1 - left_H)
        right_tip = self.right_tip.getBasis(xy) * left_H
        # self.left_H = left_H
        # self.inner_basis = self.inner_surfaces.getBasis(xy).unsqueeze(-1)
        # self.right_tip1 = right_tip
        # self.right_tip_noleftH = self.right_tip.getBasis(xy)
        return right_tip

    def set_ls(self, ls: float):
        '''便于求应力强度因子设置ls为定值'''

        self.right_tip.set_ls(ls)

    def restore_ls(self):
        self.right_tip.restore_ls()



class RELU2PSI(EnrichBasis):
    def __init__(self, levelset,
                 xy0, xy1,
                 tip,
                 left_beta,
                 right_beta):
        '''和论文中表达式一样
            未对扩充坐标进行归一化'''
        super().__init__()
        self.levelset = levelset

        self.left_beta = left_beta
        self.right_beta = right_beta

        self.left_psi = LineSegement.init_theta(xy0, left_beta - np.pi / 2).levelset_dist
        self.right_psi = LineSegement.init_theta(xy1, right_beta + np.pi / 2).levelset_dist

        self.a = LineSegement(xy0, xy1).length
        self.norm = self.a ** 2
        # self.left_psi = LineSegement.init_theta(xy0,left_beta-np.pi/2).levelset
        # self.right_psi =  LineSegement.init_theta(xy1,right_beta+np.pi/2).levelset
        if tip == 'left':
            self.right_psi = self.one
        elif tip == 'right':
            self.left_psi = self.one
        elif tip == 'both':
            # self.norm /= 4

            self.norm = (self.norm / 4) ** 2
        else:
            raise Exception()

        self.getH = self.getH_standard

    def getPSI(self, xy):
        x = xy[..., 0];
        y = xy[..., 1]
        return F.relu(- self.left_psi(x, y) * self.right_psi(x, y)) ** 2

    def getH_standard(self, xy):
        x = xy[..., 0];
        y = xy[..., 1]
        ls = self.levelset(x, y)
        return self.get_crack_sign(ls)

    def get_crack_sign(self, ls):
        return self.sign(-ls)

    def getBasis(self, xy):
        H = self.getH(xy)
        PSI = self.getPSI(xy)
        # print(self.norm)
        # print(self.a)
        return (H * PSI).unsqueeze(-1) / self.norm

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