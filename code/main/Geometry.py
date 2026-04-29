from typing import Any

from shapely.geometry import Point, Polygon, MultiPoint
import torch
import numpy as np
from stats import Stats2D
import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Integral import montecarlo,trapz1D,trapz2D,simps1D,simps2D


class LocalAxis:
    def to_tensor(self,x):
        if torch.is_tensor(x):
            return x
        else:
            return torch.tensor(x)

    def __init__(self,x0,y0,beta=0.0):
        '''使用XFEM的enrichment'''
        self.x0 = self.to_tensor(x0)
        self.y0 = self.to_tensor(y0)
        self.beta = self.to_tensor(beta)
        self.cos = torch.cos(self.beta)
        self.sin = torch.sin(self.beta)

    def cartesianToPolar(self,x,y):
        '''整体直角坐标转换为裂尖的局部极坐标'''
        r = self.getR(x,y)
        theta = self.getLocalTheta(x,y)
        #注意，此处弧度为局部坐标系x轴（裂尖切线方向）与输入坐标之间的夹角
        return r,theta
    # def getR(self,x,y):
    #     return torch.sqrt( (x-self.x0) ** 2 + (y-self.y0) ** 2 )
    def getR(self, x, y):
        return torch.norm(torch.stack((x - self.x0, y - self.y0), dim=0), dim=0)
    def getRm(self,x,y):
        return torch.sqrt( (x-self.x0) ** 2 + (y-self.y0) ** 2 +1e-6)
    def  getR2(self,x,y):
        return (x - self.x0) ** 2 + (y - self.y0) ** 2

    def getLocalTheta(self,x,y):
        local_x,local_y = self.cartesianToLocal(x,y)
        theta = torch.arctan2(local_y,local_x)
        #注意，此处弧度为局部坐标系x轴（裂尖切线方向）与输入坐标之间的夹角
        return theta
    def cartesianToLocal(self,x,y):
        '''相对于裂纹尖端方向的xy坐标'''
        # local_x = self.cos * (x-self.x0) + self.sin * (y-self.y0)
        # local_y = - self.sin * (x-self.x0) + self.cos * (y-self.y0)
        return self.cartesianVariableToLocal(x-self.x0 , y-self.y0)

    def cartesianVariableToLocal(self,f_x,f_y):
        '''将直角坐标的物理量转换为局部坐标'''
        f_local_x = self.cos * f_x + self.sin * f_y
        f_local_y = - self.sin * f_x + self.cos * f_y
        return f_local_x , f_local_y

    # def polarToCartesian(self,r,theta):
    #     '''相对于裂纹尖端方向的xy坐标'''
    #     local_x = r * torch.cos(theta)
    #     local_y = r * torch.sin(theta)
    #     '''坐标系旋转'''
    #     local_x = self.cos * local_x - self.sin * local_y
    #     local_y = self.sin * local_x + self.cos * local_y
    #     '''平移原点'''
    #     x = local_x + self.x0
    #     y = local_y + self.y0
    #     return x,y

    def polarToCartesian(self, r, theta):
        '''极坐标 → 局部直角坐标'''
        local_x = r * torch.cos(theta)
        local_y = r * torch.sin(theta)

        '''局部旋转 → 全局方向'''
        rot_x = self.cos * local_x - self.sin * local_y
        rot_y = self.sin * local_x + self.cos * local_y

        '''平移到全局坐标系'''
        x = rot_x + self.x0
        y = rot_y + self.y0

        return x, y


class Geometry1D:
    def to_tensor(self,x):
        if torch.is_tensor(x):
            return x
        else:
            return torch.tensor(x)

    def get_tangent_theta(self,x,y)->torch.Tensor:...
    def get_normal_theta(self,x,y)->torch.Tensor:
        '''线段法线方向与x轴的夹角'''
        return self.get_tangent_theta(x,y) + torch.pi/2
    
    def get_direction_cosine(self,x,y)->torch.Tensor:
        normal_theta = self.get_normal_theta(x,y)
        l_x = torch.cos(normal_theta)
        l_y = torch.cos(torch.pi/2 - normal_theta)
        return l_x,l_y

    def generate_random_points(self,num)->torch.Tensor:...
    def generate_linespace_points(self,num)->torch.Tensor:...
    def levelset(self,x,y):...
    def is_on_geometry(self,points,eps=1e-4):...
    
    def is_on_left(self, points):
        x = points[...,0]
        y = points[...,1]
        ls = self.levelset(x,y)
        return torch.where(ls < 0 , True, False)

class LineSegement(Geometry1D):
    def __init__(self,xy0,xy1) -> None:
        self.x0 = self.to_tensor(xy0[0])
        self.x1 = self.to_tensor(xy1[0])
        self.y0 = self.to_tensor(xy0[1])
        self.y1 = self.to_tensor(xy1[1])
        self.x_span = [self.x0,self.x1]
        self.y_span = [self.y0,self.y1]
        self.x_len = self.x1 - self.x0
        self.y_len = self.y1 - self.y0
        self.tangent_theta = torch.arctan2(self.y_len,self.x_len)
        self.A = self.y1 - self.y0
        self.B = self.x0 - self.x1
        self.C = self.x1 * self.y0 - self.x0 * self.y1
        self.AB2 = self.A **2 + self.B** 2
        self.length = torch.sqrt((self.x0 - self.x1)**2 + (self.y0 - self.y1)**2)


    @classmethod
    def init_theta(line,xy0,tan_beta):
        x1 = xy0[0] + np.cos(tan_beta)
        y1 = xy0[1] + np.sin(tan_beta)
        return line(xy0,[x1,y1])

    def levelset(self,x,y):
        return self.A * x + self.B * y + self.C

    def levelset_dist(self,x,y):
        ls = self.levelset(x,y)
        return ls / self.AB2
    
    def get_tangent_theta(self, x, y):
        return self.tangent_theta * torch.ones_like(x)
    
    def generate_random_points(self,num):
        random = np.random.rand(num)
        x = random * self.x_span[0].numpy() + (1 - random) * self.x_span[1].numpy()
        y = random * self.y_span[0].numpy() + (1 - random) * self.y_span[1].numpy()
        return torch.tensor(x) , torch.tensor(y)
    
    def generate_linespace_points(self, num) :
        x = torch.linspace(self.x_span[0],self.x_span[1],num)
        y = torch.linspace(self.y_span[0],self.y_span[1],num)
        return x,y

#    任意一个点到两个裂尖的距离减去裂纹长度？判断在不在裂纹上？
    def approx_dist(self,x,y):
        def dist(x1,x2,y1,y2):
            return torch.sqrt((x1 - x2)**2 + (y1 - y2)**2)    
         
        # dist_line = dist(self.x0,self.x1,self.y0,self.y1)
        dist_0 = dist(self.x0,x,self.y0,y)
        dist_1 = dist(self.x1,x,self.y1,y)   

        # return torch.abs(dist_0 + dist_1 - self.length)
        return (dist_0 + dist_1 - self.length)
    
    def is_on_geometry(self, points, eps=1e-4):

        x = points[...,0]
        y = points[...,1]

        return np.where( self.approx_dist(x,y)< eps, True, False)
    
    def clamp(self,ratio = None,
                   dist1 = None,
                   dist2 = None):
        '''
        根据比例返回线段上的点
        '''

        if ratio is not None:
            dist = ratio * self.length
        elif dist1 is not None:
            dist = dist1
        elif dist2 is not None:
            dist = self.length - dist2

        dist_x = dist * np.cos(self.tangent_theta)
        dist_y = dist * np.sin(self.tangent_theta)

        return self.x0 + dist_x, self.y0 + dist_y
        
    
class BiGeometry1D(Geometry1D):
    def __init__(self,Geo_list:list[Geometry1D],
                 divide_line:LineSegement) -> None:
        '''注意！两条线段必须首尾相连，
        第一条线段位于division左侧,因此需要注意divide_line方向'''
        super().__init__()
        self.geometries = Geo_list
        self.divide_line = divide_line
    
    def is_on_devision_left(self,points):
        ls = self.divide_line.is_on_left(points)
        return ls

    def get_tangent_theta(self,x,y)->torch.Tensor:
        points = torch.stack((x,y),dim=1)
        ls = self.is_on_devision_left(points)
        return torch.where (ls,
                            self.geometries[0].get_tangent_theta(x,y),
                            self.geometries[1].get_tangent_theta(x,y))
    
    def levelset(self,x,y)->torch.Tensor:
        points = torch.stack((x,y),dim=1)
        ls = self.is_on_devision_left(points)
        return torch.where (ls,
                            self.geometries[0].levelset(x,y),
                            self.geometries[1].levelset(x,y))



class MultiSegement1D(Geometry1D):
    def __init__(self,Geo_list:list[LineSegement]) -> None:
        '''注意！两条线段必须首尾相连，
        第一条线段位于division左侧,因此需要注意divide_line方向'''
        super().__init__()
        self.geometries = Geo_list

    
    def levelset(self,x,y)->torch.Tensor:
        # points = torch.stack((x,y),dim=1)
        levelsets = torch.stack(list(map(lambda ls: ls.levelset_dist(x,y),self.geometries)))
        distances = torch.stack(list(map(lambda d: d.approx_dist(x,y),self.geometries)))
        index = torch.argmin(torch.abs(distances),dim=0)
        # print(index.max())
        # ls =  levelsets[index, torch.arange(len(index))]
        ls =  levelsets[index, torch.arange(len(index))]
        return ls


class Circle:

    def __init__(self,x0,y0,r) -> None:
        self.local_axis = LocalAxis(x0,y0)
        self.r = r
        self.area = np.pi * r **2
    
    def dist(self,x,y):
        return self.local_axis.getR(x,y)

    def distm(self,x,y):
        return self.local_axis.getRm(x,y)

    def dist_2(self,x,y):
        return self.local_axis.getR2(x,y)

    def norm_dist(self,x,y):
        return self.dist(x,y) / self.r

    def norm_distm(self,x,y):
        return self.distm(x,y) / self.r

    def norm_dist_2(self,x,y):
        return self.dist_2(x,y) / (self.r*self.r)

    def levelset_dist(self,x,y):
        ls = self.levelset(x,y)
        return ls
    # #
    def levelset(self,x,y):
         return self.dist(x,y) - self.r
    # #

    # def levelset(self,x,y):
    #      return self.dist_2(x,y) - self.r*self.r

    # def levelset1(self,x,y):
    #     return abs(self.dist(x,y) - self.r)


    def is_in_geometry(self,x,y):
        return self.levelset(x,y) < 0

    def generate_random_points(self,num):
        random = torch.from_numpy(np.random.rand(num))
        theta = random * torch.pi * 2
        r = random * self.r
        x,y = self.local_axis.polarToCartesian(r,theta)
        return x,y
    
    def area_in_rect(self,left ,right ,bottom ,top):

        x_circle,y_circle = self.generate_random_points(10000)
        is_whole_in_rect = torch.all((x_circle > left) & (x_circle < right) & (y_circle > bottom) & (y_circle < top))
        if  is_whole_in_rect:
            return self.area
        else:

            '''蒙特卡洛积分计算矩形与圆的重叠面积'''
            square = (right - left) * (top - bottom)

            # 定义用于蒙特卡洛模拟的点的数量
            num_points = 100000000

            # 在正方形内部随机生成点
            x = torch.from_numpy(np.random.uniform(left, right, size=(num_points)))
            y = torch.from_numpy(np.random.uniform(bottom, top, size=(num_points)))
            # 计算落在圆形内部的点的数量
            # distances = np.sum(points**2, axis=1)

            mask = self.is_in_geometry(x,y)
            points_inside_circle = torch.sum(mask).numpy()

            

            # 估计重叠面积
            overlap_area = (points_inside_circle / num_points) * square
            return overlap_area

'注意 此处矩形的角度默认为0'
class Rectangle_Geometry:
    def __init__(self, xc, yc, h,w,n=4,alpha = 20) -> None:
        self.xc = xc
        self.yc = yc
        self.local_axis = LocalAxis(xc, yc)
        self.h = h
        self.w = w
        self.area = h*w
        self.n = n
        self.alpha = alpha

    def dist(self,x,y):
        left_dist = x-(self.xc-self.w/2)
        right_dist = x-(self.xc+self.w/2)
        up_dist = y-(self.yc+self.h/2)
        bottom_dist = y - (self.yc - self.h / 2)

        dist_w = left_dist*right_dist
        dist_h = up_dist*bottom_dist
        return dist_w,dist_h
# '这是相加+相乘levelset,感觉不连续'
#     def levelset(self,x,y):
#         dist_w,dist_h = self.dist(x,y)
#
#         return dist_w*dist_h
#     def levelset(self,x,y):
#         dist_w,dist_h = self.dist(x,y)
#         dist_plus = torch.relu(dist_h)+torch.relu(dist_w)
#         dist_mutiply = torch.relu(-dist_h)*torch.relu(-dist_w)
#         return dist_plus-dist_mutiply
# '这是乘积，可用于判断材料，但用做gamma效果可能不好'
#     def levelset(self,x,y):
#         dist_w,dist_h = self.dist(x,y)
#         dist1 = torch.relu(-dist_h*dist_w)
#         dist2 = torch.relu(dist_h*dist_w)
#         dist3 = torch.relu(-dist_h)*torch.relu(-dist_w)
#         return dist1+dist2-2*dist3
# '这是设置4次方，和圆的类型类似'
    def levelset(self, x, y):
        dx = (2 * (x - self.xc) / self.w) ** self.n
        dy = (2 * (y - self.yc) / self.h) ** self.n
        return dx + dy - 1

    # def levelset(self, x, y):
    #     dx = 2 * (x - self.xc) / self.w
    #     dy = 2 * (y - self.yc) / self.h
    #     soft_square = (1.0 / self.alpha) * torch.log(torch.exp(self.alpha * dx ** 2)
    #                                                  + torch.exp(self.alpha * dy ** 2))
    #     # return soft_square-1
    #     return torch.sqrt(soft_square)-1
    def is_in_geometry(self, x, y):
        return self.levelset(x, y) < 0

    # def generate_random_points(self, num):
    #     random = torch.from_numpy(np.random.rand(num))
    #     theta = random * torch.pi * 2
    #     r = random * self.r
    #     x, y = self.local_axis.polarToCartesian(r, theta)
    #     return x, y

    # def area_in_rect(self, left, right, bottom, top):
    #
    #     x_circle, y_circle = self.generate_random_points(10000)
    #     is_whole_in_rect = torch.all((x_circle > left) & (x_circle < right) & (y_circle > bottom) & (y_circle < top))
    #     if is_whole_in_rect:
    #         return self.area
    #     else:
    #
    #         '''蒙特卡洛积分计算矩形与圆的重叠面积'''
    #         square = (right - left) * (top - bottom)
    #
    #         # 定义用于蒙特卡洛模拟的点的数量
    #         num_points = 100000000
    #
    #         # 在正方形内部随机生成点
    #         x = torch.from_numpy(np.random.uniform(left, right, size=(num_points)))
    #         y = torch.from_numpy(np.random.uniform(bottom, top, size=(num_points)))
    #         # 计算落在圆形内部的点的数量
    #         # distances = np.sum(points**2, axis=1)
    #
    #         mask = self.is_in_geometry(x, y)
    #         points_inside_circle = torch.sum(mask).numpy()
    #
    #         # 估计重叠面积
    #         overlap_area = (points_inside_circle / num_points) * square
    #         return overlap_area














# import numpy as np
#
# import time
# # 测试torch.heaviside()的运行时间
# start = time.time()
#
# # 定义正方形和圆形的参数
# square = 4.0
# c = Circle(x0=1.0,y0=1.0,r=1.0)
#
# num = 1
#
# # for i in range(1):
#
#
#
# left = 0; right = 1; bottom = -1; top = 1
#
#
# # print('torch.heaviside()运行时间: ', end - start)
# print("Estimated overlap area:", c.area_in_rect(left,right,bottom,top))
#
# # 创建一个多边形对象
# polygon = Polygon([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
#
# # 创建一个点对象
# point = Point(2.0, 0.0)
# xy = np.array([[1.0,0.0],[0.0,0.5],[2.0,1.0]])
# multipoint = MultiPoint(np.array([[0.0,0.0],[0.5,0.5],[1.0,1.0]]))
#
# inner_index = [polygon.covers(Point(xy[i,0],xy[i,1])) for i in range(xy.shape[0])]
# print(inner_index)
# print(xy[inner_index])
# inner_index = [(Point(xy[i,0],xy[i,1])).within(polygon) for i in range(xy.shape[0])]
# print(inner_index)
# print(xy[inner_index])
# print(multipoint.within(polygon))
# # 将点对象添加到多边形对象中
# if point.within(polygon):
#     print("点在多边形内")
# else:
#     print("点不在多边形内")
