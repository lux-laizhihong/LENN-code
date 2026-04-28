import torch
import Elasticity2D
import NN
from Enrichment import extendAxisNet,CrackStepBasis,SQRTBasis,multiBasis,EnrichBasis
from Geometry import Geometry1D,LocalAxis
import numpy as np
from sklearn.linear_model import LinearRegression
import NodesGenerater
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def DispExpolation(model:Elasticity2D.PINN2D,
                   net:extendAxisNet,
                   crack_surface:Geometry1D,num,
                   basis:multiBasis,
                   crackStep_column_ind,
                   sqrt:SQRTBasis,
                   kappa,mu):
    '''SIF单位:MPa*sqrt(mm)'''
    x , y  = crack_surface.generate_random_points(num)
    x , y , xy = model._set_points(x,y)

    '''下表面位移直接求出来'''
    u_low,v_low = model.pred_uv(xy)
    '''上表面位移需要反一下裂纹面extension的符号'''
    extension = basis.getBasis(xy)
    extension[:,crackStep_column_ind] *= -1
    axis = torch.cat((xy,extension),dim = 1)
    output = net.infer(axis)
    u_up,v_up = model.hard_u(output[0].squeeze(-1),x,y) , model.hard_v(output[1].squeeze(-1),x,y)   

    delta_u = u_up - u_low
    delta_v = v_up - v_low

    r_sqrt = sqrt.getBasis(xy) * np.sqrt(1000)

    r = r_sqrt **2

    material_coefficient = mu/(kappa+1) * np.sqrt(np.pi*2)

    K1_bar = material_coefficient * delta_v.unsqueeze(-1) / r_sqrt
    K2_bar = material_coefficient * delta_u.unsqueeze(-1) / r_sqrt

    K1_bar = K1_bar.cpu().detach().numpy()
    K2_bar = K2_bar.cpu().detach().numpy()
    r      = r.cpu().detach().numpy()

    K1_model = LinearRegression()  # 创建一个回归分析对象
    # K1_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))  # 对x和y进行拟合
    K1_model.fit(r, K1_bar)  # 对x和y进行拟合
    K1 = K1_model.intercept_

    K2_model = LinearRegression()  # 创建一个回归分析对象
    K2_model.fit(r, K2_bar)  # 对x和y进行拟合
    K2 = K2_model.intercept_
    return K1,K2

def DispExpolation2(model:Elasticity2D.PINN2D,
                    crack_surface:Geometry1D,num,
                #    sqrt:SQRTBasis,
                    local_axis:LocalAxis,
                    kappa,mu):
    

    '''SIF单位:MPa*sqrt(mm)'''
    # x , y  = crack_surface.generate_random_points(num)
    x , y  = crack_surface.generate_linespace_points(num)
    x , y , xy = model._set_points(x,y)

    # print(x,y)
    # print(crack_surface.levelset(x,y))

    '''下表面位移直接求出来'''
    model.model.extendAxis.set_ls(-1.0)
    u_low,v_low = model.pred_uv(xy)
    '''上表面位移需要反一下裂纹面extension的符号'''
    model.model.extendAxis.set_ls(1.0)
    u_up,v_up = model.pred_uv(xy)
    model.model.extendAxis.restore_ls()


    delta_u = u_up - u_low
    delta_v = v_up - v_low

    # 坐标转换
    delta_u , delta_v = local_axis.cartesianVariableToLocal(delta_u,delta_v)


    # u_up,v_up = local_axis.cartesianVariableToLocal(u_up,v_up)
    # u_low,v_low = local_axis.cartesianVariableToLocal(u_low,v_low)
    # delta_u_1 = u_up - u_low
    # delta_v_1 = v_up - v_low
    # print(delta_u_1 - delta_u)
    # print(delta_v_1 - delta_v)
    
    # r_sqrt = sqrt.getBasis(xy) * np.sqrt(1000)

    r = local_axis.getR(x,y).unsqueeze(-1) * 1000
    r_sqrt = torch.sqrt(r)

    material_coefficient = mu/(kappa+1) * np.sqrt(np.pi*2)

    K1_bar = material_coefficient * delta_v.unsqueeze(-1) / r_sqrt
    K2_bar = material_coefficient * delta_u.unsqueeze(-1) / r_sqrt

    K1_bar = K1_bar.cpu().detach().numpy()
    K2_bar = K2_bar.cpu().detach().numpy()
    r      = r.cpu().detach().numpy()

    K1_model = LinearRegression()  # 创建一个回归分析对象
    # K1_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))  # 对x和y进行拟合
    K1_model.fit(r, K1_bar)  # 对x和y进行拟合
    K1 = K1_model.intercept_

    K2_model = LinearRegression()  # 创建一个回归分析对象
    K2_model.fit(r, K2_bar)  # 对x和y进行拟合
    K2 = K2_model.intercept_
    return K1,K2


def DispExpolation_homo(model:Elasticity2D.PINN2D,
                        basis:EnrichBasis,
                    crack_surface:Geometry1D,num,
                #    sqrt:SQRTBasis,
                    local_axis:LocalAxis,
                    kappa,mu):
    

    '''SIF单位:MPa*sqrt(mm)'''
    # x , y  = crack_surface.generate_random_points(num)
    x , y  = crack_surface.generate_linespace_points(num)
    x , y , xy = model._set_points(x,y)

    # print(x,y)
    # print(crack_surface.levelset(x,y))

    '''下表面位移直接求出来'''
    basis.set_ls(-1.0)
    u_low,v_low = model.pred_uv(xy)
    '''上表面位移需要反一下裂纹面extension的符号'''
    basis.set_ls(1.0)
    u_up,v_up = model.pred_uv(xy)
    basis.restore_ls()


    delta_u = u_up - u_low
    delta_v = v_up - v_low

    # 坐标转换
    delta_u , delta_v = local_axis.cartesianVariableToLocal(delta_u,delta_v)


    r = local_axis.getR(x,y).unsqueeze(-1) * 1000
    r_sqrt = torch.sqrt(r)

    material_coefficient = mu/(kappa+1) * np.sqrt(np.pi*2)

    K1_bar = material_coefficient * delta_v.unsqueeze(-1) / r_sqrt
    K2_bar = material_coefficient * delta_u.unsqueeze(-1) / r_sqrt

    K1_bar = K1_bar.cpu().detach().numpy()
    K2_bar = K2_bar.cpu().detach().numpy()
    r      = r.cpu().detach().numpy()

    K1_model = LinearRegression()  # 创建一个回归分析对象
    # K1_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))  # 对x和y进行拟合
    K1_model.fit(r, K1_bar)  # 对x和y进行拟合
    K1 = K1_model.intercept_

    K2_model = LinearRegression()  # 创建一个回归分析对象
    K2_model.fit(r, K2_bar)  # 对x和y进行拟合
    K2 = K2_model.intercept_
    return K1,K2


def DispExpolation_bimaterial(model:Elasticity2D.PINN2D,
                    basis:EnrichBasis,
                    crack_surface:Geometry1D,num,
                    local_axis:LocalAxis,
                    kappa_up,mu_up,kappa_low,mu_low):
    

    '''SIF单位:MPa*sqrt(mm)'''
    x , y  = crack_surface.generate_linespace_points(num)
    x , y , xy = model._set_points(x,y)

    # print(x,y)
    # print(crack_surface.levelset(x,y))

    '''下表面位移直接求出来'''
    basis.set_ls(-1.0)
    u_low,v_low = model.pred_uv(xy)
    '''上表面位移需要反一下裂纹面extension的符号'''
    basis.set_ls(1.0)
    u_up,v_up = model.pred_uv(xy)
    basis.restore_ls()


    delta_u = u_up - u_low
    delta_v = v_up - v_low

    # 坐标转换
    delta_u , delta_v = local_axis.cartesianVariableToLocal(delta_u,delta_v)
    delta_u = delta_u.unsqueeze(-1)
    delta_v = delta_v.unsqueeze(-1)
    # delta_u = torch.zeros_like(delta_v)


    r = local_axis.getR(x,y).unsqueeze(-1) * 1000
    # r = local_axis.getR(x,y).unsqueeze(-1)
    r_sqrt = torch.sqrt(r)

 
    eps = np.log( (kappa_up / mu_up + 1 / mu_low) / (kappa_low/ mu_low + 1 / mu_up) ) / (2*np.pi)
    # Q = eps * torch.log(r)    #按Q=eps*ln(r)
    Q = eps * torch.log(r/1000) #按Q=eps*ln(r/2a)

    C = 2 * np.cosh(eps * np.pi)  * np.sqrt(np.pi * 2) / (kappa_up / mu_up + 1 / mu_low + kappa_low/ mu_low + 1 / mu_up)

    cosQ = torch.cos(Q)
    sinQ = torch.sin(Q)


    # print(eps)
    # print(eps*cosQ)
    # print(sinQ)

    e1 = cosQ + 2 * eps * sinQ
    e2 = sinQ - 2 * eps * cosQ

    K1_bar = C * (delta_v * e1 + delta_u * e2) / r_sqrt 
    K2_bar = C * (delta_u * e1 - delta_v * e2) / r_sqrt

    K1_bar = K1_bar.cpu().detach().numpy()
    K2_bar = K2_bar.cpu().detach().numpy()

    r_sinQ = (r * sinQ).cpu().detach().numpy()
    r_cosQ = (r * cosQ).cpu().detach().numpy()
    r      = r.cpu().detach().numpy()

    

    K1_model = LinearRegression()  # 创建一个回归分析对象
    # K1_model.fit(x.reshape(-1, 1), y.reshape(-1, 1))  # 对x和y进行拟合
    # K1_model.fit(r, K1_bar)  # 对x和y进行拟合
    K1_model.fit(r, K1_bar)  # 对x和y进行拟合
    K1 = K1_model.intercept_

    K2_model = LinearRegression()  # 创建一个回归分析对象
    # K2_model.fit(r, K2_bar)  # 对x和y进行拟合
    K2_model.fit(r, K2_bar)  # 对x和y进行拟合
    K2 = K2_model.intercept_

    return K1,K2

def max_stress_theta(K1,K2):
    '''
    最大环向应力计算扩展角度
    '''
    
    '''环向应力极值方向,包含最大值与最小值'''
    # theta = np.arctan(np.array([
    #             (K1 + np.sqrt(K1**2+8*K2**2)) / (4*K2) ,
    #             (K1 - np.sqrt(K1**2+8*K2**2)) / (4*K2) ,
    #         ]))

    # print(np.sqrt(K1**4+8 * K1**2 * K2**2))
    # print(np.array([
    #     (3*K2**2 + np.sqrt(K1**4+8 * K1**2 * K2**2)) / (K1**2 + 9 * K2**2),
    #     (3*K2**2 - np.sqrt(K1**4+8 * K1**2 * K2**2)) / (K1**2 + 9 * K2**2)
    # ]))



    theta = np.arccos(np.array([
        (3*K2**2 + np.sqrt(K1**4+8 * K1**2 * K2**2)) / (K1**2 + 9 * K2**2),
        (3*K2**2 - np.sqrt(K1**4+8 * K1**2 * K2**2)) / (K1**2 + 9 * K2**2)
    ]))
    theta = theta[theta<np.pi/2]
    theta = np.concatenate((theta,-theta),0)
    # print(theta)
    
    '''计算环向应力'''
    stress_theta = np.cos(theta/2) * (K1*(1+np.cos(theta)) - 3*K2*np.sin(theta))
    # print(stress_theta)
    return theta[np.argmax(stress_theta)]
    # theta = -np.arccos(np.array([
    #     (3*K2**2 + np.sqrt(K1**4+8 * K1**2 * K2**2)) / (K1**2 + 9 * K2**2)
    # ]))
    
    # return theta

# print(max_stress_theta(143.5,53.4) * 180 / np.pi)
# class DispExpolationMethod:
#     def __init__(self,sqrt:SQRTBasis,
#                  kappa,mu) -> None:
#         '''SIF单位:MPa*sqrt(mm)'''
#         self.K1_model = LinearRegression()
#         self.K2_model = LinearRegression()
#         self.sqrt = sqrt
#         self.material_coefficient = mu/(kappa+1) * np.sqrt(np.pi*2)

#     def get_deltaUV(self,
#                     model:Elasticity2D.PINN2D,
#                     net:extendAxisNet,
#                     )

def get_delta_u(model:Elasticity2D.PINN2D,
                        basis:EnrichBasis,
                    crack_surface:Geometry1D,num,
                    local_axis:LocalAxis,
                    mu,kappa):
    '''SIF单位:MPa*sqrt(mm)'''
    # x , y  = crack_surface.generate_random_points(num)
    x , y  = crack_surface.generate_linespace_points(num)
    x , y , xy = model._set_points(x,y)

    # print(x,y)
    # print(crack_surface.levelset(x,y))

    '''下表面位移直接求出来'''
    basis.set_ls(-1.0)
    u_low,v_low = model.pred_uv(xy)
    '''上表面位移需要反一下裂纹面extension的符号'''
    basis.set_ls(1.0)
    u_up,v_up = model.pred_uv(xy)
    basis.restore_ls()


    delta_u = u_up - u_low
    delta_v = v_up - v_low

    # 坐标转换
    delta_u , delta_v = local_axis.cartesianVariableToLocal(delta_u,delta_v)

    r = local_axis.getR(x,y).unsqueeze(-1) * 1000
    # r = local_axis.getR(x,y).unsqueeze(-1)
    r_sqrt = torch.sqrt(r)

    material_coefficient = mu/(kappa+1) * np.sqrt(np.pi*2)

    K1_bar = material_coefficient * delta_v.unsqueeze(-1) / r_sqrt
    K2_bar = material_coefficient * delta_u.unsqueeze(-1) / r_sqrt

    K1_bar = K1_bar.cpu().detach().numpy()
    K2_bar = K2_bar.cpu().detach().numpy()
    r      = r.cpu().detach().numpy()

    # return r.cpu().detach().numpy() , delta_u.unsqueeze(-1).cpu().detach().numpy() , delta_v.unsqueeze(-1).cpu().detach().numpy()
    return r , K1_bar , K2_bar



def DispExpolation3(model:Elasticity2D.PINN2D,
                        basis:EnrichBasis,
                    crack_surface:Geometry1D,num,
                #    sqrt:SQRTBasis,
                    local_axis:LocalAxis,
                    kappa,mu):
    

    '''SIF单位:MPa*sqrt(mm)'''
    # x , y  = crack_surface.generate_random_points(num)
    x , y  = crack_surface.generate_linespace_points(num)
    x , y , xy = model._set_points(x,y)

    # print(x,y)
    # print(crack_surface.levelset(x,y))

    '''下表面位移直接求出来'''
    basis.set_ls(-1.0)
    u_low,v_low = model.pred_uv(xy)
    '''上表面位移需要反一下裂纹面extension的符号'''
    basis.set_ls(1.0)
    u_up,v_up = model.pred_uv(xy)
    basis.restore_ls()


    delta_u = u_up - u_low
    delta_v = v_up - v_low

    # 坐标转换
    delta_u , delta_v = local_axis.cartesianVariableToLocal(delta_u,delta_v)


    r = local_axis.getR(x,y).unsqueeze(-1) * 1000
    # r = local_axis.getR(x,y).unsqueeze(-1)
    r_sqrt = torch.sqrt(r)

    material_coefficient = mu/(kappa+1) * np.sqrt(np.pi*2)

    K1_bar = material_coefficient * delta_v.unsqueeze(-1) / r_sqrt
    K2_bar = material_coefficient * delta_u.unsqueeze(-1) / r_sqrt

    K1_bar = K1_bar.cpu().detach().numpy()
    K2_bar = K2_bar.cpu().detach().numpy()
    r      = r.cpu().detach().numpy()

    r_sqare = r**2
    r_sum = np.sum(r)

    r_K1 = r*K1_bar
    r_K2 = r*K2_bar

    r_K1_sum = np.sum(r*K1_bar)
    r_K2_sum = np.sum(r*K2_bar)

    K1_sum = np.sum(K1_bar)
    K2_sum = np.sum(K2_bar)

    r_sqare_sum = np.sum(r_sqare)

    K1 = (r_sum * r_K1_sum - r_sqare_sum * K1_sum) / (r_sum**2 - num * r_sum)
    K2 = (r_sum * r_K2_sum - r_sqare_sum * K2_sum) / (r_sum**2 - num * r_sum)

    return K1,K2


'''计算超弹性'''

def DispExpolationHyper(model:Elasticity2D.PINN2D,
                        basis:EnrichBasis,
                    crack_surface:Geometry1D,num,
                    local_axis:LocalAxis,):
    

    '''
        返回p1,p2,单位sqrt(m)
    '''
    x , y  = crack_surface.generate_linespace_points(num)
    x , y , xy = model._set_points(x,y)

    '''u的单位是m'''

    '''下表面位移直接求出来'''
    basis.set_ls(-1.0)
    u_low,v_low = model.pred_uv(xy)
    '''上表面位移需要反一下裂纹面extension的符号'''
    basis.set_ls(1.0)
    u_up,v_up = model.pred_uv(xy)
    basis.restore_ls()


    delta_u = u_up - u_low
    delta_v = v_up - v_low

    # 坐标转换
    delta_u , delta_v = local_axis.cartesianVariableToLocal(delta_u,delta_v)

    # 单位为m
    r = local_axis.getR(x,y).unsqueeze(-1)
    r_sqrt = torch.sqrt(r)

    p2_bar = delta_v.unsqueeze(-1) / r_sqrt
    p1_bar = delta_u.unsqueeze(-1) / r_sqrt
    # p1_bar = delta_v.unsqueeze(-1)
    # p2_bar = delta_u.unsqueeze(-1)

    p1_bar = p1_bar.cpu().detach().numpy()
    p2_bar = p2_bar.cpu().detach().numpy()
    r_sqrt = r_sqrt.cpu().detach().numpy()
    r = r.cpu().detach().numpy()

    # r_3_2 = r.cpu().detach().numpy() * r_sqrt
    # basis = np.hstack((r_sqrt,1/r_sqrt))
    # print(p1_bar/2)
    # print(p2_bar/2)
    # print(r_sqrt)


    p1_model = LinearRegression()  # 创建一个回归分析对象
    p1_model.fit(r_sqrt, p1_bar)  # 对x和y进行拟合
    # p1_model.fit(r, p1_bar)  # 对x和y进行拟合
    # p1_model.fit(basis, p1_bar)  # 对x和y进行拟合
    p1 = p1_model.intercept_/2
    # p1 = p1_model.coef_/2

    p2_model = LinearRegression()  # 创建一个回归分析对象
    p2_model.fit(r_sqrt, p2_bar)  # 对x和y进行拟合
    # p2_model.fit(r, p2_bar)  # 对x和y进行拟合
    p2 = p2_model.intercept_/2
    return p1,p2


import Geometry
import Integral
from Elasticity2D import DEM2D_2
from get_grad import get_grad




class IIM_circle2D:
    def __init__(self,r0,
                local_axis:LocalAxis,
                num:list[int],
                int_type='contour'):
        '''xy表示全局坐标,x12表示局部坐标'''
        
        self.local_axis = local_axis

        self.dx_dx_local = [local_axis.cos , -local_axis.sin]
        self.dy_dx_local = [local_axis.sin , local_axis.cos]

        # self.dx_dx2 = -local_axis.sin
        # self.dy_dx2 = local_axis.cos

        self.r0 = r0
        self.theta_init = self.local_axis.beta - torch.pi
        delta = 1e-5

        self.local_theta = torch.linspace(start = - torch.pi + delta, end = torch.pi - delta,steps=num[0])
        self.theta = self.local_theta + self.local_axis.beta

        self.int_type = int_type
        if self.int_type == 'contour':
            self.integral = self.arc_integral
            self.x , self.y = self.local_axis.polarToCartesian(self.r0,self.local_theta)

        elif self.int_type == 'domain':
            self.r = torch.linspace(start = delta * 2, end = self.r0 - delta ,steps=num[1])
            self.local_theta,self.r = NodesGenerater.meshgirdFromXY(self.local_theta,self.r)
            self.num = num
            self.integral = self.domain_integral
            self.x , self.y = self.local_axis.polarToCartesian(self.r,self.local_theta)


        else: raise Exception('error!')

        
        self.x , self.y = self.x.float().requires_grad_() , self.y.float().requires_grad_()
        self.x, self.y = self.x.float().requires_grad_().to(device), self.y.float().requires_grad_().to(device)
        self.xy = torch.stack([self.x,self.y],dim=1)

        if self.int_type == 'domain':
            self.set_g()

        # 外法线方向
        # normal = torch.vstack((self.x - self.local_axis.x0, self.y - self.local_axis.y0)) 
        # self.unit_normal = normal / torch.norm(normal,dim=0)
        self.nx = torch.cos(self.theta).to(device)
        self.ny = torch.sin(self.theta).to(device)

        self.n1 = torch.cos(self.local_theta).to(device)
        self.n2 = torch.sin(self.local_theta).to(device)

    def set_g(self):
        r , theta = self.local_axis.cartesianToPolar(self.x,self.y)
        self.g = 1 - r / self.r0
        self.dg_dx1 , self.dg_dx2 = self.get_local_derivative(self.g,self.x,self.y)
        # x1 , x2 = self.local_axis.cartesianToLocal(self.x,self.y)
        # self.dg_dx1 , self.dg_dx2 = -x1 / (self.r0 * r) , -x2 / (self.r0 * r)

    def arc_integral(self,f):
        self.local_theta = self.local_theta.to(device)
        return Integral.trapz1D(f*self.r0,self.local_theta)
    
    def domain_integral(self,f):
        self.local_theta = self.local_theta.to(device)
        self.r = self.r.to(device)
        return Integral.trapz2D(f*self.r,torch.stack([self.local_theta,self.r],dim=1),shape=self.num)
    
    def get_local_derivative(self,f,x,y):
        df_dx = get_grad(f,x)
        df_dy = get_grad(f,y)
        df_dx1 = self.transform_derivative(df_dx,df_dy,direction=1)
        df_dx2 = self.transform_derivative(df_dx,df_dy,direction=2)
        return df_dx1 , df_dx2
    
    def transform_derivative(self,df_dx,df_dy,direction=1):
        return df_dx * self.dx_dx_local[direction-1] + df_dy * self.dy_dx_local[direction-1]
    
    def get_du_dxi(self,u1,u2,x,y,i=1):
        '''注意输入位移单位必须为m'''
        # 求du_dxi                                              # 无量纲
        du1_dx = get_grad(u1,x)
        du1_dy = get_grad(u1,y)
        du2_dx = get_grad(u2,x)
        du2_dy = get_grad(u2,y)
        du1_dxi = self.transform_derivative(du1_dx,du1_dy,direction=i)
        du2_dxi = self.transform_derivative(du2_dx,du2_dy,direction=i)
        return du1_dxi , du2_dxi
    
    def compute_local_strain(self,u1,u2,x,y):
        du1aux_dx1 , du2aux_dx1 = self.get_du_dxi(u1,u2,x,y,i=1)
        du1aux_dx2 , du2aux_dx2 = self.get_du_dxi(u1,u2,x,y,i=2)

        e11 = du1aux_dx1
        e22 = du2aux_dx2
        e12 = du2aux_dx1 + du1aux_dx2

        return e11 , e22 , e12
    
    def get_traction_from_stress(self,s11,s22,s12,direction='local'):
        if direction == 'local':
            T1 = s11 * self.n1 + s12 * self.n2
            T2 = s12 * self.n1 + s22 * self.n2 
        elif direction == 'global':
            T1 = s11 * self.nx + s12 * self.ny
            T2 = s12 * self.nx + s22 * self.ny
        else: raise Exception('errorrrrrrrrrrrr!') 
        
        return T1 , T2

    def get_actual_field(self,model:DEM2D_2):
        # u,v,sxx,syy,sxy = model.infer(self.xy)
        # self.xy = self.xy.to(device)
        # self.y = self.y.to(device)
        # self.x = self.x.to(device)
        u,v = model.pred_uv(self.xy)

        u1 , u2 = self.local_axis.cartesianVariableToLocal(u,v)   # mm
        u1 , u2 = model.mm_to_m(u1,u2)                                  # m

        e11 , e22 , e12 = self.compute_local_strain(u1,u2,self.x,self.y)

        s11 , s22 , s12 = model.constitutive(e11 , e22 , e12)

        return u1 , u2 , e11, e22, e12 , s11 , s22 , s12
    
    def get_aux_field(self,kappa,mu,K1_aux,K2_aux):

        r , theta = self.local_axis.cartesianToPolar(self.x,self.y)       # r单位m
        r_sqrt_2pi = torch.sqrt(r / (2*torch.pi))
        cos = torch.cos(theta)
        cos_half = torch.cos(theta/2)
        sin_half = torch.sin(theta/2)
        u1aux = r_sqrt_2pi * (K1_aux * (kappa - cos) * cos_half / (2 * mu) \
                              + K2_aux * (2 + kappa + cos) * sin_half / (2 * mu))
        u2aux = r_sqrt_2pi * (K1_aux * (kappa - cos) * sin_half / (2 * mu) \
                              + K2_aux * (2 - kappa - cos) * cos_half / (2 * mu))
        #求应力场
        sqrt_2pir = torch.sqrt(r * (2*torch.pi))
        cos_3_half = torch.cos(3 * theta/2)
        sin_3_half = torch.sin(3 * theta/2)

        g11 = cos_half * (1 - sin_half*sin_3_half)
        g12 = -sin_half * (2 + cos_half*cos_3_half)

        g21 = cos_half * (1 + sin_half*sin_3_half)
        g22 = sin_half * cos_half * cos_3_half

        s11aux = (K1_aux * g11 + K2_aux * g12) / sqrt_2pir
        s22aux = (K1_aux * g21 + K2_aux * g22) / sqrt_2pir
        s12aux = (K1_aux * g22 + K2_aux * g11) / sqrt_2pir

        return u1aux , u2aux , s11aux , s22aux , s12aux
    
    def aux_field_J_integral(self,K1_aux,K2_aux,kappa,mu):
        '''测试构造的可能场的J积分'''

        u1aux , u2aux , s11aux , s22aux , s12aux = self.get_aux_field(kappa,mu,K1_aux,K2_aux)

        du1aux_dx1 , du2aux_dx1 = self.get_du_dxi(u1aux,u2aux,self.x,self.y,i=1)

        e11 , e22, e12 = self.compute_local_strain(u1aux,u2aux,self.x,self.y)

        T1aux , T2aux = self.get_traction_from_stress(s11aux , s22aux , s12aux,direction='local')

        w = 0.5 * (e11 * s11aux + e22 * s22aux + e12 * s12aux)

        f = w * self.n1 - T1aux * du1aux_dx1 - T2aux * du2aux_dx1

        return self.integral(f=f)*1000

    def J_integral(self, model: DEM2D_2):
        '''输出单位:MPa * mm'''

        u1, u2, e11, e22, e12, s11, s22, s12 = self.get_actual_field(model)
        du1_dx1, du2_dx1 = self.get_du_dxi(u1, u2, self.x, self.y, i=1)
        T1, T2 = self.get_traction_from_stress(s11, s22, s12, direction='local')

        w = 0.5 * (s11 * e11 + s22 * e22 + s12 * e12)

        f = w * self.n1 - T1 * du1_dx1 - T2 * du2_dx1

        return self.integral(f=f) * 1000
    def compute_K_J(self, model: DEM2D_2, E_prime):
        J1 = self.J_integral(model)
        K2 = torch.sqrt(E_prime * J1)

        J2 = self.J_integral(model)
        K1 = torch.sqrt(E_prime * J2)

        return K1, K2
    # def compute_K_J(self, model: DEM2D_2, E_prime):
    #     J1 = self.J_integral(model, direction='1')
    #     K2 = torch.sqrt(E_prime * J1)
    #
    #     J2 = self.J_integral(model, direction='2')
    #     K1 = torch.sqrt(E_prime * J2)
    #
    #     return K1, K2
    def I_integral(self,K1_aux,K2_aux,model:DEM2D_2,kappa,mu):




        u1 , u2 , e11, e22, e12 , s11 , s22 , s12 = self.get_actual_field(model)

        du1_dx1 , du2_dx1 = self.get_du_dxi(u1,u2,self.x,self.y,i=1)
        # 求面力
        T1 , T2 = self.get_traction_from_stress(s11,s22,s12,direction='local')
        
        u1aux , u2aux , s11aux , s22aux , s12aux = self.get_aux_field(kappa,mu,K1_aux,K2_aux)
        du1aux_dx1 , du2aux_dx1 = self.get_du_dxi(u1aux,u2aux,self.x,self.y,i=1)
        T1aux , T2aux = self.get_traction_from_stress(s11aux , s22aux , s12aux,direction='local')
        v = (3-kappa)/(1+kappa)
        E = 8*mu/(1+kappa)
        e11aux = (s11aux-v*s22aux)/E
        e22aux = (s22aux - v * s11aux) / E
        e12aux = s12aux/(E / (2 * (1 + v)))
        # w12 = (s11aux * e11 + s22aux * e22 + s12aux * e12+s11 * e11aux+s12 * e12aux+s22 * e22aux)/2
        w12 = s11aux*e11 + s22aux*e22 + s12aux*e12


        f = w12 * self.n1 - (T1 * du1aux_dx1 + T2 * du2aux_dx1) - (T1aux * du1_dx1 + T2aux * du2_dx1)

        return self.integral(f)


    def compute_K_IIM(self,model:DEM2D_2,kappa,mu,E_prime):
        '''IIM计算应力强度因子'''
        # model = model.todevice()
        I10 = self.I_integral(1.0,0.0,model,kappa,mu)
        K1 = I10 * E_prime / 2 * np.sqrt(1000)

        I01 = self.I_integral(0.0,1.0,model,kappa,mu)
        K2 = I01 * E_prime / 2 * np.sqrt(1000)

        return K1,K2
    def I_integral_data(self, K1_aux, K2_aux, model: DEM2D_2, kappa, mu,u_x,v_x,u_y,v_y):
        v = (3 - kappa) / (1 + kappa)
        E = 8 * mu / (1 + kappa)
        d11 = E/(1-v*v)
        d12 = E/(1-v*v)*v
        G = E/(2*(1+v))
        u_x = u_x
        v_x = v_x
        v_y = v_y
        u_y = u_y
        cosl = self.local_axis.cos
        sinl = self.local_axis.sin
        u_xp = u_x * cosl ** 2 + (u_y + v_x) * sinl * cosl + v_y * sinl ** 2
        v_xp = v_x * cosl ** 2 - (u_x - v_y) * sinl * cosl - u_y * sinl ** 2
        u_yp = u_y * cosl ** 2 - (u_x - v_y) * sinl * cosl - v_x * sinl ** 2
        v_yp = v_y * cosl ** 2 - (u_y + v_x) * sinl * cosl + u_x * sinl ** 2
        e11 = u_xp
        e22 = v_yp
        e12 = v_xp + u_yp
        s11 = e11*d11+e22*d12
        s22 = e11*d12+e22*d11
        s12 = e12*G



        # u1, u2, e11, e22, e12, s11, s22, s12 = self.get_actual_field(model)
        #
        # du1_dx1, du2_dx1 = self.get_du_dxi(u1, u2, self.x, self.y, i=1)



        # 求面力
        T1, T2 = self.get_traction_from_stress(s11, s22, s12, direction='local')

        u1aux, u2aux, s11aux, s22aux, s12aux = self.get_aux_field(kappa, mu, K1_aux, K2_aux)
        du1aux_dx1, du2aux_dx1 = self.get_du_dxi(u1aux, u2aux, self.x, self.y, i=1)
        T1aux, T2aux = self.get_traction_from_stress(s11aux, s22aux, s12aux, direction='local')

        e11aux = (s11aux - v * s22aux) / E
        e22aux = (s22aux - v * s11aux) / E
        e12aux = s12aux / (E / (2 * (1 + v)))
        # w12 = (s11aux * e11 + s22aux * e22 + s12aux * e12+s11 * e11aux+s12 * e12aux+s22 * e22aux)/2
        w12 = s11aux * e11 + s22aux * e22 + s12aux * e12

        f = w12 * self.n1 - (T1 * du1aux_dx1 + T2 * du2aux_dx1) - (T1aux * u_xp + T2aux * v_xp)

        return self.integral(f)
    def compute_K_IIM_data(self,model:DEM2D_2,kappa,mu,E_prime,u_x,v_x,u_y,v_y):
        '''IIM计算应力强度因子'''
        # model = model.todevice()
        I10 = self.I_integral_data(1.0,0.0,model,kappa,mu,u_x,v_x,u_y,v_y)
        K1 = I10 * E_prime / 2 * np.sqrt(1000)

        I01 = self.I_integral_data(0.0,1.0,model,kappa,mu,u_x,v_x,u_y,v_y)
        K2 = I01 * E_prime / 2 * np.sqrt(1000)

        return K1,K2
    def J_integral_data(self, model: DEM2D_2,kappa,mu,u_x,v_x,u_y,v_y):
        '''输出单位:MPa * mm'''
        v = (3 - kappa) / (1 + kappa)
        E = 8 * mu / (1 + kappa)
        d11 = E/(1-v*v)
        d12 = E/(1-v*v)*v
        G = E/(2*(1+v))
        u_x = u_x
        v_x = v_x
        v_y = v_y
        u_y = u_y
        cosl = self.local_axis.cos
        sinl = self.local_axis.sin
        u_xp = u_x * cosl ** 2 + (u_y + v_x) * sinl * cosl + v_y * sinl ** 2
        v_xp = v_x * cosl ** 2 - (u_x - v_y) * sinl * cosl - u_y * sinl ** 2
        u_yp = u_y * cosl ** 2 - (u_x - v_y) * sinl * cosl - v_x * sinl ** 2
        v_yp = v_y * cosl ** 2 - (u_y + v_x) * sinl * cosl + u_x * sinl ** 2
        e11 = u_xp
        e22 = v_yp
        e12 = v_xp + u_yp
        s11 = e11*d11+e22*d12
        s22 = e11*d12+e22*d11
        s12 = e12*G


        # u1, u2, e11, e22, e12, s11, s22, s12 = self.get_actual_field(model)
        # du1_dx1, du2_dx1 = self.get_du_dxi(u1, u2, self.x, self.y, i=1)
        T1, T2 = self.get_traction_from_stress(s11, s22, s12, direction='local')

        w = 0.5 * (s11 * e11 + s22 * e22 + s12 * e12)

        f = w * self.n1 - T1 * u_xp - T2 * v_xp

        return self.integral(f=f) * 1000


    
    # def compute_K_CIM(self,model:DEM2D_2,E_prime):








# from hyperelasticity2D import hyperelasticity2D
#
# class IIM_hyperelasticity(IIM_circle2D):
#     def __init__(self, r0, local_axis, num,int_type='contour'):
#         super().__init__(r0, local_axis, num,int_type=int_type)
#
#     def get_actual_field(self, model:hyperelasticity2D):
#
#         F_global = model.getF(self.xy)
#         F = self.local_axis.tensorTolocal(F_global)
#
#         W = model.constitutive(F)
#         P = get_grad(W,F)
#
#         return F , P
#
#     def get_aux_field(self,mu):
#
#         r , theta = self.local_axis.cartesianToPolar(self.x,self.y)       # r单位m
#
#         yaux = torch.sqrt(r) * torch.sin(theta/2)
#
#         dyaux_dx1 , dyaux_dx2 = self.get_local_derivative(yaux,self.x,self.y)
#
#         P1aux = mu * dyaux_dx1
#         P2aux = mu * dyaux_dx2
#
#         return yaux , P1aux , P2aux
#
#     def I_integral(self, mu , P1 , P2 , dy_dx1):
#
#         yaux , P1aux , P2aux = self.get_aux_field(mu)
#
#         dyaux_dx1 , dyaux_dx2 = self.get_local_derivative(yaux,self.x,self.y)
#
#         w12 = P1 * dyaux_dx1 + P2 * dyaux_dx2
#
#         if self.int_type == 'contour':
#             T = P1 * self.n1 + P2 * self.n2
#             Taux = P1aux * self.n1 + P2aux * self.n2
#             f = w12 * self.n1 - T * dyaux_dx1 - Taux *dy_dx1
#
#         elif self.int_type == 'domain':
#             wext1 = P1 * dyaux_dx1 + P1aux * dy_dx1
#             wext2 = P2 * dyaux_dx1 + P2aux * dy_dx1
#
#             f = wext1 * self.dg_dx1 + wext2 * self.dg_dx2 - w12 * self.dg_dx1
#
#             # T = P1 * self.dg_dx1 + P2 * self.dg_dx2
#             # Taux = P1aux * self.dg_dx1 + P2aux * self.dg_dx2
#             # f = T * dyaux_dx1 + Taux * dy_dx1 - w12 * self.dg_dx1
#
#         return self.integral(f)
#
#     def aux_field_J_integral(self, mu):
#         yaux , P1aux , P2aux = self.get_aux_field(mu)
#         dyaux_dx1 , dyaux_dx2 = self.get_local_derivative(yaux,self.x,self.y)
#
#         return self.J1_integral(dyaux_dx1,dyaux_dx2,P1aux,P2aux,mu)
#
#     def compute_K_IIM(self, model:hyperelasticity2D, mu):
#
#         F , P = self.get_actual_field(model)
#
#         dy1_dx1 , dy2_dx1 = F[:,0,0] , F[:,1,0]
#
#         P11 , P12 , P21 , P22 = P[:,0,0] , P[:,0,1], P[:,1,0] , P[:,1,1]
#
#         I_y1 = self.I_integral(mu, P11 , P12 , dy1_dx1)
#         p1 = I_y1 * 2 / (torch.pi * mu)
#
#         I_y2 = self.I_integral(mu, P21 , P22 , dy2_dx1)
#         p2 = I_y2 * 2 / (torch.pi * mu)
#
#         return p1 , p2
#
#     def compute_K_J(self, model:hyperelasticity2D):
#
#         F , P = self.get_actual_field(model)
#
#         dy1_dx1 , dy1_dx2 , dy2_dx1 , dy2_dx2 = F[:,0,0] , F[:,0,1] , F[:,1,0] ,  F[:,1,1]
#
#         P11 , P12 , P21 , P22 = P[:,0,0] , P[:,0,1], P[:,1,0] , P[:,1,1]
#
#         J1_1 = self.J1_integral(dy1_dx1,dy1_dx2,P11,P12,model.G)
#         p1 = torch.sqrt(J1_1 * 4 / (model.G * torch.pi))
#
#         J1_2 = self.J1_integral(dy2_dx1,dy2_dx2,P21,P22,model.G)
#         p2 = torch.sqrt(J1_2 * 4 / (model.G * torch.pi))
#
#         return p1 , p2
#
#
#
#     def J1_integral(self, dy_dx1 , dy_dx2 , P1 , P2 , mu):
#         W = 0.5 * mu * (dy_dx1**2 + dy_dx2**2)
#
#         if self.int_type == 'contour':
#
#             Taux = P1 * self.n1 + P2 * self.n2
#             f = W * self.n1 - Taux * dy_dx1
#
#         elif self.int_type == 'domain':
#             Taux = P1 * self.dg_dx1 + P2 * self.dg_dx2
#             f = Taux * dy_dx1 - W * self.dg_dx1
#
#         return self.integral(f)
#
#
#
#     def J_integral(self, model:hyperelasticity2D):
#
#         W = model.get_energy_density(self.xy)
#
#         F , P = self.get_actual_field(model)
#
#         dy1_dx1 , dy2_dx1 = F[:,0,0] , F[:,1,0]
#
#         P11 , P12 , P21 , P22 = P[:,0,0] , P[:,0,1], P[:,1,0] , P[:,1,1]
#
#         if self.int_type == 'contour':
#             T1 = P11 * self.n1 + P12 * self.n2
#             T2 = P21 * self.n1 + P22 * self.n2
#             f = W * self.n1 - T1 * dy1_dx1 - T2 * dy2_dx1
#
#         elif self.int_type == 'domain':
#             T1 = P11 * self.dg_dx1 + P12 * self.dg_dx2
#             T2 = P21 * self.dg_dx1 + P22 * self.dg_dx2
#             f = T1 * dy1_dx1 + T2 * dy2_dx1 - W * self.dg_dx1
#
#         return self.integral(f)
#
#
#
#
#
#
#
#     # def compute_K_J(self, model):

        


    


        

        
# class IIM_rect:
#     '''矩形积分路径,下表面到上表面'''
#     def __init__(self,kappa,mu,Geometry:Geometry.MultiSegement1D,
#                  nodes_num:list[int],crack_axis:LocalAxis) -> None:

#         self.kappa = kappa
#         self.mu = mu
#         self.Geometry = Geometry
#         self.nodes_num = nodes_num
#         self.crack_axis = crack_axis
#         # 裂尖和x轴的夹角
#         self.beta=self.crack_axis.beta
    
#     def generate_nodes(self):
#         for i in range(5):
#             line_temp = self.Geometry.geometries[i]
#             x , y = line_temp.generate_linespace_points(self.nodes_num[i])
#             # 点的法线和x轴的夹角
#             theta = line_temp.get_normal_theta(x,y)
#             theta - self.beta    



# class IIM_circle:

#     def __init__(self,r0,
#                 # crack_surface:Geometry.LineSegement,
#                 local_axis:LocalAxis,
#                 num:int):
#         '''xy表示全局坐标,x12表示局部坐标'''
        
#         self.local_axis = local_axis

#         self.dx_dx1 = local_axis.cos
#         self.dy_dx1 = local_axis.sin

#         self.dx_dx2 = -local_axis.sin
#         self.dy_dx2 = local_axis.cos

#         self.r0 = r0
#         self.theta_init = self.local_axis.beta - torch.pi
#         delta = 1e-5

#         self.local_theta = torch.linspace(start = - torch.pi + delta, end = torch.pi - delta,steps=num)
#         self.theta = self.local_theta + self.local_axis.beta
#         self.x , self.y = self.local_axis.polarToCartesian(self.r0,self.local_theta)
#         self.x , self.y = self.x.float().requires_grad_() , self.y.float().requires_grad_()
#         self.xy = torch.stack([self.x,self.y],dim=1)

#         # 外法线方向
#         # normal = torch.vstack((self.x - self.local_axis.x0, self.y - self.local_axis.y0)) 
#         # self.unit_normal = normal / torch.norm(normal,dim=0)
#         self.nx = torch.cos(self.theta)
#         self.ny = torch.sin(self.theta)

#         self.n1 = torch.cos(self.local_theta)
#         self.n2 = torch.sin(self.local_theta)      
#         # self.x_init, self.y_init = crack_surface.clamp(dist2=r0)

#     def arc_integral(self,f,theta):
#         return Integral.trapz1D(f*self.r0,theta)
    
#     def J_integral(self,model:DEM2D_2):
#         '''输出单位:MPa * mm'''

#         w = model.get_energy_density(self.xy)                   # MPa
        
#         du1_dx1 , du2_dx1 , T1, T2 = self.get_actual_field(model)

#         f = w * self.n1 - T1 * du1_dx1 - T2 * du2_dx1

#         return self.arc_integral(f=f,theta=self.local_theta) * 1000
    
#     def get_actual_field(self,model:DEM2D_2):
#         u,v,sxx,syy,sxy = model.infer(self.xy)

#         u1 , u2 = self.local_axis.cartesianVariableToLocal(u,v)         # mm
#         u1 , u2 = model.mm_to_m(u1,u2)                          # m

#         # 求du_dx1                                              # 无量纲
#         du1_dxy = get_grad(u1,self.xy)
#         du2_dxy = get_grad(u2,self.xy)
#         du1_dx , du1_dy = du1_dxy[...,0] , du1_dxy[...,1]
#         du2_dx , du2_dy = du2_dxy[...,0] , du2_dxy[...,1]
#         du1_dx1 = du1_dx * self.dx_dx1 + du1_dy * self.dy_dx1
#         du2_dx1 = du2_dx * self.dx_dx1 + du2_dy * self.dy_dx1

#         # 求面力并转换至局部坐标系
#         Tx = sxx * self.nx + sxy * self.ny
#         Ty = sxy * self.nx + syy * self.ny
#         T1, T2 = self.local_axis.cartesianVariableToLocal(Tx,Ty)        

#         return du1_dx1 , du2_dx1 , T1, T2
    
#     def get_aux_field(self,kappa,mu,K1_aux,K2_aux):

#         r , theta = self.local_axis.cartesianToPolar(self.x,self.y)       # r单位m

#         cos = torch.cos(theta)

#         r_sqrt_2pi = torch.sqrt(r / (2*torch.pi))
#         cos = torch.cos(theta)
#         cos_half = torch.cos(theta/2)
#         sin_half = torch.sin(theta/2)

#         u1aux = r_sqrt_2pi * (K1_aux * (kappa - cos) * cos_half / (2 * mu) \
#                               + K2_aux * (2 + kappa + cos) * sin_half / (2 * mu))
#         u2aux = r_sqrt_2pi * (K1_aux * (kappa - cos) * sin_half / (2 * mu) \
#                               + K2_aux * (2 - kappa - cos) * cos_half / (2 * mu))
        
#         # 求du_dx1                                              # 无量纲
#         du1aux_dx = get_grad(u1aux,self.x)
#         du1aux_dy = get_grad(u1aux,self.y)
#         du2aux_dx = get_grad(u2aux,self.x)
#         du2aux_dy = get_grad(u2aux,self.y)

#         du1aux_dx1 = du1aux_dx * self.dx_dx1 + du1aux_dy * self.dy_dx1
#         du2aux_dx1 = du2aux_dx * self.dx_dx1 + du2aux_dy * self.dy_dx1

#         #求应力场
#         sqrt_2pir = torch.sqrt(r * (2*torch.pi))
#         cos_3_half = torch.cos(3 * theta/2)
#         sin_3_half = torch.sin(3 * theta/2)

#         g11 = cos_half * (1 - sin_half*sin_3_half)
#         g12 = -sin_half * (2 + cos_half*cos_3_half)

#         g21 = cos_half * (1 + sin_half*sin_3_half)
#         g22 = sin_half * cos_half * cos_3_half

#         s11aux = (K1_aux * g11 + K2_aux * g12) / sqrt_2pir
#         s22aux = (K1_aux * g21 + K2_aux * g22) / sqrt_2pir
#         s12aux = (K1_aux * g22 + K2_aux * g11) / sqrt_2pir

#         # n1 , n2 = torch.cos(self.local_theta), torch.sin(self.local_theta)

#         return du1aux_dx1 , du2aux_dx1 , s11aux , s22aux , s12aux
    
#     def aux_field_J_integral(self,K1_aux,K2_aux,kappa,mu):
#         '''测试构造的可能场的J积分'''
#         r , theta = self.local_axis.cartesianToPolar(self.x,self.y)       # r单位m

#         cos = torch.cos(theta)

#         r_sqrt_2pi = torch.sqrt(r / (2*torch.pi))
#         cos = torch.cos(theta)
#         cos_half = torch.cos(theta/2)
#         sin_half = torch.sin(theta/2)

#         u1aux = r_sqrt_2pi * (K1_aux * (kappa - cos) * cos_half / (2 * mu) \
#                               + K2_aux * (2 + kappa + cos) * sin_half / (2 * mu))
#         u2aux = r_sqrt_2pi * (K1_aux * (kappa - cos) * sin_half / (2 * mu) \
#                               + K2_aux * (2 - kappa - cos) * cos_half / (2 * mu))
        
#         # 求du_dx1                                              # 无量纲
#         du1aux_dx = get_grad(u1aux,self.x)
#         du1aux_dy = get_grad(u1aux,self.y)
#         du2aux_dx = get_grad(u2aux,self.x)
#         du2aux_dy = get_grad(u2aux,self.y)

#         du1aux_dx1 = du1aux_dx * self.dx_dx1 + du1aux_dy * self.dy_dx1
#         du2aux_dx1 = du2aux_dx * self.dx_dx1 + du2aux_dy * self.dy_dx1

#         du1aux_dx2 = du1aux_dx * self.dx_dx2 + du1aux_dy * self.dy_dx2
#         du2aux_dx2 = du2aux_dx * self.dx_dx2 + du2aux_dy * self.dy_dx2

#         e11 = du1aux_dx1
#         e22 = du2aux_dx2
#         e12 = du2aux_dx1 + du1aux_dx2

#         #求应力场
#         sqrt_2pir = torch.sqrt(r * (2*torch.pi))
#         cos_3_half = torch.cos(3 * theta/2)
#         sin_3_half = torch.sin(3 * theta/2)

#         g11 = cos_half * (1 - sin_half*sin_3_half)
#         g12 = -sin_half * (2 + cos_half*cos_3_half)

#         g21 = cos_half * (1 + sin_half*sin_3_half)
#         g22 = sin_half * cos_half * cos_3_half

#         s11aux = (K1_aux * g11 + K2_aux * g12) / sqrt_2pir
#         s22aux = (K1_aux * g21 + K2_aux * g22) / sqrt_2pir
#         s12aux = (K1_aux * g22 + K2_aux * g11) / sqrt_2pir

#         # du1aux_dx1,du2aux_dx1,s11aux,s22aux,s12aux = self.get_aux_field(kappa,mu,K1_aux,K2_aux)
#         T1aux = s11aux * self.n1 + s12aux * self.n2
#         T2aux = s12aux * self.n1 + s22aux * self.n2   

#         w = 0.5 * (e11 * s11aux + e22 * s22aux + e12 * s12aux)

#         f = w * self.n1 - T1aux * du1aux_dx1 - T2aux * du2aux_dx1

#         return self.arc_integral(f=f,theta=self.local_theta)


    
#     def I_integral(self,K1_aux,K2_aux,model:DEM2D_2,kappa,mu):

#         u,v = model.pred_uv(self.xy)
#         eXX,eYY,eXY = model.compute_Strain(u,v,self.xy)
#         e = self.local_axis.tensorTolocal((torch.stack([eXX,0.5*eXY,
#                                                         0.5*eXY,eYY],dim=1)).reshape(-1, 2, 2))
#         e11,e22,e12 = e[:,0,0],e[:,1,1],e[:,0,1]*2
#         # e11,e22,e12 = eXX,eYY,eXY
        
#         du1aux_dx1,du2aux_dx1,s11aux,s22aux,s12aux = self.get_aux_field(kappa,mu,K1_aux,K2_aux)
#         T1aux = s11aux * self.n1 + s12aux * self.n2
#         T2aux = s12aux * self.n1 + s22aux * self.n2     

#         du1_dx1 , du2_dx1 , T1, T2 = self.get_actual_field(model)   

#         w12 = s11aux*e11 + s22aux*e22 + s12aux*e12

#         f = w12 * self.n1 - (T1 * du1aux_dx1 + T2 * du2aux_dx1) - (T1aux * du1_dx1 + T2aux * du2_dx1)

#         return self.arc_integral(f,self.local_theta)
    
#     def compute_K(self,model:DEM2D_2,kappa,mu,E_prime):
#         '''IIM计算应力强度因子'''
#         I10 = self.I_integral(1.0,0.0,model,kappa,mu)
#         K1 = I10 * E_prime / 2 * np.sqrt(1000)

#         I01 = self.I_integral(0.0,1.0,model,kappa,mu)
#         K2 = I01 * E_prime / 2 * np.sqrt(1000)

#         return K1,K2

















if __name__ == '__main__':

    # print('hello')
    r0 = 2.0

    # analytical = torch.pi * r0 * 2
    analytical = torch.pi * r0**2

    # arc = IIM_circle2D(kappa=0.0,mu=0.0,r0=r0,local_axis=LocalAxis(x0=0.0,y0=0.0,beta=np.pi/4),num=200)

    # import matplotlib.pyplot as plt

    # plt.plot(arc.theta,arc.nx)
    # plt.show()

    domain = IIM_circle2D(r0=r0,local_axis=LocalAxis(x0=0.0,y0=0.0,beta=np.pi/4),num=[200,100],int_type='domain')

    numerical = domain.integral(torch.ones_like(domain.x))

    print(analytical)
    print(numerical)




    

        



