import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from Integral import montecarlo,trapz1D,trapz2D,simps1D,simps2D
import Geometry

from stats import Stats2D,Stats1D
import stats
from matplotlib.path import Path

def meshgirdFromXY(x,y):

    X,Y = torch.meshgrid(x, y)     
    X=X.reshape(-1)
    Y=Y.reshape(-1)
    # X = X.to(device)
    # X.requires_grad = True
    # Y = Y.to(device)
    # Y.requires_grad = True  
    return X,Y    


def genMeshNodes2D(xstart,xend,xnum,ystart,yend,ynum):
    '''生成规则网格排列的二维点'''
    x = torch.linspace(xstart,xend,xnum)
    y = torch.linspace(ystart,yend,ynum)

    return meshgirdFromXY(x,y)


def genRandomNodes2D(left, right, bottom, top, num):
    '''生成服从均匀分布的随机二维点'''
    random = np.random.rand(num)
    x = random * left + (1 - random) * right
    random = np.random.rand(num)
    y = random * bottom + (1 - random) * top
    return torch.stack([torch.tensor(x) , torch.tensor(y)],dim=1)


def AcceptanceSampling1D(num, stats:Stats1D)->torch.Tensor:
    # 计算概率密度函数的最大值
    x = np.linspace(stats.start, stats.end, 100)
    Z = stats.pdf(x)
    pdf_max = Z.max()


    # 初始化随机点数组
    points = np.zeros((num, 1))

    # 使用接受采样方法(Acceptance Sampling)生成随机点
    i = 0
    while i < num:
        # 在矩形内生成一个均匀分布的随机点
        x = np.random.uniform(stats.start, stats.end)
        
        # 计算该点的概率密度值
        p = stats.pdf(x)
        
        # 以概率p / pdf_max接受该点
        if np.random.uniform(0, pdf_max) < p:
            points[i] = x
            i += 1
    return torch.from_numpy(points).squeeze(-1)

def AcceptanceSampling2D(left, right, bottom, top, num, stats:Stats2D)->torch.Tensor:
    # 计算矩形内概率密度函数的最大值
    x = np.linspace(left, right, 100)
    y = np.linspace(bottom, top, 100)
    X, Y = np.meshgrid(x, y)
    Z = stats.pdf(X, Y)
    pdf_max = Z.max()


    # 初始化随机点数组
    points = np.zeros((num, 2))

    # 使用接受采样方法(Acceptance Sampling)生成随机点
    i = 0
    while i < num:
        # 在矩形内生成一个均匀分布的随机点
        x = np.random.uniform(left, right)
        y = np.random.uniform(bottom, top)
        
        # 计算该点的概率密度值
        p = stats.pdf(x, y)
        
        # 以概率p / pdf_max接受该点
        if np.random.uniform(0, pdf_max) < p:
            points[i] = [x, y]
            i += 1
    return torch.from_numpy(points)



def genUniformInnerDense2D(left, right, bottom, top,
                          inner_left, inner_right, inner_bottom, inner_top,
                          uniform_num,inner_dense_factor=0.5):
    '''点的总数: uniform_num * (1+inner_dense_factor)'''
    # total_area = (top - bottom) * (right - left)
    # inner_area = (inner_top - inner_bottom) * (inner_right - inner_left)
    # out_area = total_area - inner_area
#到这一步为止，还没有生成网格对吧
    # 均匀部分
    points_uniform = genRandomNodes2D(left, right, bottom, top, uniform_num)

    # 内部加密部分
    points_inner = genRandomNodes2D(inner_left, inner_right, inner_bottom, inner_top, int(uniform_num * inner_dense_factor))

    points = torch.cat((points_uniform,points_inner),dim=0)


    stat1 = Stats2D(stats.Uniform(left, right),stats.Uniform(bottom, top))
    stat2 = Stats2D(stats.Uniform(inner_left, inner_right),stats.Uniform(inner_bottom, inner_top))
    points_distribution = stat1 + inner_dense_factor * stat2
    points_pdf = points_distribution.pdf(points[:,0],points[:,1])

    return points,points_pdf


def genHeteroMeshNodesPDF(xstats:Stats1D,ystats:Stats1D,
                          x_num,y_num):
    '''从概率分布生成局部加密的网格点'''
    x,_ = torch.sort(AcceptanceSampling1D(x_num,xstats))
    y,_ = torch.sort(AcceptanceSampling1D(y_num,ystats))

    return meshgirdFromXY(x,y)

def genHeteroNodes1D(sep:list,num:list):
    '''从给定加密区域生成一维局部均匀分布点'''
    region_num = len(sep) - 1
    if region_num != len(num):
        raise Exception()
    '''前一段终点与后一段起点间距2e-3'''
    # node_set = [torch.linspace(sep[i]+9e-4,sep[i+1]-1e-3,num[i]) for i in range(region_num)]
    node_set = [torch.linspace(sep[i],sep[i+1],num[i]) for i in range(region_num)]
    return torch.cat(node_set,dim=-1)



def genHeteroNodes2D(xsep,xnum,ysep,ynum):
    x = genHeteroNodes1D(xsep,xnum)
    y = genHeteroNodes1D(ysep,ynum)
    return meshgirdFromXY(x,y)



def genHeteroTip2D(xstart,xend,
                    ystart,yend,
                    xTip,yTip,
                    x_dense_num,y_dense_num,
                    x_outer_num,y_outer_num,
                    x_inteval=0.2,y_inteval=0.2):
    
    def segmentTip(start,end,Tip,
                   dense_num,outer_num,
                   inteval):
        #  Tip是啥，裂纹坐标？
        # length = end - start - 2*inteval
        Tip_left = Tip - inteval
        Tip_right = Tip + inteval
        length = end - start - min([inteval,Tip - start]) - min([end - Tip,inteval])
        segments = [start]
        num = []
        if Tip_left - start > 0.01:
            segments.append(Tip_left)
            left_num = round(outer_num * (Tip_left - start)/length)
            num.append(left_num)
        else: left_num = 0
        # segments.append(Tip)
        num.append(dense_num)
        if end - Tip_right > 0.01:
            segments.append(Tip_right)
            num.append(outer_num-left_num)
        segments.append(end)

        return segments,num
    
    xsep,xnum = segmentTip(xstart,xend,xTip,x_dense_num,x_outer_num,x_inteval)

    ysep,ynum = segmentTip(ystart,yend,yTip,y_dense_num,y_outer_num,y_inteval)

    return meshgirdFromXY(genHeteroNodes1D(xsep,xnum),genHeteroNodes1D(ysep,ynum))


#弯弯绕绕这么多，本质是？
def FunctionSampling2D(function,num,refer_points,refer_pdf,alpha = 0.75):
    '''根据函数值大小撒点'''
    
    x_samples = refer_points[...,0];y_samples = refer_points[...,1]

    refer_num = len(refer_pdf)

    # 计算每个样本点的函数值
    values = function(x_samples,y_samples)


    # 计算每个样本点的权重
    # weights = 0.5 * (values / np.sum(values) + 1.0 / values.size)
    weights = (values / torch.sum(values)) * refer_num * refer_pdf
    weights = alpha * weights + (1 - alpha) * refer_pdf
    # weights = uniform_pdf * torch.ones_like(values)

    # 根据权重进行无放回抽样
    # indices = np.random.choice(range(len(x_samples)), size=num, replace=False, p=weights)
    indices = torch.multinomial(weights,num_samples=num,replacement=False)

    # 得到最终的采样结果
    x_samples = x_samples[indices].detach()
    y_samples = y_samples[indices].detach()
    pdf = weights[indices]

    return torch.stack((x_samples,y_samples),dim=1),pdf.detach().numpy()


def FunctionSamplingUniform2D(function,x_range,y_range,
                       num,background_num=1000000,alpha = 0.75):
    '''根据函数值大小撒点'''

    points = genRandomNodes2D(x_range[0],x_range[1],
                              y_range[0],y_range[1],
                              background_num)
    
    area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
    uniform_pdf = torch.ones_like(points[...,0]) / area

    return FunctionSampling2D(function,num,points,uniform_pdf,alpha)


def DeleteTips(points,pdf,axes:list[Geometry.LocalAxis],delta = 1e-2):
    for i,axis in enumerate(axes):
        r = axis.getR(points[...,0],points[...,1])
        points = points[r>delta]
        pdf = pdf[r>delta]
    return points,pdf
        



def genDenseCircles(left, right, bottom, top, outer_num,
                    circles:list[Geometry.Circle],
                    inner_num):
    '''加密多个裂尖周围的积分点'''

    points_x=[]
    points_y=[]
    points_pdf = []

    # total_circle_num = inner_num * len(circles)

    circles_area = 0.0
    total_circle_num = 0

    outer_points = genRandomNodes2D(left, right, bottom, top,
                num = int(outer_num * (right - left) * (top - bottom)/ ((right - left) * (top - bottom) - sum([circle.area for circle in circles]))))
    
    # 用于记录在圆内的外部点的索引
    mask_outer = torch.zeros_like(outer_points[...,0],dtype=torch.bool)
    for i,circle in enumerate(circles):

        # rtheta = genRandomNodes2D(circle.r * 1e-3, circle.r, 0, 2*np.pi, inner_num)
        rtheta = genRandomNodes2D(2e-4, circle.r, 0, 2*np.pi, inner_num)
        x,y = circle.local_axis.polarToCartesian(rtheta[...,0],rtheta[...,1])

        # 筛选掉不在矩形域内的点
        circle_area = circle.area_in_rect(left, right, bottom, top)
        circles_area += circle_area

        ind = torch.where((x > left) & (x < right) & (y > bottom) & (y < top))
        x = x[ind]
        y = y[ind]
        rtheta = rtheta[ind]

        points_x.append(x) ; points_y.append(y)

        total_circle_num += x.data.nelement()

        # points_pdf.append(torch.ones_like(x) / circle.area)
        # points_pdf.append(torch.ones_like(x) / circle_area)
        # 按极坐标积分方法：I=int(f(r,theta)*r*drdtheta,不只适用于整个圆都在域内
        points_pdf.append(torch.ones_like(x) / (rtheta[...,0] * (2*np.pi * circle.r)))
        # rtheta = rtheta[ind]
        # points_pdf.append(torch.ones_like(x) / (rtheta[...,0] * (circle_area)))         

        mask_outer |= circle.is_in_geometry(outer_points[...,0],outer_points[...,1])


    # outer_pdf = torch.ones_like(outer_points[...,0]) / ((right - left) * (top - bottom) - sum([circle.area for circle in circles]))   
    outer_pdf = torch.ones_like(outer_points[...,0]) / ((right - left) * (top - bottom) - circles_area)    
    outer_points = outer_points[~mask_outer]

    outer_num = outer_points[...,0].data.nelement()
    total_num = outer_num + total_circle_num    
    outer_pdf = outer_pdf[~mask_outer] * outer_num / (total_num)
    # plot_points(-1,1,-1,1,points=outer_points)

    for point_pdf in points_pdf : point_pdf *= inner_num / (total_num)

    points_x.append(outer_points[...,0])
    points_y.append(outer_points[...,1])
    points_pdf.append(outer_pdf)

    points_x = torch.cat(points_x)
    points_y = torch.cat(points_y)
    points_pdf = torch.cat(points_pdf)
    N = len(points_pdf)
    estimated_area = torch.sum(1.0 / (N * points_pdf)).item()
    area_rect = (right - left) * (top - bottom)

    print(f"[PDF 验证] 理论面积        = {area_rect:.6f}")
    print(f"[PDF 验证] 积分估计面积    = {estimated_area:.6f}")
    print(f"[PDF 验证] 相对误差        = {(estimated_area - area_rect)/area_rect*100:.4f}%")
    return torch.stack((points_x,points_y),dim=1) , points_pdf.numpy()



def genDenseCirques(left, right, bottom, top, outer_num,eplison,circle_x,circle_y,
                    circles:list[Geometry.Circle],
                    inner_num):
    '''加密多个裂尖周围的积分点'''

    points_x = []
    points_y = []
    points_pdf = []

    # total_circle_num = inner_num * len(circles)

    circles_area = 0.0
    total_circle_num = 0

    outer_points = genRandomNodes2D(left, right, bottom, top,
                                    num=int(outer_num * (right - left) * (top - bottom) / (
                                                (right - left) * (top - bottom) - sum(
                                            [circle.area for circle in circles]))))

    mask_outer = torch.zeros_like(outer_points[..., 0], dtype=torch.bool)

    for i, circle in enumerate(circles):

        rtheta = genRandomNodes2D(circle.r-eplison+2e-4, circle.r+eplison+2e-4, 0, 2 * np.pi, inner_num)
        x, y = circle.local_axis.polarToCartesian(rtheta[..., 0], rtheta[..., 1])

        circle1 = Geometry.Circle(circle_x, circle_y, circle.r - eplison)
        circle2 = Geometry.Circle(circle_x, circle_y, circle.r + eplison)
        circle1_area = circle1.area_in_rect(left, right, bottom, top)
        circle2_area = circle2.area_in_rect(left, right, bottom, top)
        circle_area = circle2_area - circle1_area
        circles_area += circle_area

        ind = torch.where((x > left) & (x < right) & (y > bottom) & (y < top))
        x = x[ind]
        y = y[ind]
        rtheta = rtheta[ind]

        points_x.append(x);
        points_y.append(y)
        total_circle_num += x.data.nelement()
        points_pdf.append(torch.ones_like(x) / (rtheta[..., 0] * (2 * np.pi * 2*eplison)))
        # rtheta = rtheta[ind]
        # points_pdf.append(torch.ones_like(x) / (rtheta[...,0] * (circle_area)))
        mask_outer1 = circle1.is_in_geometry(outer_points[..., 0], outer_points[..., 1])
        mask_outer2 = circle2.is_in_geometry(outer_points[..., 0], outer_points[..., 1])
        mask_outer |= mask_outer2 & (~mask_outer1)
    # outer_pdf = torch.ones_like(outer_points[...,0]) / ((right - left) * (top - bottom) - sum([circle.area for circle in circles]))
    outer_pdf = torch.ones_like(outer_points[..., 0]) / ((right - left) * (top - bottom) - circles_area)
    outer_points = outer_points[~mask_outer]

    outer_num = outer_points[..., 0].data.nelement()
    total_num = outer_num + total_circle_num
    outer_pdf = outer_pdf[~mask_outer] * outer_num / (total_num)
    # plot_points(-1,1,-1,1,points=outer_points)

    for point_pdf in points_pdf: point_pdf *= inner_num / (total_num)

    points_x.append(outer_points[..., 0])
    points_y.append(outer_points[..., 1])
    points_pdf.append(outer_pdf)

    points_x = torch.cat(points_x)
    points_y = torch.cat(points_y)
    points_pdf = torch.cat(points_pdf)
    pdf_tensor = points_pdf
    N = len(pdf_tensor)
    estimated_area = torch.sum(1.0 / (N * pdf_tensor)).item()
    area_rect = (right - left) * (top - bottom)
    theory_area = area_rect

    print(f"[PDF 验证] 理论面积        = {theory_area:.6f}")
    print(f"[PDF 验证] 积分估计面积    = {estimated_area:.6f}")
    print(f"[PDF 验证] 相对误差        = {(estimated_area - theory_area) / theory_area * 100:.4f}%")
    return torch.stack((points_x, points_y), dim=1), points_pdf.numpy()


def genUniform_PolygonNodes(polygon_xy: list, num: int):

    polygon_path = Path(polygon_xy)
    polygon_np = np.array(polygon_xy)

    x_min, y_min = polygon_np.min(axis=0)
    x_max, y_max = polygon_np.max(axis=0)

    def polygon_area(pts):
        x, y = zip(*pts)
        return 0.5 * abs(sum(x[i]*y[(i+1)%len(x)] - x[(i+1)%len(x)]*y[i] for i in range(len(x))))

    poly_area = polygon_area(polygon_xy)
    pdf_value = 1.0 / poly_area

    valid_points = []
    batch_size = int(num * 1.5)

    while len(valid_points) < num:
        x = np.random.uniform(x_min, x_max, batch_size)
        y = np.random.uniform(y_min, y_max, batch_size)
        candidates = np.stack([x, y], axis=1)
        mask = polygon_path.contains_points(candidates)
        inside = candidates[mask]
        valid_points.extend(inside.tolist())

    points = np.array(valid_points[:num], dtype=np.float32)
    pdf = np.full(num, pdf_value, dtype=np.float32)
    # pdf_tensor = torch.tensor(pdf)
    # N = len(pdf_tensor)
    # estimated_area = torch.sum(1.0 / (N * pdf_tensor)).item()
    # print(f"[PDF 验证] 理论面积        = {poly_area:.6f}")
    # print(f"[PDF 验证] 积分估计面积    = {estimated_area:.6f}")
    # print(f"[PDF 验证] 相对误差        = {(estimated_area - poly_area)/poly_area*100:.4f}%")
    return torch.tensor(points), pdf



# ========= 主函数：多边形 + 圆域加密采样 =========
def genPolygon_with_corner_Dense(polygon_xy: list,
                                         corner: tuple,
                                         radius: float,
                                         total_num: int,
                                         inner_ratio: float = 0.2):
    """
    修复版：支持多边形角点处圆域加密，自动计算加密圆在多边形内的实际有效面积

    - polygon_xy: 多边形顶点列表
    - corner: 加密圆心 (x, y)
    - radius: 圆半径
    - total_num: 总采样点数
    - inner_ratio: 圆域加密点数占比（默认 0.2）
    """
    from matplotlib.path import Path
    import numpy as np
    import torch

    def polygon_area(pts):
        x, y = zip(*pts)
        return 0.5 * abs(sum(x[i]*y[(i+1)%len(x)] - x[(i+1)%len(x)]*y[i] for i in range(len(x))))

    polygon_path = Path(polygon_xy)
    polygon_np = np.array(polygon_xy)
    x_min, y_min = polygon_np.min(axis=0)
    x_max, y_max = polygon_np.max(axis=0)
    poly_area = polygon_area(polygon_xy)

    inner_num = int(total_num * inner_ratio)
    outer_num = total_num - inner_num
    cx, cy = corner

    # Step 1: 圆域采样并记录有效面积比例
    inner_points = []
    total_sampled = 0
    valid_sampled = 0
    batch_size = inner_num * 2

    while len(inner_points) < inner_num:
        r = np.sqrt(np.random.uniform(0, radius**2, batch_size))
        theta = np.random.uniform(0, 2 * np.pi, batch_size)
        x = cx + r * np.cos(theta)
        y = cy + r * np.sin(theta)
        candidates = np.stack([x, y], axis=1)

        mask = polygon_path.contains_points(candidates)
        inside = candidates[mask]
        inner_points.extend(inside.tolist())

        valid_sampled += mask.sum()
        total_sampled += len(mask)

    inner_points = np.array(inner_points[:inner_num], dtype=np.float32)
    ratio = valid_sampled / total_sampled
    effective_circle_area = np.pi * radius**2 * ratio
    pdf_inner = np.full(inner_num, 1.0 / effective_circle_area, dtype=np.float32)

    # Step 2: polygon \ circle 区域均匀采样
    outer_points = []
    batch_size = int(outer_num * 1.5)

    while len(outer_points) < outer_num:
        x = np.random.uniform(x_min, x_max, batch_size)
        y = np.random.uniform(y_min, y_max, batch_size)
        candidates = np.stack([x, y], axis=1)
        in_poly = polygon_path.contains_points(candidates)
        dist2 = (candidates[:, 0] - cx)**2 + (candidates[:, 1] - cy)**2
        out_circle = dist2 > radius**2
        mask = in_poly & out_circle
        outer_points.extend(candidates[mask].tolist())

    outer_points = np.array(outer_points[:outer_num], dtype=np.float32)
    outer_area = poly_area - effective_circle_area
    pdf_outer = np.full(outer_num, 1.0 / outer_area, dtype=np.float32)

    # Step 3: 合并并全局归一化 PDF
    total_points = np.concatenate([outer_points, inner_points], axis=0)
    total_pdf = np.concatenate([
        pdf_outer * outer_num / total_num,
        pdf_inner * inner_num / total_num
    ], axis=0)

    # === 验证 PDF 准确性 ===
    # pdf_tensor = torch.tensor(total_pdf)
    # N = len(pdf_tensor)
    # estimated_area = torch.sum(1.0 / (N * pdf_tensor)).item()
    # print(f"[PDF 验证] 多边形面积           = {poly_area:.6f}")
    # print(f"[PDF 验证] 圆域有效面积         = {effective_circle_area:.6f}")
    # print(f"[PDF 验证] 积分估计面积         = {estimated_area:.6f}")
    # print(f"[PDF 验证] 相对误差             = {(estimated_area - poly_area)/poly_area*100:.4f}%")

    return torch.tensor(total_points), total_pdf




#这里不用1/(num-1)吗
#是靠近end的越密吗
def gen1DLine(start,end,num):
    # sep = (end - start) / num
    x_uniform = np.arange(0.001, 1, 1/num)
    x_transform = np.sqrt(x_uniform)
    x = x_transform * (end - start) + start
    return torch.from_numpy(x)
  ##裂纹加密？？？
def genTipDense1D(start,end,Tip,num):
    left_num = int((Tip - start) / (end - start) * num)
    right_num = num - left_num
    x_left = gen1DLine(start,Tip,left_num)
    x_right = gen1DLine(end,Tip,right_num).flip(dims=[-1])
    return torch.cat((x_left,x_right),dim=-1)

def genTipDenseMesh(xstart,xend,
                    ystart,yend,
                    xTip,yTip,
                    xnum,ynum):
    x = genTipDense1D(xstart,xend,xTip,xnum)
    y = genTipDense1D(ystart,yend,yTip,ynum)
    return meshgirdFromXY(x,y)
    
    
    
    



# def genDenseCircles(left, right, bottom, top, outer_num,
#                     circles:list[Geometry.Circle],
#                     inner_num):
#     '''加密多个裂尖周围的积分点'''

#     points_x=[]
#     points_y=[]
#     points_pdf = []

#     total_circle_num = inner_num * len(circles)

#     circles_area = 0.0

#     outer_points = genRandomNodes2D(left, right, bottom, top,
#                 num = int(outer_num * (right - left) * (top - bottom)/ ((right - left) * (top - bottom) - sum([circle.area for circle in circles]))))
#     mask = torch.zeros_like(outer_points[...,0],dtype=torch.bool)
#     for i,circle in enumerate(circles):

#         rtheta = genRandomNodes2D(circle.r * 1e-3, circle.r, 0, 2*np.pi, inner_num)
#         x,y = circle.local_axis.polarToCartesian(rtheta[...,0],rtheta[...,1])

#         # 筛选掉不在举行域内的点
#         circle_area = circle.area_in_rect(left, right, bottom, top)
#         circles_area += circle_area

#         x = x[ x > left ] ; x = x[ x < right ]
#         y = y[ y > bottom ] ; y = y[ y < top ]

#         points_x.append(x) ; points_y.append(y)
#         # points_pdf.append(torch.ones_like(x) / circle.area)
#         # points_pdf.append(torch.ones_like(x) / (rtheta[...,0] * (2*np.pi * circle.r)))    #按极坐标积分方法：I=int(f(r,theta)*r*drdtheta
#         points_pdf.append(torch.ones_like(x) / (circle_area))         

#         mask |= circle.is_in_geometry(outer_points[...,0],outer_points[...,1])


#     outer_pdf = torch.ones_like(outer_points[...,0]) / ((right - left) * (top - bottom) - sum([circle.area for circle in circles]))    
#     outer_points = outer_points[~mask]

#     outer_num = outer_points[...,0].data.nelement()
#     total_num = outer_num + total_circle_num    
#     outer_pdf = outer_pdf[~mask] * outer_num / (total_num)
#     # plot_points(-1,1,-1,1,points=outer_points)

#     for point_pdf in points_pdf : point_pdf *= inner_num / (total_num)

#     points_x.append(outer_points[...,0])
#     points_y.append(outer_points[...,1])
#     points_pdf.append(outer_pdf)

#     points_x = torch.cat(points_x)
#     points_y = torch.cat(points_y)
#     points_pdf = torch.cat(points_pdf)
#     return torch.stack((points_x,points_y),dim=1) , points_pdf.numpy()
  

def plot_points(left, right, bottom, top, points):
    plt.gcf().gca().add_artist(Rectangle(xy=(left,bottom),
                                         width=right-left,
                                         height=top-bottom,
                                         edgecolor='black',
                                         fill=False,
                                         linewidth=1))
    plt.scatter(points[:,0],points[:,1],s=1)
    plt.axis('equal')
    plt.axis('off')
    # plt.xlim(left,right)
    # plt.ylim(bottom,top)
    plt.show()



if __name__ == '__main__':

    # from stats import Uniform,Norm,Stats2D
    # stat1 = Uniform(0,1)
    # # stat2 = Uniform(0.1,0.4)
    # stat2D = Stats2D(stat1,stat1) 
    # # stat2D.view_PDF(-1,1,-1,1)
    # # points = AcceptanceSampling2D(-1,1,-1,1,num= 1000, stats=stat2D)
    # # plot_points(-1,1,-1,1,points)
    
    # stat1 = stats.Stats2D(stats.Uniform(0,1),stats.Uniform(0,1))
    # stat2 = stats.Stats2D(stats.Uniform(0.25,0.75),stats.Uniform(0.25,0.75))
    # # stat3 = stats.Stats2D(stats.Uniform(0.4,0.6),stats.Uniform(0.4,0.6))
    # points_distribution = 2 * stat1 + stat2
    # # points_distribution = stats.Stats2D(stats.Uniform(0,1),stats.Uniform(0,1))
    # points = AcceptanceSampling2D(0,1,0,1,20000,points_distribution)
    # points_pdf = points_distribution.pdf(points[:,0],points[:,1])

    # x = gen1DLine(0.2,0.8,20)
    # y = torch.zeros_like(x)
    # plt.scatter(x,y)
    # plt.show()

    def f(x,y):
        return torch.ones_like(x) 
        # return (x+0.5)**2 * (y+0.5)**3 # 
    def f1(x):
        return x**4
        # return torch.ones_like(x) 





    # pdf = stat2D.pdf(x,y)
    # print(montecarlo(fxy,pdf))
    
    # x = genTipDense1D(0,1,0.5,100)
    # fx = f1(x)
    # print(trapz1D(fx,x))

    x, y = genMeshNodes2D(-1, 1, 200, -1, 1, 200)

    fxy = f(x, y)
    # print(trapz2D(fxy,torch.stack([x,y],dim=1),[5,5]))
    # print(simps2D(fxy,torch.stack([x,y],dim=1),[99,99]))
    print(trapz2D(fxy, torch.stack([x, y], dim=1), [200, 200]))

    # x,y = genTipDenseMesh(-1,1,-1,1,0.5,0.5,100,100)
    # plot_points(-1,1,-1,1,points=torch.stack([x,y],dim=1))

    # fxy = f(x,y)
    # print(trapz2D(fxy,torch.stack([x,y],dim=1),[100,100]))
    # xstats = stats.Uniform(-1,1)
    # ystats = stats.Uniform(-1,1)
    # xnum = 200;ynum = 200
    # x,y = genHeteroMeshNodesPDF(xstats,ystats,xnum,ynum)

    # x,y = genHeteroNodes2D(xsep=[-1,-0.5,0.5,1],xnum=[25,100,25],
    #                        ysep=[-1,-0.5,0.5,1],ynum=[25,100,25])

    x, y = genHeteroTip2D(-1, 1, -1, 1,
                          0.0, 0.0,
                          100, 100, 100, 100, x_inteval=0.2, y_inteval=0.35)

    fxy = f(x, y)
    points = torch.stack([x, y], dim=1)
    plot_points(-1, 1, -1, 1, points=points)

    # points , pdf = FunctionSamplingUniform2D(function=f,
    #                                     x_range=[-1,1],
    #                                     y_range=[-1,1],
    #                                     num=10000)
    # x = points[...,0]; y= points[...,1]
    # print(montecarlo(f(x,y),pdf))

    print(trapz2D(fxy, points, [200, 200]))

    # points,points_pdf = genDenseCircles(-1,1,-1,1,15000,
    #                                     [Geometry.Circle(0.5,0.5,0.25),
    #                                      Geometry.Circle(-0.5,0.5,0.25),
    #                                      Geometry.Circle(-0.5,-0.5,0.25),
    #                                      Geometry.Circle(0.5,-0.5,0.25),
    #                                      Geometry.Circle(0.0,0.0,0.25),],
    #                                     15000)
    # points,points_pdf = genDenseCircles(-1,1,-1,1,10000,
    #                                 [Geometry.Circle(0.75,0.5,0.5),
    #                                  Geometry.Circle(-1.0,0.5,0.5)],
    #                                 5000)
    # plot_points(-1,1,-1,1,points=points)
    # print(montecarlo(f(points[...,0],points[...,1]),points_pdf))

#
# circle1 = Geometry.Circle(0.5,0.5,0.1)
# points, _ = genDenseCircles(left=-1, right=1, bottom=-1, top=1, outer_num=500, circles=[circle1], inner_num=200)
# def plot_points(points_x, points_y, title="Points Visualization"):
#     '''绘制生成的积分点的分布'''
#     plt.figure(figsize=(6,6))
#     plt.scatter(points_x, points_y, s=1, color='blue', label='Generated Points')
#     plt.title(title)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.xlim(-1, 1)
#     plt.ylim(-1, 1)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.legend()
#     plt.show()
#
# plot_points(points[:, 0], points[:, 1], title="Dense Circle Points")