import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import rv_continuous

class Stats1D:
    def __init__(self) -> None:
        self.weights = np.array([1.0])
        self.submodels = [self]
        self.start = 0.0
        self.end = 0.0
    def pdf(self,x):...
    # def pdf(self,x):
    #     '''外部调用的概率密度函数计算'''
    #     # if self.submodels is None:
    #     #     return self._pdf(x)
    #     # else:
    #     pdf = np.zeros_like(x)
    #     for submodel, weight in zip(self.submodels, self.weights):
    #         pdf += weight * submodel._pdf(x)
    #     return pdf
    def check_weights(self):
        if self.weights.sum() - 1 >1e-8:
            print(self.weights.sum())
            raise Exception('weights error!')
        
    def view_PDF(self ,start , end):
        x = np.linspace(start, end, 1000)
        y = self.pdf(x)
        plt.plot(x, y)
        plt.show()

    def __add__(self, other):
        if isinstance(other, Stats1D):
            submodels = self.submodels + other.submodels
            weights = np.concatenate([self.weights, other.weights])
            weights /= weights.sum()
            return MixtureModel1D(submodels, weights)
        else:
            return NotImplemented

    def __mul__(self, other):
        if np.isscalar(other):
            weights = self.weights * other
            return MixtureModel1D(self.submodels, weights)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)

class MixtureModel1D(Stats1D):
    def __init__(self, submodels:list[Stats1D], weights:np.ndarray):
        self.submodels = submodels
        self.weights = weights
        self.start = min([model.start for model in self.submodels])
        self.end = max([model.end for model in self.submodels])

    def pdf(self, x):
        pdf = np.zeros_like(x)
        for submodel, weight in zip(self.submodels, self.weights):
            pdf += weight * submodel.pdf(x)
        return pdf
    




class Stats2D:
    def __init__(self,xstats:Stats1D,ystats:Stats1D) -> None:
        self.weights = np.array([1.0])
        self.submodels = [self]
        self.xstats=xstats
        self.ystats=ystats
    def pdf(self, x , y):
        return self.xstats.pdf(x) * self.ystats.pdf(y)
    
    def view_PDF(self, left, right, bottom, top):

        x = np.linspace(left, right, 100)
        y = np.linspace(bottom, top, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.pdf(X, Y)
        plt.contourf(X,Y,Z,200,cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.show()

    def __add__(self, other):
        if isinstance(other, Stats2D):
            submodels = self.submodels + other.submodels
            weights = np.concatenate([self.weights, other.weights])
            weights /= weights.sum()
            return MixtureModel2D(submodels, weights)
        else:
            return NotImplemented

    def __mul__(self, other):
        if np.isscalar(other):
            weights = self.weights * other
            return MixtureModel2D(self.submodels, weights)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other)


class MixtureModel2D(Stats2D):
    def __init__(self, submodels:list[Stats2D], weights:np.ndarray):
        self.submodels = submodels
        self.weights = weights

    def pdf(self, x, y):
        pdf = np.zeros_like(x)
        for submodel, weight in zip(self.submodels, self.weights):
            pdf += weight * submodel.pdf(x , y)
        return pdf



from scipy.stats import norm as n
class Norm(Stats1D):
    '''正态分布'''
    def __init__(self , mean , std) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.norm = n(loc=mean, scale=std)
    def pdf(self, x):
        return self.norm.pdf(x)

from scipy.stats import uniform as u
class Uniform(Stats1D):
    '''均匀分布'''
    def __init__(self , start , end) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.length = end - start
        self.uniform = u(loc = start, scale = self.length)
    def pdf(self, x):
        return self.uniform.pdf(x)
    
class LinePDF(Stats1D):
    '''pdf线性增长,a处最大'''
    def __init__(self , start , end) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.length = end - start
    def pdf(self, x):
        return 2 * x / (self.length) **2 * np.heaviside(x-self.start,0) * np.heaviside(self.end - x,0)
        
  
if __name__ == '__main__':
    stat1 = LinePDF(0,2)
    stat1.view_PDF(-5,5)
    # stat2 = Norm(2,1)
    # stat3 = stat1 + stat2
    # stat =  stat1 + stat3
    # stat.view_PDF(-5,5)
    
    