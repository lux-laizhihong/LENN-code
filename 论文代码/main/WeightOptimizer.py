import torch
import numpy as np
from scipy import stats

class ReLoBRaLo:
    def __init__(self,init_weights:list,L0:torch.Tensor,alpha = 0.999, tau = 0.1, E_rou = 0.9995) -> None:
        # self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.L0 = L0
        self.m = len(init_weights)
        self.E_rou = E_rou
        self.alpha = alpha
        self.tau = tau
        self.weights = init_weights
        self.L_past = self.L0

    def ReLoB(self,L,L_old):   
        ReLo = L/(L_old * self.tau)
        expReLo = torch.exp(ReLo - torch.max(ReLo))
        return self.m * expReLo / torch.sum(expReLo)
    
    def step(self,L:torch.Tensor):
        rou = stats.bernoulli.rvs(p = self.E_rou, size = 1)[0]
        weight_hist = rou * self.weights + (1-rou) * self.ReLoB(L,self.L0)
        self.weights = self.alpha * weight_hist + (1-self.alpha) * self.ReLoB(L,self.L_past)
        # print(self.weights)
        self.L_past = L
        return self.weights





