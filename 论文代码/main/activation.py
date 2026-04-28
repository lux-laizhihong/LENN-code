from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F
from get_grad import get_grad
import torch

class ReLU2(Module):

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU2, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace) ** 2

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class ReLU3(Module):

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ReLU3, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace) ** 3

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str

class ELU2(Module):

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(ELU2, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return  F.elu(input, inplace=self.inplace) ** 2 

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
    
class SIN(Module):

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(SIN, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return  torch.sin(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
    
# torch.nn.ELU
# class ReLU2(torch.autograd.Function):

#     @staticmethod
#     def forward(self, input_):
#         # 在forward中，需要定义MyReLU这个运算的forward计算过程
#         # 同时可以保存任何在后向传播中需要使用的变量值
#         self.save_for_backward(input_)    # 将输入保存起来，在backward时使用
#         output = input_.clamp(min=0)               # relu就是截断负数，让所有负数等于0
#         return output ** 2
    
#     @staticmethod
#     def backward(self, grad_output):
#         # 根据BP算法的推导（链式法则），dloss / dx = (dloss / doutput) * (doutput / dx)
#         # dloss / doutput就是输入的参数grad_output、
#         # 因此只需求relu的导数，在乘以grad_outpu    
#         input_, = self.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input_ < 0] = 0                # 上诉计算的结果就是左式。即ReLU在反向传播中可以看做一个通道选择函数，所有未达到阈值（激活值<0）的单元的梯度都为0
#         return 2 * input_ * grad_input

# def relu(input_):
#     # MyReLU()是创建一个MyReLU对象，
#     # Function类利用了Python __call__操作，使得可以直接使用对象调用__call__制定的方法
#     # __call__指定的方法是forward，因此下面这句MyReLU（）（input_）相当于
#     # return MyReLU().forward(input_)
#     return ReLU2()(input_)
if __name__ == '__main__':
    input_ = torch.linspace(-1, 2, steps=100)
    input_.requires_grad_()
    import matplotlib.pyplot as plt

    def Mollifier(t):
        # t = t*20 - 1
        return 0.5 * (torch.sign(1 - torch.abs(t)) +1) * torch.exp(1/(t**2 -1))
        # return 0.5 * (torch.sign(1 - torch.abs(t)) +1) * (1-t**2)**2

    def RELU2PSI(t):
        # t = t*20 - 1
        # return 0.5 * (torch.sign(1 - torch.abs(t)) +1) * (1-t**2)**2
        return F.relu(1.0 - F.relu(t)**2)**2

    out_Mollifier = Mollifier(input_) 
    out_RELU2PSI = RELU2PSI(input_) 
    # out =  F.sigmoid(10 * input_)
    # grad = get_grad(out_Mollifier,input_)
    # F.relu(1.0 - F.relu(self.linear_interp(x))**2)**2

    # plt.plot(input_.detach().numpy(),out_Mollifier.detach().numpy())
    plt.plot(input_.detach().numpy(),out_RELU2PSI.detach().numpy())
    plt.show()
    # plt.plot(input_.detach().numpy(),get_grad(out_Mollifier,input_).detach().numpy())
    plt.plot(input_.detach().numpy(),get_grad(out_RELU2PSI,input_).detach().numpy())
    plt.show()
    # plt.plot(input_.detach().numpy(),get_grad(get_grad(out_Mollifier,input_),input_).detach().numpy())
    plt.plot(input_.detach().numpy(),get_grad(get_grad(out_RELU2PSI,input_),input_).detach().numpy())
    plt.show()
# out = ReLU3()(input_)
# plt.plot(input_.detach().numpy(),out.detach().numpy())
# grad = get_grad(out,input_)
# plt.show()
# plt.plot(input_.detach().numpy(),grad.detach().numpy())
# plt.show()
# plt.plot(input_.detach().numpy(),get_grad(grad,input_).detach().numpy())
# plt.show()