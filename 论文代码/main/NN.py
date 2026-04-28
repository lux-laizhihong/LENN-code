import torch.nn as nn
import torch

from get_grad import get_grad
class Block(nn.Module):
    '''网络模型'''

    def __init__(self, input_shape, width, activation=nn.ReLU) -> None:
        super().__init__()
        '''定义用到的具有可训练参数的层'''

        self.layers = nn.Sequential(nn.Linear(input_shape, width), activation(),
                                    nn.Linear(width, input_shape), activation())

    def forward(self, x):
        '''创建网络'''
        return self.layers(x) + x


class ResidualNet(nn.Module):
    '''网络模型'''

    def __init__(self, input=2, output=2, width=50, activation=nn.ReLU, hidden_layer_num=4) -> None:
        '''隐藏层数量:int(hidden_layer_num/2)*2+1
            Example:hidden_layer_num=2
            input...>activation(linear)...>activation(linear)...>activation(linear)...>linear
                                        :                                           :
                                        :...........................................:
            画这个的时候纶纶刚下飞机到克罗地亚给我打了个电话
        '''
        super().__init__()
        self.activation = activation()
        block_num = int(hidden_layer_num / 2)
        # 创建输入层
        self.input_layer = nn.Linear(input, width)
        # 创建隐藏层
        self.block_layers = nn.ModuleList()
        for i in range(block_num - 1):
            self.block_layers.append(Block(width, width, activation))
        # 创建输出层
        self.output_layer = nn.Linear(width, output)

    def forward(self, x):
        x = x.to(torch.float32)

        x = self.activation(self.input_layer(x))
        for hidden_layer in self.block_layers:
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x


class stack_net(nn.Module):
    def __init__(self, input=2, output=2, width=50, activation=nn.Tanh, net=ResidualNet, depth=4) -> None:
        super().__init__()

        for i in range(output):
            setattr(self, "tower" + str(i + 1),
                    net(input=input, output=1, width=width, activation=activation, hidden_layer_num=depth))
        self.towers = [getattr(self, "tower" + str(i + 1)) for i in range(output)]

    def forward(self, x):
        '''创建网络'''
        return [model(x) for model in self.towers]
        # return list(map(lambda model: model(x),self.towers))


class MultilayerNN(nn.Module):
    def __init__(self, width, hidden_layer_num=4, activation=nn.Tanh, input=2, output=1):
        super(MultilayerNN, self).__init__()
        self.activation = activation()
        if type(width) == int:
            width = [width] * hidden_layer_num

        # 创建输入层
        self.input_layer = nn.Linear(input, width[0])
        # 创建隐藏层
        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_layer_num - 1):
            self.hidden_layers.append(nn.Linear(width[i], width[i + 1]))
        # 创建输出层
        self.output_layer = nn.Linear(width[-1], output)

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x


# 归一化作用
class AxisScalar2D(nn.Module):
    def __init__(self, net: nn.Module, A: torch.Tensor, B: torch.Tensor) -> None:
        '''X_out=A*X+B'''
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = net
        # self.xscalar=xscalar
        self.A = A.to(self.device)
        self.B = B.to(self.device)
        # self.A = A
        # self.B = B

    def forward(self, xy):
        # print(torch.max(xy,dim=0))
        # print(torch.min(xy,dim=0))
        # xy[...,:] *= self.A[:]
        # xy[...,:] += self.B[:]
        xy_normed = self.A[:] * xy[..., :] + self.B[:]
        # print(xy)
        # print(get_grad(xy_normed,xy))
        # print(torch.max(xy_normed,dim=0))
        # print(torch.min(xy_normed,dim=0))
        # print(xy_normed[:,2:3])
        return self.net(xy_normed)


class AxisScalar2D_withinput(nn.Module):
    def __init__(self, net: nn.Module, A: torch.Tensor, B: torch.Tensor) -> None:
        '''X_out=A*X+B'''
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = net
        # self.xscalar=xscalar
        self.A = A.to(self.device)
        self.B = B.to(self.device)

    def forward(self, xy):
        # print(torch.max(xy,dim=0))
        # print(torch.min(xy,dim=0))
        # xy[...,:] *= self.A[:]
        # xy[...,:] += self.B[:]
        xy_normed = self.A[:] * xy[..., :] + self.B[:]
        # print(torch.max(xy_normed,dim=0))
        # print(torch.min(xy_normed,dim=0))
        # print(xy_normed[:,2:3])
        return self.net(xy_normed),xy_normed
