import torch

def montecarlo(f:torch.Tensor,pdf:torch.Tensor):
    return torch.sum(f/pdf) / f.data.nelement()



# def trapz(f:torch.Tensor, x:torch.Tensor):
#     # y = np.asanyarray(y)

#     d = x[1:] - x[:-1]
#     # # reshape to correct shape
#     # shape = [1] * f.ndimension()
#     # shape[-1] = d.shape[0]
#     # d = d.reshape(shape)
#     ret = 0.5 * torch.sum(d * (f[...,1:] + f[...,:-1]) , -1)
#     return ret
def trapz1D(f:torch.Tensor,x:torch.Tensor):
    d = x[1:] - x[:-1]
    # print(d.device)
    return 0.5 * torch.sum(d * (f[...,1:] + f[...,:-1]) , -1)


def trapz2D(f:torch.Tensor,xy:torch.Tensor,shape):
        f2D = f.reshape(shape[0], shape[1])
        x = xy[:, 0].flatten().reshape(shape[0], shape[1])
        y = xy[:, 1].flatten().reshape(shape[0], shape[1])
        a = trapz1D(f2D, y[0, :])
        b = trapz1D(a, x[:, 0])
        return trapz1D(trapz1D(f2D, y[0, :]), x[:, 0])



def simps2D(f:torch.Tensor,xy:torch.Tensor,shape):
        f2D = f.reshape(shape[0], shape[1])
        x = xy[:, 0].flatten().reshape(shape[0], shape[1])
        y = xy[:, 1].flatten().reshape(shape[0], shape[1])
        return simps1D(simps1D(f2D, y[0, :]), x[:, 0])

def simps1D(y, x, axis=-1):
    d = x[2::2] - x[:-1:2]
    # Even spaced Simpson's rule.
    result = torch.sum(d / 6.0 * (y[...,:-1:2] + 4 * y[...,1::2] + y[...,2::2]), axis) 
    return result