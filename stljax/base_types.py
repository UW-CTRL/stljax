import torch 

class Expression(torch.nn.Module):

    def __init__(self, name, value, reverse, time_dim=1):
        super(Expression, self).__init__()
        self.value = value
        self.name = name
        self.reverse = reverse
        self.time_dim = time_dim

    def set_name(self, new_name):
        self.name = new_name

    def set_value(self, new_value):
        self.value = new_value
    
    # Unary operators
    def __str__(self):
        return str(self.name)

    def __call__(self):
        return self.value



class Minish(torch.nn.Module):
    '''
    Function to compute the min, or softmin, or other variants of the min function.
    '''
    def __init__(self, name="Minish input"):
        super().__init__()
        self.input_name = name

    def forward(self, x, scale=1, dim=1, keepdim=True, agm=False, distributed=False):
        '''
        x is of size [batch_size, T, ...] where T is typically the trace length.

        if scale <= 0, then the true max is used, otherwise, the softmax is used.

        dim is the dimension which the max is applied over. Default: 1

        keepdim keeps the dimension of the input tensor. Default: True

        agm is the arithmetic-geometric mean. Currently in progress. If all elements are >0, output is ᵐ√(Πᵢ (1 + ηᵢ)) - 1.If some the elements <= 0, output is the average of those negative values. scale doesn't play a role here except to switch between the using the AGM or true robustness value (scale <=0).

        distributed addresses the case when there are multiple max values. As max is poorly defined in these cases, PyTorch (randomly?) selects one of the max values only. If distributed=True and scale <=0 then it will average over the max values and split the gradients equally. Default: False
        '''

        if isinstance(x, Expression):
            assert x.value is not None, "Input Expression does not have numerical values"
            x = x.value
        
        return -torch.logsumexp(-scale*x, dim=dim, keepdim=keepdim)/scale 
        #+ torch.log(torch.tensor(x.shape[dim], dtype=x.dtype, device=x.device))
        # if agm == True:
        #     if torch.gt(x, 0).all():
        #         return torch.log(1+x).mean(dim=dim, keepdim=keepdim).exp() - 1
        #     else:
        #         # return x[torch.lt(x, 0)].reshape(*x.shape[:-1], -1).mean(dim=dim, keepdim=keepdim)
        #         return  (torch.lt(x,0) * x).sum(dim, keepdim=keepdim) / torch.lt(x, 0).sum(dim, keepdim=keepdim)
        # else:
        #     # return -torch.log(torch.exp(-x*scale).sum(dim=dim, keepdim=keepdim))/scale
         
class Maxish(torch.nn.Module):
    '''
    Function to compute the min, or softmin, or other variants of the min function.
    '''
    def __init__(self, name="Minish input"):
        super().__init__()
        self.input_name = name

    def forward(self, x, scale=1, dim=1, keepdim=True, agm=False, distributed=False):
        '''
        x is of size [batch_size, T, ...] where T is typically the trace length.

        if scale <= 0, then the true max is used, otherwise, the softmax is used.

        dim is the dimension which the max is applied over. Default: 1

        keepdim keeps the dimension of the input tensor. Default: True

        agm is the arithmetic-geometric mean. Currently in progress. If all elements are >0, output is ᵐ√(Πᵢ (1 + ηᵢ)) - 1.If some the elements <= 0, output is the average of those negative values. scale doesn't play a role here except to switch between the using the AGM or true robustness value (scale <=0).

        distributed addresses the case when there are multiple max values. As max is poorly defined in these cases, PyTorch (randomly?) selects one of the max values only. If distributed=True and scale <=0 then it will average over the max values and split the gradients equally. Default: False
        '''

        if isinstance(x, Expression):
            assert x.value is not None, "Input Expression does not have numerical values"
            x = x.value
        
        return torch.logsumexp(scale*x, dim=dim, keepdim=keepdim)/scale 