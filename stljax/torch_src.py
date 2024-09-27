import torch
import numpy as np
import os
import sys
from .base_types import Expression, Minish, Maxish
# from torchviz import make_dot
# Set the device based on runtime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_input_values(inputs):
    '''
    TODO:
    This function is supposed to do all the necessary conversions to the input signal. Currently, it just returns the inputs as it is.
    '''
    return inputs



class STL_Formula(torch.nn.Module):
    '''
    NOTE: All the inputs are assumed to be TIME REVERSED. The outputs are also TIME REVERSED
    All STL formulas have the following functions:
    robustness_trace: Computes the robustness trace.
    robustness: Computes the robustness value of the trace
    eval_trace: Computes the robustness trace and returns True in each entry if the robustness value is > 0
    eval: Computes the robustness value and returns True if the robustness value is > 0
    forward: The forward function of this STL_formula PyTorch module (default to the robustness_trace function)

    Inputs to these functions:
    trace: the input signal assumed to be TIME REVERSED. If the formula has two subformulas (e.g., And), then it is a tuple of the two inputs. An input can be a tensor of size [batch_size, time_dim,...], or an Expression with a .value (Tensor) associated with the expression.
    pscale: predicate scale. Default: 1
    scale: scale for the max/min function.  Default: -1
    keepdim: Output shape is the same as the input tensor shapes. Default: True
    agm: Use arithmetic-geometric mean. (In progress.) Default: False
    distributed: Use the distributed mean. Default: False
    '''

    def __init__(self):
        super(STL_Formula, self).__init__()

    def robustness_trace(self, signal, **kwargs):
        raise NotImplementedError("robustness_trace not yet implemented")

    def robustness(self, signal, time_dim, **kwargs):
        '''
        Extracts the robustness_trace value at the given time.
        Default: time=0 assuming this is the index for the NON-REVERSED trace. But the code will take it from the end since the input signal is TIME REVERSED.

        '''
        #return self.forward(inputs, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)[:,-(time+1),:].unsqueeze(1)

    
    def forward(self, inputs, **kwargs):
        '''
        Evaluates the robustness_trace given the input. The input is converted to the numerical value first.
        '''

        inputs = convert_to_input_values(inputs)
        return self.robustness_trace(inputs, **kwargs)


class LessThan(STL_Formula):
    """
    The LessThan operation. lhs < val where lhs is a placeholder for a signal, and val is a constant.
    Args:
        lhs: string, Expression, or Predicate
        val: float, int, Expression, or array (of appropriate size). It cannot be a string.
    """
    def __init__(self, lhs='x', val='c'):
        super().__init__()
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "val on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val

    def robustness_trace(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        """
        Computes robustness trace:  rhs - lhs
        Args:
            signal: Tensor of Expected size [bs, time_dim, state_dim]

        Returns:
            robustness_trace: Tensor. Same size as signal.
        """
        if isinstance(self.val, Expression):
            assert self.val.value is not None, "Expression does not have numerical values"
            c_val = self.val.value

        else:
            c_val = self.val


        return (c_val - signal) 

    def forward(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        return self.robustness_trace(signal, **kwargs)

class GreaterThan(STL_Formula):
    """
    The GreaterThan operation. lhs > val where lhs is a placeholder for a signal, and val is a constant.
    Args:
        lhs: string, Expression, or Predicate
        val: float, int, Expression, or array (of appropriate size). It cannot be a string.
    """
    def __init__(self, lhs='x', val='c'):
        super().__init__()
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "val on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val

    def robustness_trace(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        """
        Computes robustness trace:  lhs - rhs
        Args:
            signal: Tensor of Expected size [bs, time_dim, state_dim]

        Returns:
            robustness_trace: Tensor. Same size as signal.
        """
        if isinstance(self.val, Expression):
            assert self.val.value is not None, "Expression does not have numerical values"
            c_val = self.val.value

        else:
            c_val = self.val


        return (signal - c_val) 

    def forward(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        return self.robustness_trace(signal, **kwargs)

class Negation(STL_Formula):
    """
    The Negation operation. !lhs
    Args:
        lhs: string, Expression, or Predicate
    """
    def __init__(self, subformula):
        super().__init__()
        self.subformula = subformula

    def robustness_trace(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        """
        Computes robustness trace:  -lhs
        Args:
            signal: Tensor of Expected size [bs, time_dim, state_dim]

        Returns:
            robustness_trace: Tensor. Same size as signal.
        """
        return -self.subformula(signal)

    def forward(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        return self.robustness_trace(signal, **kwargs)

class And(STL_Formula):
    '''
    inputs: tuple (x,y) where x and y are the inputs to each subformula respectively. x or y can also be a tuple if the subformula requires multiple inputs (e.g, ϕ₁(x) ∧ (ϕ₂(y) ∧ ϕ₃(z)) would have inputs=(x, (y,z)))
    trace1 and trace2 are size [batch_size, time_dim, x_dim]
    '''
    def __init__(self, subformula1, subformula2):
        # super(And, self).__init__()
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Minish()


    def robustness_trace(self, inputs, **kwargs):
        """
        Computing robustness trace of subformula1 ∧ subformula2  min(subformula1(input1), subformula2(input2))

        The original one used to take arbitary ands and then separate them. We are assuming only two subformulas for now.

        Args:
            inputs: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs (e.g, ϕ₁(x) ∧ (ϕ₂(y) ∧ ϕ₃(z)) would have inputs=(x, (y,z))). If using Predicates to define the formula, then inputs is just a single jnp.array. Not need for different signals for each subformula. Expected signal is size [batch_size, time_dim, x_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: Tensor. Same size as signal.
        """
        # xx = And.separate_and(self, inputs, **kwargs)
        pre =self.subformula1(inputs[0])
        post = self.subformula2(inputs[1])
        temp = torch.stack([pre, post], dim=0)
        return self.operation(temp,scale=100000, dim=0, keepdim=False, **kwargs)  # [batch_size, time_dim, ...]                                          # [batch_size, time_dim, ...]

    def forward(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        return self.robustness_trace(signal, **kwargs)

class Always(STL_Formula):
    """
    The Always STL formula □_[a,b] subformula
    The robustness value is the minimum value of the input trace over a prespecified time interval

    Args:
        subformula: subformula that the Always operation is applied on
        interval: time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep size is.
    """
    
    def __init__(self, subformula, interval=None):
        super().__init__()
        self.interval = interval
        self.operation = Minish()  
        self.subformula = subformula

    def robustness_trace(self, signal, time_dim=0, padding="last", large_number=1e9, **kwargs):
        signal = self.subformula(signal, time_dim=time_dim, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = large_number
        
        if self.interval is None:
            interval = [0, T]
        else:
            interval = self.interval

        signal_matrix = signal.view(T, 1) * torch.ones(1, T, device=device)

        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(dim=time_dim)
        else:
            pad_value = padding

        signal_pad = torch.ones(interval[1] + 1, T, device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)

        subsignal_mask = torch.tril(torch.ones(T + interval[1] + 1, T, device=device))
        time_interval_mask = torch.triu(torch.ones(T + interval[1] + 1, T, device=device), -interval[-1]) * torch.tril(torch.ones(T + interval[1] + 1, T, device=device), -interval[0])

        masked_signal_matrix = torch.where((time_interval_mask * subsignal_mask).bool(), signal_padded, torch.tensor(mask_value, dtype=signal_padded.dtype, device=device))

        return self.operation(masked_signal_matrix, scale=large_number, dim=time_dim, keepdim=False, **kwargs)

    def forward(self, signal: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.robustness_trace(signal, **kwargs)

class Eventually(STL_Formula):
    """
    The Eventually STL formula ◇_[a,b] subformula
    The robustness value is the maximum value of the input trace over a prespecified time interval

    Args:
        subformula: subformula that the Eventually operation is applied on
        interval: time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep size is
    """
    
    def __init__(self, subformula, interval=None):
        super().__init__()
        self.interval = interval
        self.operation = Maxish()
        self.subformula = subformula

    def robustness_trace(self, signal, time_dim=0, padding="last", large_number=1e9, **kwargs):
        signal = self.subformula(signal, time_dim=time_dim, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = -large_number
        
   
        if self.interval is None:
            interval = [0, T]
        else:
            interval = self.interval

        signal_matrix = signal.view(T, 1) * torch.ones(1, T, device=device)

        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(dim=time_dim)
        else:
            pad_value = padding

        signal_pad = torch.ones(interval[1] + 1, T, device=device) * pad_value
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)

        subsignal_mask = torch.tril(torch.ones(T + interval[1] + 1, T, device=device))
        time_interval_mask = torch.triu(torch.ones(T + interval[1] + 1, T, device=device), -interval[-1]) * torch.tril(torch.ones(T + interval[1] + 1, T, device=device), -interval[0])

        masked_signal_matrix = torch.where((time_interval_mask * subsignal_mask).bool(), signal_padded, torch.tensor(mask_value, dtype=signal_padded.dtype, device=device))

        return self.operation(masked_signal_matrix, scale=large_number, dim=time_dim, keepdim=False, **kwargs)

    def forward(self, signal: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.robustness_trace(signal, **kwargs)



class trial(STL_Formula):
    def __init__(self):
        super().__init__()
        # self.lt = LessThan('x', torch.tensor([0.5]))
        # self.gt = GreaterThan('x', torch.tensor([0.5]))
        # # self.neg = Negation(self.lt)
        # self.and_ = And(self.gt, self.lt)

        self.always = Always(GreaterThan('x', torch.tensor([0.5]).to('cuda')))

        self.eventually = Eventually(GreaterThan('x', torch.tensor([0.5]).to('cuda')))

    def robustness_trace(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        # x = self.lt(signal)
        # y_1 = self.gt(signal)
        # y_2 = self.neg(signal)

        # y = self.and_([signal, signal])

        y = self.always(signal)
        #y = self.eventually(signal)

        return y

    def forward(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        return self.robustness_trace(signal, **kwargs)



if __name__ == "__main__":
    x = torch.tensor([0.6, 0.3, 0.2, 0.5, 0.6, 0.7])
    model = trial()
    y = model(x)
    print(y)


    # # y = model(x)

    # # gt_model = GreaterThan('x', torch.tensor([0.5]))
    # # y_gt = gt_model(y)
    

    # dot = make_dot(y, params=dict(model.named_parameters()))    
    # dot.render("computational_graph", format="png")