import torch
import numpy as np
import os
import sys
from .base_types import Expression, Minish, Maxish

# Set the device based on runtime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_input_values(inputs):
    '''
    This function is supposed to do all the necessary conversions to the input signal. Currently, it just returns the inputs as it is.
    '''
    return inputs

class STL_Formula(torch.nn.Module):
    def __init__(self):
        super(STL_Formula, self).__init__()

    def robustness_trace(self, signal, **kwargs):
        raise NotImplementedError("robustness_trace not yet implemented")

    def robustness(self, signal, time_dim, **kwargs):
        pass
    
    def forward(self, inputs, **kwargs):
        inputs = convert_to_input_values(inputs)
        return self.robustness_trace(inputs, **kwargs)

class LessThan(STL_Formula):
    def __init__(self, lhs='x', val='c'):
        super().__init__()
        assert isinstance(lhs, str) or isinstance(lhs, Expression), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "val on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val

    def robustness_trace(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        """
        Batched input handling: Assumes signal of shape [batch_size, time_dim, ...]
        """
        if isinstance(self.val, Expression):
            assert self.val.value is not None, "Expression does not have numerical values"
            c_val = self.val.value.to(device)
        else:
            c_val = torch.tensor(self.val, device=device)

        # Ensure that c_val is broadcastable to signal's shape
        c_val = c_val.expand_as(signal) if not torch.is_tensor(c_val) or c_val.dim() == 0 else c_val

        return (c_val - signal.to(device))

    def forward(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        return self.robustness_trace(signal, **kwargs)

class GreaterThan(STL_Formula):
    def __init__(self, lhs='x', val='c'):
        super().__init__()
        assert isinstance(lhs, str) or isinstance(lhs, Expression), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "val on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val

    def robustness_trace(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        """
        Batched input handling: Assumes signal of shape [batch_size, time_dim, ...]
        """
        if isinstance(self.val, Expression):
            assert self.val.value is not None, "Expression does not have numerical values"
            c_val = self.val.value.to(device)
        else:
            c_val = torch.tensor(self.val, device=device)

        # Ensure that c_val is broadcastable to signal's shape
        c_val = c_val.expand_as(signal) if not torch.is_tensor(c_val) or c_val.dim() == 0 else c_val

        return (signal.to(device) - c_val)

    def forward(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        return self.robustness_trace(signal, **kwargs)

class Negation(STL_Formula):
    def __init__(self, subformula):
        super().__init__()
        self.subformula = subformula

    def robustness_trace(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        """
        Batched input handling: Assumes signal of shape [batch_size, time_dim, ...]
        """
        return -self.subformula(signal.to(device))

    def forward(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        return self.robustness_trace(signal, **kwargs)

class And(STL_Formula):
    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.operation = Minish()

    def robustness_trace(self, inputs, **kwargs):
        """
        Batched input handling: Assumes signal of shape [batch_size, time_dim, ...]
        """

        pre = self.subformula1(inputs[0].to(device))
        post = self.subformula2(inputs[1].to(device))

        # Stack along a new dimension for batch operations
        temp = torch.stack([pre, post], dim=-1)

        # Minish operation across the new dimension
        return self.operation(temp, scale=100000, dim=-1, keepdim=False, **kwargs)

    def forward(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        return self.robustness_trace(signal, **kwargs)

class Always(STL_Formula):
    def __init__(self, subformula, interval=None):
        super().__init__()
        self.interval = interval
        self.operation = Minish()  
        self.subformula = subformula

    def robustness_trace(self, signal, time_dim=1, padding="last", large_number=1e9, **kwargs):
        signal = self.subformula(signal.to(device), time_dim=time_dim, padding=padding, large_number=large_number, **kwargs)
        batch_size, T = signal.shape[0], signal.shape[time_dim]
        mask_value = large_number

        if self.interval is None:
            interval = [0, T]
        else:
            interval = self.interval

        signal_matrix = signal.unsqueeze(2) * torch.ones(1, 1, T, device=device)

        if padding == "last":
            pad_value = signal[:, -1].unsqueeze(1)
        elif padding == "mean":
            pad_value = signal.mean(dim=time_dim).unsqueeze(1)
        else:
            pad_value = padding

        signal_pad = pad_value.unsqueeze(1).expand(-1, interval[1] + 1, T)
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)

        subsignal_mask = torch.tril(torch.ones(T + interval[1] + 1, T, device=device))
        time_interval_mask = torch.triu(torch.ones(T + interval[1] + 1, T, device=device), -interval[-1]) * torch.tril(torch.ones(T + interval[1] + 1, T, device=device), -interval[0])

        masked_signal_matrix = torch.where((time_interval_mask * subsignal_mask).bool(), signal_padded, torch.tensor(mask_value, dtype=signal_padded.dtype, device=device))

        return self.operation(masked_signal_matrix, scale=large_number, dim=time_dim, keepdim=False, **kwargs)

    def forward(self, signal: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.robustness_trace(signal, **kwargs)


class Eventually(STL_Formula):
    def __init__(self, subformula, interval=None):
        super().__init__()
        self.interval = interval
        self.operation = Maxish()
        self.subformula = subformula

    def robustness_trace(self, signal, time_dim=1, padding="last", large_number=1e9, **kwargs):
        signal = self.subformula(signal.to(device), time_dim=time_dim, padding=padding, large_number=large_number, **kwargs)
        batch_size, T = signal.shape[0], signal.shape[time_dim]
        mask_value = -large_number

        if self.interval is None:
            interval = [0, T]
        else:
            interval = self.interval

        signal_matrix = signal.unsqueeze(2) * torch.ones(1, 1, T, device=device)

        if padding == "last":
            pad_value = signal[:, -1].unsqueeze(1)
        elif padding == "mean":
            pad_value = signal.mean(dim=time_dim).unsqueeze(1)
        else:
            pad_value = padding

        signal_pad = pad_value.unsqueeze(1).expand(-1, interval[1] + 1, T)
        signal_padded = torch.cat([signal_matrix, signal_pad], dim=time_dim)

        subsignal_mask = torch.tril(torch.ones(T + interval[1] + 1, T, device=device))
        time_interval_mask = torch.triu(torch.ones(T + interval[1] + 1, T, device=device), -interval[-1]) * torch.tril(torch.ones(T + interval[1] + 1, T, device=device), -interval[0])

        masked_signal_matrix = torch.where((time_interval_mask * subsignal_mask).bool(), signal_padded, torch.tensor(mask_value, dtype=signal_padded.dtype, device=device))

        return self.operation(masked_signal_matrix, scale=large_number, dim=time_dim, keepdim=False, **kwargs)

    def forward(self, signal: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.robustness_trace(signal, **kwargs)


class trial(STL_Formula):
    def __init__(self, x_val, y_val):
        super().__init__()
        #self.always = Always(GreaterThan('x', torch.tensor([0.5], device=device)))
        self.pred_x = GreaterThan('x', torch.tensor([0.0], device=device))
        self.pred_y = GreaterThan('y', torch.tensor([0.0], device=device))
        self.pred_x_ub = LessThan('x', torch.tensor([0.0], device=device))
        self.pred_y_ub = LessThan('y', torch.tensor([0.0], device=device))
        self.x_and = And(self.pred_x, self.pred_x_ub)
        self.y_and = And(self.pred_y, self.pred_y_ub)
        self.eventually_x = Eventually(self.x_and)
        self.eventually_y = Eventually(self.y_and)
        # self.dis = self.eventually_y
        # self.dis_x = self.eventually_x
        # self.eventually_y= Eventually(self.pred_y)
        # self.eventually_x_ub = Eventually(self.pred_x_ub)
        # self.eventually_y_ub = Eventually(self.pred_y_ub)
        # self.first = And(self.eventually_x_ub, self.eventually_y_ub)
        # self.second = And(self.eventually_x, self.eventually_y)
        self.dis = self.eventually_x
        self.dis_2 = self.eventually_y
    


    def robustness_trace(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        """
        Batched input handling: Assumes signal of shape [batch_size, time_dim, ...]
        """
        x = signal[:, :, 0]
        y = signal[:, :, 1]
        concatenated_1 = torch.stack([x, x], dim=0)
        concatenated_2 = torch.stack([y, y], dim=0)
        final = self.dis(concatenated_1)
        final_2 = self.dis_2(concatenated_2)
        return final,final_2

    def forward(self, signal:torch.Tensor, **kwargs)->torch.Tensor:
        return self.robustness_trace(signal, **kwargs)

if __name__ == "__main__":
    # Example input with batch_size=2, time_dim=6
    x = torch.tensor([[0.6, 0.3, 0.2, 0.5, 0.6, 0.7],
                      [0.4, 0.8, 0.1, 0.3, 0.2, 0.9]], device=device)
    y = torch.tensor([[0.6, 0.9, 0.9, 0.9, 0.9, 0.7],
                      [0.4, 0.8, 0.1, 0.3, 0.2, 0.9]], device=device)
    stack = torch.stack([x, y], dim=-1)
    print(stack.shape)
    model = trial(0.5, 0.5)
    x1, x2 = model(stack)
    scale = 10000.0  # Scale factor for smooth minimum approximation

    # Stack x1 and x2 along a new dimension for logsumexp
    stacked = torch.stack([-scale * x1, -scale * x2], dim=0)

    # Apply logsumexp over the stacked tensors along the new dimension (dim=0)
    smooth_min = -torch.logsumexp(stacked, dim=0) / scale

