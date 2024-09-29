import torch
import numpy as np
import os
import sys
from stljax.base_types import Expression, Minish, Maxish
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

    def robustness(self, signal, time_dim=0, **kwargs):
        '''
        Extracts the robustness_trace value at the given time.
        Default: time=0 assuming this is the index for the NON-REVERSED trace. But the code will take it from the end since the input signal is TIME REVERSED.

        '''
        return self.robustness_trace(signal, **kwargs)[0]
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
            interval = [0, T-1]
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

        return self.operation(masked_signal_matrix, dim=time_dim, keepdim=False, **kwargs)

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
            interval = [0, T-1]
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

        return self.operation(masked_signal_matrix, dim=time_dim, keepdim=False, **kwargs)

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

class Temporal_Operator(STL_Formula):
    """
    Class to compute Eventually and Always. This builds a recurrent cell to perform dynamic programming

    Args:
        subformula: The subformula that the temporal operator is applied to.
        interval: The time interval that the temporal operator operates on. Default: None which means [0, torch.inf]. Other options car be: [a, b] (b < torch.inf), [a, torch.inf] (a > 0)

    NOTE: Assume that the interval is describing the INDICES of the desired time interval. The user is responsible for converting the time interval (in time units) into indices (integers) using knowledge of the time step size.
    """
    def __init__(self, subformula, interval=None, device="cpu"):
        super().__init__()
        self.subformula = subformula
        self.interval = interval
        self.device = device
        self._interval = [0, torch.inf] if self.interval is None else self.interval
        self.hidden_dim = 1 if not self.interval else self.interval[-1]    # hidden_dim=1 if interval is [0, ∞) otherwise hidden_dim=end of interval
        if self.hidden_dim == torch.inf:
            self.hidden_dim = self.interval[0]
        self.steps = 1 if not self.interval else self.interval[-1] - self.interval[0] + 1   # steps=1 if interval is [0, ∞) otherwise steps=length of interval
        self.operation = None
        # Matrices that shift a vector and add a new entry at the end.
        self.M = torch.diag(torch.ones(self.hidden_dim-1, device=device), diagonal=1)
        self.b = torch.zeros(self.hidden_dim, device=device)
        self.b[-1] = 1.
        # self.b = self.b.at[-1].set(1)


    def _cell(self, x, hidden_state, time_dim=1, **kwargs):
        """
        This function describes the operation that takes place at each recurrent step.
        Args:
            x: the input state at time t [batch_size, 1, ...]
            hidden_state: the hidden state. It is either a tensor, or a tuple of tensors, depending on the interval chosen and other arguments. Generally, the hidden state is of size [batch_size, hidden_dim,...]

        Return:
            output and next hidden_state
        """
        raise NotImplementedError("_cell is not implemented")


    def _run_cell(self, signal, time_dim=0, **kwargs):
        """
        Function to run a signal through a cell T times, where T is the length of the signal in the time dimension

        Args:
            signal: input signal, size = [time_dim,]
            time_dim: axis corresponding to time_dim. Default: 0
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return:
            outputs: list of outputs
            states: list of hidden_states
        """
        hidden_state = self._initialize_hidden_state(signal)                               # [hidden_dim]
        outputs = []
        states = []

        signal_split = torch.split(signal, 1, time_dim)    # list of x at each time step
        for i in range(signal.shape[time_dim]):
            o, hidden_state = self._cell(signal_split[i], hidden_state, time_dim, **kwargs)
            outputs.append(o)
            states.append(hidden_state)
        return outputs, states

    def robustness_trace(self, signal, time_dim=0, **kwargs):
        """
        Function to compute robustness trace of a temporal STL formula
        First, compute the robustness trace of the subformula, and use that as the input for the recurrent computation

        Args:
            signal: input signal, size = [bs, time_dim, ...]
            time_dim: axis corresponding to time_dim. Default: 1
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: torch.array. Same size as signal.
        """

        trace = self.subformula(signal, **kwargs)
        outputs, _ = self._run_cell(trace, time_dim, **kwargs)
        return torch.concatenate(outputs, axis=time_dim)                              # [time_dim, ]

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]


class AlwaysRecurrent(Temporal_Operator):
    """
    The Always STL formula □_[a,b] subformula
    The robustness value is the minimum value of the input trace over a prespecified time interval

    Args:
        subformula: subformula that the Always operation is applied on
        interval: time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep size is.
    """
    def __init__(self, subformula, interval=None):
        super().__init__(subformula=subformula, interval=interval)
        self.operation = Minish()


    def _initialize_hidden_state(self, signal):
        """
        Compute the initial hidden state.

        Args:
            signal: the input signal. Expected size [time_dim,]

        Returns:
            h0: initial hidden state is [hidden_dim,]

        Notes:
        Initializing the hidden state requires padding on the signal. Currently, the default is to extend the last value.
        TODO: have option on this padding

        """
        # Case 1, 2, 4
        # TODO: make this less hard-coded. Assumes signal is [bs, time_dim, signal_dim], and already reversed
        # pads with the signal value at the last time step.
        h0 = torch.ones([self.hidden_dim, *signal.shape[1:]], device=self.device) * 1E3

        # Case 3: if self.interval is [a, torch.inf), then the hidden state is a tuple (like in an LSTM)
        if (self._interval[1] == torch.inf) & (self._interval[0] > 0):
            c0 = signal[:1]
            return (c0, h0)
        return h0

    def _cell(self, x, hidden_state, time_dim, **kwargs):
        """
        see Temporal_Operator._cell
        """
        # Case 1, interval = [0, inf]
        if self.interval is None:
            input_ = torch.concatenate([hidden_state, x], axis=time_dim)                # [rnn_dim+1,]
            output = self.operation(input_, dim=time_dim, **kwargs)       # [1,]
            return output, output

        # Case 3: self.interval is [a, np.inf)
        if (self._interval[1] == torch.inf) & (self._interval[0] > 0):
            c, h = hidden_state
            ch = torch.concatenate([c, h[:1]], axis=time_dim)                             # [2,]
            output = self.operation(ch, dim=time_dim, **kwargs)               # [1,]
            hidden_state_ = (output, self.M @ h + self.b * x)

        # Case 2 and 4: self.interval is [a, b]
        else:
            hidden_state_ = self.M @ hidden_state + self.b * x
            hx = torch.concatenate([hidden_state, x], axis=time_dim)                             # [rnn_dim+1,]
            input_ = hx[:self.steps]                               # [self.steps,]
            output = self.operation(input_, dim=time_dim, **kwargs)               # [1,]
        return output, hidden_state_

    def __str__(self):
        return "◻ " + str(self._interval) + "( " + str(self.subformula) + " )"

class EventuallyRecurrent(Temporal_Operator):
    """
    The Eventually STL formula □_[a,b] subformula
    The robustness value is the minimum value of the input trace over a prespecified time interval

    Args:
        subformula: subformula that the Eventually operation is applied on
        interval: time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep size is.
    """
    def __init__(self, subformula, interval=None):
        super().__init__(subformula=subformula, interval=interval)
        self.operation = Maxish()


    def _initialize_hidden_state(self, signal):
        """
        Compute the initial hidden state.

        Args:
            signal: the input signal. Expected size [time_dim,]

        Returns:
            h0: initial hidden state is [hidden_dim,]

        Notes:
        Initializing the hidden state requires padding on the signal. Currently, the default is to extend the last value.
        TODO: have option on this padding

        """
        # Case 1, 2, 4
        # TODO: make this less hard-coded. Assumes signal is [bs, time_dim, signal_dim], and already reversed
        # pads with the signal value at the last time step.
        h0 = torch.ones([self.hidden_dim, *signal.shape[1:]], device=self.device) * -1E3

        # Case 3: if self.interval is [a, torch.inf), then the hidden state is a tuple (like in an LSTM)
        if (self._interval[1] == torch.inf) & (self._interval[0] > 0):
            c0 = signal[:1]
            return (c0, h0)
        return h0

    def _cell(self, x, hidden_state, time_dim, **kwargs):
        """
        see Temporal_Operator._cell
        """
        # Case 1, interval = [0, inf]
        if self.interval is None:
            input_ = torch.concatenate([hidden_state, x], axis=time_dim)                # [rnn_dim+1,]
            output = self.operation(input_, dim=time_dim, **kwargs)       # [1,]
            return output, output

        # Case 3: self.interval is [a, np.inf)
        if (self._interval[1] == torch.inf) & (self._interval[0] > 0):
            c, h = hidden_state
            ch = torch.concatenate([c, h[:1]], axis=time_dim)                             # [2,]
            output = self.operation(ch, dim=time_dim, **kwargs)               # [1,]
            hidden_state_ = (output, self.M @ h + self.b * x)

        # Case 2 and 4: self.interval is [a, b]
        else:
            hidden_state_ = self.M @ hidden_state + self.b * x
            hx = torch.concatenate([hidden_state, x], axis=time_dim)                             # [rnn_dim+1,]
            input_ = hx[:self.steps]                               # [self.steps,]
            output = self.operation(input_, dim=time_dim, **kwargs)               # [1,]
        return output, hidden_state_

    def __str__(self):
        return "♢ " + str(self._interval) + "( " + str(self.subformula) + " )"


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