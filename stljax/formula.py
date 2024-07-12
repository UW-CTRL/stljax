import jax
import jax.numpy as jnp
import warnings
warnings.simplefilter("default")


def bar_plus(signal, p=2):
    return jax.nn.relu(signal) ** p


def bar_minus(signal, p=2):
    return (-jax.nn.relu(-signal)) ** p


def M0(signal, eps, weights=None, axis=1, keepdims=True):
    if weights is None:
        weights = jnp.ones_like(signal)
    sum_w = weights.sum(axis)
    return (
        eps**sum_w + jnp.prod(signal**weights, axis=axis, keepdims=keepdims)
    ) ** (1 / sum_w)


def Mp(signal, eps, p, weights=None, axis=1, keepdims=True):
    if weights is None:
        weights = jnp.ones_like(signal)
    sum_w = weights.sum(axis)
    return (
        eps**p + 1 / sum_w * jnp.sum(weights * signal**p, axis=axis, keepdims=keepdims)
    ) ** (1 / p)


def gmsr_min(signal, eps, p, weights=None, axis=1, keepdims=True):
    return (
        M0(bar_plus(signal, 2), eps, weights=weights, axis=axis, keepdims=keepdims)
        ** 0.5
        - Mp(
            bar_minus(signal, 2), eps, p, weights=weights, axis=axis, keepdims=keepdims
        )
        ** 0.5
    )


def gmsr_max(signal, eps, p, weights=None, axis=1, keepdims=True):
    return -gmsr_min(-signal, eps, p, weights=weights, axis=axis, keepdims=keepdims)


def gmsr_min_fast(signal, eps, p):
    # TODO: (norrisg) allow `axis` specification

    # Split indices into positive and non-positive values
    pos_idx = signal > 0.0
    neg_idx = ~pos_idx

    weights = jnp.ones_like(signal)

    # Sum of all weights
    sum_w = weights.sum()

    # If there exists a negative element
    if signal[neg_idx].size > 0:
        sums = 0.0
        sums = jnp.sum(weights[neg_idx] * (signal[neg_idx] ** (2 * p)))
        Mp = (eps**p + (sums / sum_w)) ** (1 / p)
        h_min = eps**0.5 - Mp**0.5

    # If all values are positive
    else:
        mult = 1.0
        mult = jnp.prod(signal[pos_idx] ** (2 * weights[pos_idx]))
        M0 = (eps**sum_w + mult) ** (1 / sum_w)
        h_min = M0**0.5 - eps**0.5

    return jnp.reshape(h_min, (1, 1, 1))


def gmsr_max_fast(signal, eps, p):
    return -gmsr_min_fast(-signal, eps, p)


def maxish(signal, axis, keepdims=True, approx_method="true", temperature=None):
    if isinstance(signal, Expression):
        assert (
            signal.value is not None
        ), "Input Expression does not have numerical values"
        signal = signal.value

    match approx_method:
        case "true":
            """jax keeps track of multiple max values and will distribute the gradients across all max values!
            e.g., jax.grad(jnp.max)(jnp.array([0.01, 0.0, 0.01])) # --> Array([0.5, 0. , 0.5], dtype=float32)
            """
            return jnp.max(signal, axis, keepdims=keepdims)

        case "logsumexp":
            """https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.special.logsumexp.html"""
            assert temperature is not None, "need a temperature value"
            return (
                jax.scipy.special.logsumexp(
                    temperature * signal, axis=axis, keepdims=keepdims
                )
                / temperature
            )

        case "softmax":
            assert temperature is not None, "need a temperature value"
            return (jax.nn.softmax(temperature * signal, axis) * signal).sum(
                axis, keepdims=keepdims
            )

        case "gmsr":
            assert (
                temperature is not None
            ), "temperature tuple containing (eps, p) is required"
            (eps, p) = temperature
            return gmsr_max(signal, eps, p, axis=axis, keepdims=keepdims)

        case _:
            raise ValueError("Invalid approx_method")


def minish(signal, axis, keepdims=True, approx_method="true", temperature=None):
    return -maxish(-signal, axis, keepdims, approx_method, temperature)


class STL_Formula:

    def __init__(self):
        super(STL_Formula, self).__init__()

    def robustness_trace(self, signal, **kwargs):
        """ Computes the robustness trace of the formula given an input signal. 
        Outputs: tensor [bs, time_dim,...]
        """
        raise NotImplementedError("robustness_trace not yet implemented")

    def robustness(self, signal, time_dim, **kwargs):
        """
        Extracts the robustness_trace value at the given time.
        Assumes signal is time reversed

        """
        # np.rollaxis(a, axis)[state]
        return jnp.rollaxis(self.__call__(signal, **kwargs), time_dim)[-1]

    def eval_trace(self, signal, **kwargs):
        """ The values in eval_trace are 0 or 1 (False or True) """
        return self.__call__(signal, **kwargs) > 0

    def eval(self, signal, time_dim, **kwargs):
        """
        Extracts the eval_trace value at the given time.
        Default: time=0 assuming this is the index for the NON-REVERSED trace. But the code will take it from the end since the input signal is TIME REVERSED.
        """
        return self.robustness(signal, time_dim, **kwargs) > 0


    def __call__(self, signal, **kwargs):
        """
        Extracts the robustness_trace value at the given time.
        Default: time=0 assuming this is the index for the NON-REVERSED trace. But the code will take it from the end since the input signal is TIME REVERSED.

        """
        
        """
        Evaluates the robustness_trace given the input. The input is converted to the numerical value first.
        """
        if isinstance(signal, Expression):
            assert signal.value is not None, "Input Expression does not have numerical values"
            if not signal.reverse:
                warnings.warn("Input Expression \"{input_name}\" is not time reversed".format(input_name=signal.name))
            signal = signal.value
            return self.robustness_trace(signal, **kwargs)
        elif isinstance(signal, jax.Array):
            return self.robustness_trace(signal, **kwargs)
        elif isinstance(signal, tuple):
            """ Accounts for the case that the formula requires two signal (e.g., And) """
            return self.robustness_trace(convert_to_input_values(signal), **kwargs)
        else:
            raise ValueError("Not a invalid input trace")
        
    def _next_function(self):
        """Function to keep track of the subformulas. For visualization purposes"""
        raise NotImplementedError("_next_function not year implemented")
    
    def __str__(self):
        raise NotImplementedError("__str__ not yet implemented")

    """ Overwriting some built-in functions for notational simplicity """
    def __and__(self, psi):
        return And(self, psi)

    def __or__(self, psi):
        return Or(self, psi)

    def __invert__(self):
        return Negation(self)


class LessThan(STL_Formula):
    """
    The LessThan predicate  (signal) < c 
    lhs < val where lhs is a placeholder for a signal, and val is the constant.
    lhs can be a string or an Expression
    val can be a float, int, Expression, or tensor. It cannot be a string.
    """
    def __init__(self, lhs='x', val='c'):
        super().__init__()
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "val on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val
        self.subformula = None

    def robustness_trace(self, trace, predicate_scale=1.0, **kwargs):
        """
        Computing robustness trace of trace < c 
        Optional: scale the robustness by a factor predicate_scale. Default: 1.0
        """
        if isinstance(trace, Expression):
            trace = trace.value
        if isinstance(self.val, Expression):
            return (self.val.value - trace) * predicate_scale
        else:
            return (self.val - trace) * predicate_scale

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.lhs, self.val]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        if isinstance(self.val, str): # could be a string if robustness_trace is never called
            return lhs_str + " < " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " < " + self.val.name
        if isinstance(self.val, jax.Array):
            return lhs_str + " < " + str(self.val)
        # if self.value is a single number (e.g., int, or float)
        return lhs_str + " < " + str(self.val)

class GreaterThan(STL_Formula):
    """
    The GreaterThan predicate  (signal) > c 
    lhs > val where lhs is a placeholder for a signal, and val is the constant.
    lhs can be a string or an Expression
    val can be a float, int, Expression, or tensor. It cannot be a string.
    """
    def __init__(self, lhs='x', val='c'):
        super().__init__()
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "val on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val
        self.subformula = None

    def robustness_trace(self, trace, predicate_scale=1.0, **kwargs):
        """
        Computing robustness trace of trace > c 
        Optional: scale the robustness by a factor predicate_scale. Default: 1.0
        """
        if isinstance(trace, Expression):
            trace = trace.value
        if isinstance(self.val, Expression):
            return -(self.val.value - trace) * predicate_scale
        else:
            return -(self.val - trace) * predicate_scale

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.lhs, self.val]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        if isinstance(self.val, str): # could be a string if robustness_trace is never called
            return lhs_str + " > " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " > " + self.val.name
        if isinstance(self.val, jax.Array):
            return lhs_str + " > " + str(self.val)
        # if self.value is a single number (e.g., int, or float)
        return lhs_str + " > " + str(self.val)


class Equal(STL_Formula):
    """
    The Equal predicate  (signal) == c 
    lhs == val where lhs is a placeholder for a signal, and val is the constant.
    lhs can be a string or an Expression
    val can be a float, int, Expression, or tensor. It cannot be a string.
    """
    def __init__(self, lhs='x', val='c'):
        super().__init__()
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "val on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val
        self.subformula = None

    def robustness_trace(self, trace, predicate_scale=1.0, **kwargs):
        """
        Computing robustness trace of trace == c 
        Optional: scale the robustness by a factor predicate_scale. Default: 1.0
        """
        if isinstance(trace, Expression):
            trace = trace.value
        if isinstance(self.val, Expression):
            return -jnp.abs(self.val.value - trace) * predicate_scale
        else:
            return -jnp.abs(self.val - trace) * predicate_scale

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.lhs, self.val]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression):
            lhs_str = self.lhs.name
        if isinstance(self.val, str): # could be a string if robustness_trace is never called
            return lhs_str + " == " + self.val
        if isinstance(self.val, Expression):
            return lhs_str + " == " + self.val.name
        if isinstance(self.val, jax.Array):
            return lhs_str + " == " + str(jax.Array)
        # if self.value is a single number (e.g., int, or float)
        return lhs_str + " == " + str(self.val)


class Negation(STL_Formula):
    """
    The Negation STL formula ¬
    Negates the subformula.

    Input
    subformula --- the subformula to be negated
    """
    def __init__(self, subformula):
        super(Negation, self).__init__()
        self.subformula = subformula

    def robustness_trace(self, trace, **kwargs):
        """
        Computing robustness trace of ¬(subformula)
        """
        return -self.subformula(trace, **kwargs)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

    def __str__(self):
        return "¬(" + str(self.subformula) + ")"


class And(STL_Formula):
    """
    The And STL formula ∧ (subformula1 ∧ subformula2)
    Input
    subformula1 --- subformula for lhs of the And operation
    subformula2 --- subformula for rhs of the And operation
    """
    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    @staticmethod
    def separate_and(formula, input_, **kwargs):
        """
        Function of seperate out multiple And operations. e.g., ϕ₁ ∧ ϕ₂ ∧ ϕ₃ ∧ ϕ₄ ∧ ...    
        """
        if formula.__class__.__name__ != "And":
            return jnp.expand_dims(formula(input_, **kwargs), -1)
        else:
            return jnp.concatenate([And.separate_and(formula.subformula1, input_[0], **kwargs), And.separate_and(formula.subformula2, input_[1], **kwargs)], axis=-1)

    def robustness_trace(self, inputs, **kwargs):
        """
        Computing robustness trace of subformula1 ∧ subformula2
        Input
        inputs --- a tuple of input traces corresponding to each subformula respectively. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs (e.g, ϕ₁(x) ∧ (ϕ₂(y) ∧ ϕ₃(z)) would have inputs=(x, (y,z)))
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        """
        xx = And.separate_and(self, inputs, **kwargs)
        return minish(xx, axis=-1, keepdims=False, **kwargs)                                         # [batch_size, time_dim, ...]

    def _next_function(self):
        """ next function is the input subformulas. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ∧ (" + str(self.subformula2) + ")"


class Or(STL_Formula):
    """
    The Or STL formula ∨ (subformula1 ∨ subformula2)
    Input
    subformula1 --- subformula for lhs of the And operation
    subformula2 --- subformula for rhs of the And operation
    """
    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    @staticmethod
    def separate_and(formula, input_, **kwargs):
        """
        Function of seperate out multiple And operations. e.g., ϕ₁ ∨ ϕ₂ ∨ ϕ₃ ∨ ϕ₄ ∨ ...    
        """
        if formula.__class__.__name__ != "Or":
            return jnp.expand_dims(formula(input_, **kwargs), -1)
        else:
            return jnp.concatenate([Or.separate_and(formula.subformula1, input_[0], **kwargs), Or.separate_and(formula.subformula2, input_[1], **kwargs)], axis=-1)

    def robustness_trace(self, inputs, **kwargs):
        """
        Computing robustness trace of subformula1 ∧ subformula2
        Input
        inputs --- a tuple of input traces corresponding to each subformula respectively. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs (e.g, ϕ₁(x) ∨ (ϕ₂(y) ∨ ϕ₃(z)) would have inputs=(x, (y,z)))
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        """
        xx = Or.separate_and(self, inputs, **kwargs)
        return maxish(xx, axis=-1, keepdims=False, **kwargs)                                         # [batch_size, time_dim, ...]

    def _next_function(self):
        """ next function is the input subformulas. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ∨ (" + str(self.subformula2) + ")"

class Temporal_Operator(STL_Formula):
    """
    Class to compute Eventually and Always. This builds a recurrent cell to perform dynamic programming
    
    Inputs
    subformula --- The subformula that the temporal operator is applied to.
    interval   --- The time interval that the temporal operator operates on. 
                   Default: None which means [0, jnp.inf]. Other options car be: [a, b] (b < jnp.inf), [a, jnp.inf] (a > 0)
    NOTE: Assume that the interval is describing the INDICES of the desired time interval. The user is responsible for converting the time interval (in time units) into indices (integers) using knowledge of the time step size.
    """
    def __init__(self, subformula, interval=None):
        super().__init__()
        self.subformula = subformula
        self.interval = interval
        self._interval = [0, jnp.inf] if self.interval is None else self.interval
        self.hidden_dim = 1 if not self.interval else self.interval[-1]    # hidden_dim=1 if interval is [0, ∞) otherwise hidden_dim=end of interval
        if self.hidden_dim == jnp.inf:
            self.hidden_dim = self.interval[0]
        self.steps = 1 if not self.interval else self.interval[-1] - self.interval[0] + 1   # steps=1 if interval is [0, ∞) otherwise steps=length of interval
        self.operation = None
        # Matrices that shift a vector and add a new entry at the end.
        self.M = jnp.diag(jnp.ones(self.hidden_dim-1), k=1)
        self.b = jnp.expand_dims(jnp.zeros(self.hidden_dim), -1)
        self.b = self.b.at[-1].set(1)


    def _initialize_hidden_state(self, signal):

        """
        Compute the initial hidden state.
        
        Input
        signal --- the input signal [batch_size, time_dim, signal_dim]

        Output
        h0 --- initial hidden state is [batch_size, hidden_dim, signal_dim]

        Notes:
        Initializing the hidden state requires padding on the signal. Currently, the default is to extend the last value.
        TODO: have option on this padding

        """
        # Case 1, 2, 4
        # TODO: make this less hard-coded. Assumes signal is [bs, time_dim, signal_dim], and already reversed
        h0 = jnp.ones([signal.shape[0], self.hidden_dim, signal.shape[2]])*signal[:,:1,:]

        # Case 3: if self.interval is [a, jnp.inf), then the hidden state is a tuple (like in an LSTM)
        if (self._interval[1] == jnp.inf) & (self._interval[0] > 0):
            c0 = signal[:,:1,:]
            return (c0, h0)
        return h0

    def _cell(self, x, hidden_state, time_dim=1, **kwargs):
        """
        This function describes the operation that takes place at each recurrent step.
        Input: 
        x  --- the input state at time t [batch_size, 1, ...]
        hidden_state --- the hidden state. It is either a tensor, or a tuple of tensors, depending on the interval chosen and other arguments. Generally, the hidden state is of size [batch_size, hidden_dim,...]
        """
        raise NotImplementedError("_cell is not implemented")


    def _run_cell(self, signal, time_dim=1, **kwargs):
        """
        Function to run a signal through a cell T times, where T is the length of the signal in the time dimension
        Input
        signal      --- input signal, size = [bs, time_dim, ...]
        scale       --- scale factor used for the minish/maxish function
        distributed --- Boolean to indicate whether to distribute gradients over multiple min/max values. Default: False

        """

        outputs = []
        states = []
        hidden_state = self._initialize_hidden_state(signal)                               # [batch_size, hidden_dim, x_dim]
        signal_split = jnp.split(signal, signal.shape[time_dim], time_dim)    # list of x at each time step
        for i in range(signal.shape[time_dim]):
            o, hidden_state = self._cell(signal_split[i], hidden_state, time_dim, **kwargs)
            outputs.append(o)
            states.append(hidden_state)
        return outputs, states


    def robustness_trace(self, inputs, time_dim=1, **kwargs):
        """
        Function to compute robustness trace of a temporal STL formula
        First, compute the robustness trace of the subformula, and use that as the input for the recurrent computation
        """
        
        trace = self.subformula(inputs, **kwargs)
        outputs, _ = self._run_cell(trace, time_dim, **kwargs)
        return jnp.concatenate(outputs, axis=time_dim)                              # [batch_size, time_dim, ...]

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]


class Always(Temporal_Operator):
    """
    The Always STL formula □_[a,b] 
    The robustness value is the minimum value of the input trace over a prespecified time interval

    Input
    subformula --- subformula that the always operation is applied on
    interval   --- time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep is.
    """
    def __init__(self, subformula, interval=None):
        super().__init__(subformula=subformula, interval=interval)

    def _cell(self, x, hidden_state, time_dim, **kwargs):
        """
        This function describes the operation that takes place at each recurrent step.
        Input: 
        x  --- the input state at time t [batch_size, 1, ...]
        hidden_state --- the hidden state is of size [batch_size, rnn_dim,...]
        """
        
        # Case 1, interval = [0, inf]
        if self.interval is None:
            input_ = jnp.concatenate([hidden_state, x], axis=time_dim)                # [batch_size, rnn_dim+1, x_dim]
            output = minish(input_, time_dim, keepdims=True, **kwargs)       # [batch_size, 1, x_dim]
            return output, output

        # Case 3: self.interval is [a, np.inf)
        if (self._interval[1] == np.inf) & (self._interval[0] > 0):
            c, h = hidden_state
            ch = jnp.concatenate([c, h[:,:1,:]], axis=time_dim)                             # [batch_size, 2, x_dim]
            output = minish(ch, time_dim, keepdims=True, **kwargs)               # [batch_size, 1, x_dim]
            hidden_state = (output, self.M @ h + self.b * x)
        # Case 2 and 4: self.interval is [a, b]
        else: 
            
            hidden_state = self.M @ hidden_state + self.b * x
            hx = jnp.concatenate([hidden_state, x], axis=time_dim)                             # [batch_size, rnn_dim+1, x_dim]
            input_ = hx[:,:self.steps,:]                               # [batch_size, self.steps, x_dim]
            output = minish(input_, time_dim, **kwargs)               # [batch_size, 1, x_dim]
        return output, hidden_state

    def __str__(self):
        return "◻ " + str(self._interval) + "( " + str(self.subformula) + " )"


class Eventually(Temporal_Operator):
    """
    The Eventually STL formula □_[a,b] 
    The robustness value is the minimum value of the input trace over a prespecified time interval

    Input
    subformula --- subformula that the Eventually operation is applied on
    interval   --- time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep is.
    """
    def __init__(self, subformula, interval=None):
        super().__init__(subformula=subformula, interval=interval)

    def _cell(self, x, hidden_state, time_dim, **kwargs):
        """
        This function describes the operation that takes place at each recurrent step.
        Input: 
        x  --- the input state at time t [batch_size, 1, ...]
        hidden_state --- the hidden state is of size [batch_size, rnn_dim,...]
        """
        
        # Case 1, interval = [0, inf]
        if self.interval is None:
            input_ = jnp.concatenate([hidden_state, x], axis=time_dim)                # [batch_size, rnn_dim+1, x_dim]
            output = maxish(input_, time_dim, keepdims=True, **kwargs)       # [batch_size, 1, x_dim]
            return output, output

        # Case 3: self.interval is [a, np.inf)
        if (self._interval[1] == jnp.inf) & (self._interval[0] > 0):
            c, h = hidden_state
            ch = jnp.concatenate([c, h[:,:1,:]], axis=time_dim)                             # [batch_size, 2, x_dim]
            output = maxish(ch, time_dim, keepdims=True, **kwargs)               # [batch_size, 1, x_dim]
            hidden_state = (output, self.M @ h + self.b * x)
        # Case 2 and 4: self.interval is [a, b]
        else: 
            
            hidden_state = self.M @ hidden_state + self.b * x
            hx = jnp.concatenate([hidden_state, x], axis=time_dim)                             # [batch_size, rnn_dim+1, x_dim]
            input_ = hx[:,:self.steps,:]                               # [batch_size, self.steps, x_dim]
            output = maxish(input_, time_dim, **kwargs)               # [batch_size, 1, x_dim]
        return output, hidden_state

    def __str__(self):
        return "♢ " + str(self._interval) + "( " + str(self.subformula) + " )"


class Expression:
    name: str
    value: jnp.array
    reverse: bool

    def __init__(self, name, value, reverse):
        self.value = value
        self.name = name
        self.reverse = reverse

    def set_name(self, new_name):
        self.name = new_name

    def set_value(self, new_value):
        self.value = new_value

    def flip(self, dim):
        self.value = jnp.flip(self.value, dim)
        self.reverse = not self.reverse
        return self.value
    
    def __neg__(self):
        return Expression('-' + self.name, -self.value, self.reverse)
    
    def __add__(self, other):
        if isinstance(other, Expression):
            if self.reverse == other.reverse:
                return Expression(self.name + ' + ' + other.name, self.value + other.value, self.reverse)
            else:
                raise ValueError("reverse attribute do not match")
        else:
            return Expression(self.name + ' + other', self.value + other, self.reverse)

    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, Expression):
            if self.reverse == other.reverse:
                return Expression(self.name + ' - ' + other.name, self.value - other.value, self.reverse)
            else:
                raise ValueError("reverse attribute do not match")        
        else:
            return Expression(self.name + " - other", self.value - other, self.reverse)

    def __rsub__(self, other):
        return self.__sub__(other)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular sub

    def __mul__(self, other):
        if isinstance(other, Expression):
            if self.reverse == other.reverse:
                return Expression(self.name + ' × ' + other.name, self.value * other.value, self.reverse)
            else:
                raise ValueError("reverse attribute do not match")    
        else:
            return Expression(self.name + " * other", self.value * other, self.reverse)


    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(a, b):
        # This is the new form required by Python 3
        numerator = a
        denominator = b
        if numerator.reverse == denominator.reverse:
            return Expression(numerator.name + '/' + denominator.name, numerator.value/denominator.value, denominator.reverse)
        else:
            raise ValueError("reverse attribute do not match") 
        

    # Comparators
    def __lt__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of LessThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs)

    def __le__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of LessThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs)

    def __gt__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of GreaterThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)

    def __ge__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of GreaterThan needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)

    def __eq__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of Equal needs to be a string or Expression"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return Equal(lhs, rhs)

    # def __ne__(lhs, rhs):
    #     raise NotImplementedError("Not supported yet")

    # def __str__(self):
    #     return str(self.name)
    
    def __call__(self):
        return self.value


def convert_to_input_values(inputs):
    x_, y_ = inputs
    if isinstance(x_, Expression):
        assert x_.value is not None, "Input Expression does not have numerical values"
        x_ret = x_.value
    elif isinstance(x_, jax.Array):
        x_ret = x_
    elif isinstance(x_, tuple):
        x_ret = convert_to_input_values(x_)
    else:
        raise ValueError("First argument is an invalid input trace")

    if isinstance(y_, Expression):
        assert y_.value is not None, "Input Expression does not have numerical values"
        y_ret = y_.value
    elif isinstance(y_, jax.Array):
        y_ret = y_
    elif isinstance(y_, tuple):
        y_ret = convert_to_input_values(y_)
    else:
        raise ValueError("Second argument is an invalid input trace")

    return (x_ret, y_ret)
