import jax
import jax.numpy as jnp
import warnings
from typing import Callable
import functools

warnings.simplefilter("default")
from .utils import *

class STL_Formula:
    '''
    Class for an STL formula
    NOTE: If Expressions and Predicates are used, then the signals will be reversed if needed. Otherwise, user is responsibile for keeping track.
    '''
    def __init__(self):
        super(STL_Formula, self).__init__()

    def robustness_trace(self, signal, **kwargs):
        """
        Computes the robustness trace of the formula given an input signal.

        Args:
            signal: jnp.array or Expression. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: jnp.array of size equal to the input. index=0 along axis=time_dim is the robustness of the last subsignal. index=-1 along axis=time_dim is the robustness of the entire signal.
        """

        raise NotImplementedError("robustness_trace not yet implemented")

    def robustness(self, signal, **kwargs):
        """
        Computes the robustness value. Extracts the last entry along time_dim of robustness trace.

        Args:
            signal: jnp.array or Expression. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return: jnp.array, same as input with the time_dim removed.
        """
        return self.__call__(signal, **kwargs)[0]
        # return jnp.rollaxis(self.__call__(signal, **kwargs), time_dim)[-1]

    def eval_trace(self, signal, **kwargs):
        """
        Boolean of robustness_trace

        Args:
            signal: jnp.array or Expression. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            eval_trace: jnp.array of size equal to the input but with True/False. index=0 along axis=time_dim is the robustness of the last subsignal. index=-1 along axis=time_dim is the robustness of the entire signal.
        """

        return self.__call__(signal, **kwargs) > 0

    def eval(self, signal, **kwargs):
        """
        Boolean of robustness

        Args:
            signal: jnp.array or Expression. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return: jnp.array with True/False, same as input with the time_dim removed.
        """
        return self.robustness(signal, **kwargs) > 0


    def __call__(self, signal, **kwargs):
        """
        Evaluates the robustness_trace given the input. The input is converted to the numerical value first.

        See  STL_Formula.robustness_trace
        """

        inputs = convert_to_input_values(signal)
        return self.robustness_trace(inputs, **kwargs)

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

class Identity(STL_Formula):
    """ The identity formula. Use in Until"""
    def __init__(self, name='x'):
        super().__init__()
        self.name = name

    def robustness_trace(self, trace, **kwargs):
        return trace

    def _next_function(self):
        return []

    def __str__(self):
        return "%s" %self.name



class LessThan(STL_Formula):
    """
    The LessThan operation. lhs < val where lhs is a placeholder for a signal, and val is a constant.
    Args:
        lhs: string, Expression, or Predicate
        val: float, int, Expression, or array (of appropriate size). It cannot be a string.
    """
    def __init__(self, lhs='x', val='c'):
        super().__init__()
        assert isinstance(lhs, str) | isinstance(lhs, Expression) | isinstance(lhs, Predicate), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "val on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val

    def robustness_trace(self, signal, predicate_scale=1.0, **kwargs):
        """
        Computes robustness trace:  rhs - lhs
        Args:
            signal: jnp.array. Expected size [bs, time_dim, state_dim]
            predicate_scale: Optional. scale the robustness by a factor predicate_scale. Default: 1.0

        Returns:
            robustness_trace: jnp.array. Same size as signal.
        """
        if isinstance(self.val, Expression):
            assert self.val.value is not None, "Expression does not have numerical values"
            c_val = self.val.value

        else:
            c_val = self.val

        if isinstance(self.lhs, Predicate):
            return (c_val - self.lhs(signal)) * predicate_scale
        else:
            return (c_val - signal) * predicate_scale


    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.lhs, self.val]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression) | isinstance(self.lhs, Predicate):
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
    The GreaterThan operation. lhs > val where lhs is a placeholder for a signal, and val is a constant.
    Args:
        lhs: string, Expression, or Predicate
        val: float, int, Expression, or array (of appropriate size). It cannot be a string.
    """
    def __init__(self, lhs='x', val='c'):
        super().__init__()
        assert isinstance(lhs, str) | isinstance(lhs, Expression) | isinstance(lhs, Predicate), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "val on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val
        self.subformula = None

    def robustness_trace(self, trace, predicate_scale=1.0, **kwargs):
        """
        Computes robustness trace:  lhs - rhs
        Args:
            signal: jnp.array. Expected size [bs, time_dim, state_dim]
            predicate_scale: Optional. scale the robustness by a factor predicate_scale. Default: 1.0

        Returns:
            robustness_trace: jnp.array. Same size as signal.
        """
        if isinstance(self.val, Expression):
            assert self.val.value is not None, "Expression does not have numerical values"
            c_val = self.val.value

        else:
            c_val = self.val

        if isinstance(self.lhs, Predicate):
            return -(c_val - self.lhs(trace)) * predicate_scale
        else:
            return -(c_val - trace) * predicate_scale

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.lhs, self.val]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression) | isinstance(self.lhs, Predicate):
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
    The Equal operation. lhs == val where lhs is a placeholder for a signal, and val is a constant.
    Args:
        lhs: string, Expression, or Predicate
        val: float, int, Expression, or array (of appropriate size). It cannot be a string.
    """
    def __init__(self, lhs='x', val='c'):
        super().__init__()
        assert isinstance(lhs, str) | isinstance(lhs, Expression) | isinstance(lhs, Predicate), "LHS of expression needs to be a string (input name) or Expression"
        assert not isinstance(val, str), "val on the rhs cannot be a string"
        self.lhs = lhs
        self.val = val
        self.subformula = None

    def robustness_trace(self, trace, predicate_scale=1.0, **kwargs):
        """
        Computes robustness trace:  -abs(lhs - rhs)
        Args:
            signal: jnp.array. Expected size [bs, time_dim, state_dim]
            predicate_scale: Optional. scale the robustness by a factor predicate_scale. Default: 1.0

        Returns:
            robustness_trace: jnp.array. Same size as signal.
        """
        if isinstance(self.val, Expression):
            assert self.val.value is not None, "Expression does not have numerical values"
            c_val = self.val.value

        else:
            c_val = self.val

        if isinstance(self.lhs, Predicate):
            return -jnp.abs(c_val - self.lhs(trace)) * predicate_scale
        else:
            return -jnp.abs(c_val - trace) * predicate_scale

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.lhs, self.val]

    def __str__(self):
        lhs_str = self.lhs
        if isinstance(self.lhs, Expression) | isinstance(self.lhs, Predicate):
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
    The Negation STL formula ¬ negates the subformula.

    Args:
        subformula: an STL formula
    """
    def __init__(self, subformula):
        super(Negation, self).__init__()
        self.subformula = subformula

    def robustness_trace(self, signal, **kwargs):
        """
        Computes robustness trace:  -subformula(signal)
        Args:
            signal: jnp.array. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature
        Returns:
            robustness_trace: jnp.array. Same size as signal.
        """
        return -self.subformula(signal, **kwargs)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

    def __str__(self):
        return "¬(" + str(self.subformula) + ")"


class And(STL_Formula):
    """
    The And STL formula ∧ (subformula1 ∧ subformula2)
    Args:
        subformula1: subformula for lhs of the And operation
        subformula2: subformula for rhs of the And operation
    """

    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    @staticmethod
    def separate_and(formula, input_, **kwargs):
        """
        Function of seperate out multiple And operations. e.g., ϕ₁ ∧ ϕ₂ ∧ ϕ₃ ∧ ϕ₄ ∧ ...

        Args:
            formula: STL_formula
            input_: input of STL_formula
        """
        if formula.__class__.__name__ != "And":
            return jnp.expand_dims(formula(input_, **kwargs), -1)
        else:
            if isinstance(input_, tuple):
                return jnp.concatenate([And.separate_and(formula.subformula1, input_[0], **kwargs), And.separate_and(formula.subformula2, input_[1], **kwargs)], axis=-1)
            else:
                return jnp.concatenate([And.separate_and(formula.subformula1, input_, **kwargs), And.separate_and(formula.subformula2, input_, **kwargs)], axis=-1)

    def robustness_trace(self, inputs, **kwargs):
        """
        Computing robustness trace of subformula1 ∧ subformula2  min(subformula1(input1), subformula2(input2))

        Args:
            inputs: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs (e.g, ϕ₁(x) ∧ (ϕ₂(y) ∧ ϕ₃(z)) would have inputs=(x, (y,z))). If using Predicates to define the formula, then inputs is just a single jnp.array. Not need for different signals for each subformula. Expected signal is size [batch_size, time_dim, x_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: jnp.array. Same size as signal.
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
    Args:
        subformula1: subformula for lhs of the Or operation
        subformula2: subformula for rhs of the Or operation
    """
    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    @staticmethod
    def separate_or(formula, input_, **kwargs):
        """
        Function of seperate out multiple Or operations. e.g., ϕ₁ ∨ ϕ₂ ∨ ϕ₃ ∨ ϕ₄ ∨ ...

        Args:
            formula: STL_formula
            input_: input of STL_formula
        """

        if formula.__class__.__name__ != "Or":
            return jnp.expand_dims(formula(input_, **kwargs), -1)
        else:
            if isinstance(input_, tuple):
                return jnp.concatenate([Or.separate_or(formula.subformula1, input_[0], **kwargs), Or.separate_or(formula.subformula2, input_[1], **kwargs)], axis=-1)
            else:
                return jnp.concatenate([Or.separate_or(formula.subformula1, input_, **kwargs), Or.separate_or(formula.subformula2, input_, **kwargs)], axis=-1)

    def robustness_trace(self, inputs, **kwargs):
        """
        Computing robustness trace of subformula1 ∨ subformula2  max(subformula1(input1), subformula2(input2))

        Args:
            inputs: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs (e.g, ϕ₁(x) ∨ (ϕ₂(y) ∨ ϕ₃(z)) would have inputs=(x, (y,z))). If using Predicates to define the formula, then inputs is just a single jnp.array. Not need for different signals for each subformula. Expected signal is size [batch_size, time_dim, x_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: jnp.array. Same size as signal.
        """
        xx = Or.separate_or(self, inputs, **kwargs)
        return maxish(xx, axis=-1, keepdims=False, **kwargs)                                         # [batch_size, time_dim, ...]

    def _next_function(self):
        """ next function is the input subformulas. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ∨ (" + str(self.subformula2) + ")"


class Implies(STL_Formula):
    """
    The Implies STL formula ⇒. subformula1 ⇒ subformula2
    Args:
        subformula1: subformula for lhs of the Implies operation
        subformula2: subformula for rhs of the Implies operation
    """
    def __init__(self, subformula1, subformula2):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2

    def robustness_trace(self, trace, **kwargs):
        """
        Computing robustness trace of subformula1 ⇒ subformula2    max(-subformula1(input1), subformula2(input2))

        Args:
            inputs: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. Each element of the tuple could also be a tuple if the corresponding subformula requires multiple inputs (e.g, ϕ₁(x) ∨ (ϕ₂(y) ∨ ϕ₃(z)) would have inputs=(x, (y,z))). If using Predicates to define the formula, then inputs is just a single jnp.array. Not need for different signals for each subformula. Expected signal is size [batch_size, time_dim, x_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: jnp.array. Same size as signal.
        """
        if isinstance(trace, tuple):
            trace1, trace2 = trace
            signal1 = self.subformula1(trace1, **kwargs)
            signal2 = self.subformula2(trace2, **kwargs)
        else:
            signal1 = self.subformula1(trace, **kwargs)
            signal2 = self.subformula2(trace, **kwargs)
        xx = jnp.stack([-signal1, signal2], axis=-1)      # [batch_size, time_dim, ..., 2]
        return maxish(xx, axis=-1, keepdims=False, **kwargs)   # [batch_size, time_dim, ...]


    def _next_function(self):
        """ next function is the input subformulas. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ⇒ (" + str(self.subformula2) + ")"

class TemporalOperator(STL_Formula):

    def __init__(self, subformula, interval=None):
        super().__init__()
        self.subformula = subformula
        self.interval = interval

        if self.interval is None:
            self.hidden_dim = None
            self._interval = None
        elif interval[1] == jnp.inf:
            self.hidden_dim = None
            self._interval = [interval[0], interval[1]]
        else:
            self.hidden_dim = interval[1] + 1
            self._interval = [interval[0], interval[1]]


        self.LARGE_NUMBER = 1E9
        self.operation = None

    def _get_interval_indices(self):
        start_idx = -self.hidden_dim
        end_idx = -self._interval[0]

        return start_idx, (None if end_idx  == 0 else end_idx)

    def _run_cell(self, signal, padding=None, **kwargs):

        hidden_state = self._initialize_hidden_state(signal, padding=padding) # [hidden_dim]
        def f_(hidden, state):
            hidden, o = self._cell(state, hidden, **kwargs)
            return hidden, o

        _, outputs_stack = jax.lax.scan(f_, hidden_state, signal)
        return outputs_stack

    def _initialize_hidden_state(self, signal, padding=None):
        if padding == "last":
            pad_value = jax.lax.stop_gradient(signal)[0]
        elif padding == "mean":
            pad_value = jax.lax.stop_gradient(signal).mean(0)
        else:
            pad_value = -self.LARGE_NUMBER

        n_time_steps = signal.shape[0]

        # compute hidden dim if signal length was needed

        if (self.interval is None) or (self.interval[1] == jnp.inf):
            self.hidden_dim = n_time_steps
        if self.interval is None:
            self._interval = [0, n_time_steps - 1]
        elif self.interval[1] == jnp.inf:
            self._interval[1] = n_time_steps - 1

        self.M = jnp.diag(jnp.ones(self.hidden_dim-1), k=1)
        self.b = jnp.zeros(self.hidden_dim)
        self.b = self.b.at[-1].set(1)

        if (self.interval is None) or (self.interval[1] == jnp.inf):
            pad_value = jnp.concatenate([jnp.ones(self._interval[0] + 1) * pad_value, jnp.ones(self.hidden_dim - self._interval[0] - 1) * self.sign * pad_value])

        h0 = jnp.ones(self.hidden_dim) * pad_value

        return h0

    def _cell(self, state, hidden, **kwargs):

        h_new = self.M @ hidden + self.b * state
        start_idx, end_idx = self._get_interval_indices()
        output = self.operation(h_new[start_idx:end_idx], axis=0, keepdims=False, **kwargs)

        return h_new, output


    def robustness_trace(self, signal, padding=None, **kwargs):

        trace = self.subformula(signal, **kwargs)
        outputs = self._run_cell(trace, padding, **kwargs)
        return outputs

    def robustness(self, signal, **kwargs):
        return self.__call__(signal, **kwargs)[-1]


    def _next_function(self):
        return [self.subformula]

class AlwaysRecurrent(TemporalOperator):

    def __init__(self, subformula, interval=None):
        super().__init__(subformula=subformula, interval=interval)
        self.operation = minish
        self.sign = -1.

    def __str__(self):
        return "◻ " + str(self._interval) + "( " + str(self.subformula) + " )"

class EventuallyRecurrent(TemporalOperator):

    def __init__(self, subformula, interval=None):
        super().__init__(subformula=subformula, interval=interval)
        self.operation = maxish
        self.sign = 1.

    def __str__(self):
        return "♢ " + str(self._interval) + "( " + str(self.subformula) + " )"


class UntilRecurrent(STL_Formula):

    def __init__(self, subformula1, subformula2, interval=None, overlap=True):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.interval = interval
        if overlap == False:
            self.subformula2 = Eventually(subformula=subformula2, interval=[0,1])
        self.LARGE_NUMBER = 1E9
        # self.Alw = AlwaysRecurrent(subformula=Identity(name=str(self.subformula1))
        self.Alw = AlwaysRecurrent(Predicate('x', lambda x: x) > 0.)

        if self.interval is None:
            self.hidden_dim = None
        elif interval[1] == jnp.inf:
            self.hidden_dim = None
        else:
            self.hidden_dim = interval[1] + 1


    def _initialize_hidden_state(self, signal, padding=None, **kwargs):
        time_dim = 0  # assuming signal is [time_dim,...]

        if isinstance(signal, tuple):
            # for formula defined using Expression
            assert signal[0].shape[time_dim] == signal[1].shape[time_dim]
            trace1 = self.subformula1(signal[0], **kwargs)
            trace2 = self.subformula2(signal[1], **kwargs)
            n_time_steps = signal[0].shape[time_dim]
        else:
            # for formula defined using Predicate
            trace1 = self.subformula1(signal, **kwargs)
            trace2 = self.subformula2(signal, **kwargs)
            n_time_steps = signal.shape[time_dim]

        # compute hidden dim if signal length was needed
        if self.hidden_dim is None:
            self.hidden_dim = n_time_steps
        if self.interval is None:
            self.interval = [0, n_time_steps - 1]
        elif self.interval[1] == jnp.inf:
            self.interval[1] = n_time_steps - 1

        self.ones_array = jnp.ones(self.hidden_dim)

        # set shift operations given hidden_dim
        self.M = jnp.diag(jnp.ones(self.hidden_dim-1), k=1)
        self.b = jnp.zeros(self.hidden_dim)
        self.b = self.b.at[-1].set(1)

        if self.hidden_dim == n_time_steps:
            pad_value = self.LARGE_NUMBER
        else:
            pad_value = -self.LARGE_NUMBER

        h1 = pad_value * self.ones_array
        h2 = -self.LARGE_NUMBER * self.ones_array
        return (h1, h2), trace1, trace2

    def _get_interval_indices(self):
        start_idx = -self.hidden_dim
        end_idx = -self.interval[0]

        return start_idx, (None if end_idx  == 0 else end_idx)

    def _cell(self, state, hidden, **kwargs):
        x1, x2 = state
        h1, h2 = hidden
        h1_new = self.M @ h1 + self.b * x1
        h1_min = jnp.flip(self.Alw(jnp.flip(h1_new), **kwargs))
        h2_new = self.M @ h2 + self.b * x2
        start_idx, end_idx = self._get_interval_indices()
        z = minish(jnp.stack([h1_min, h2_new]), axis=0, keepdims=False, **kwargs)[start_idx:end_idx]

        def g_(carry, x):
            carry = maxish(jnp.array([carry, x]), axis=0, keepdims=False, **kwargs)
            return carry, carry

        output, _ = jax.lax.scan(g_,  -self.LARGE_NUMBER, z)

        return output, (h1_new, h2_new)

    def robustness_trace(self, signal, padding=None, **kwargs):
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
        hidden_state, trace1, trace2 = self._initialize_hidden_state(signal, padding=padding, **kwargs)

        def f_(hidden, state):
            o, hidden = self._cell(state, hidden, **kwargs)
            return hidden, o

        _, outputs_stack = jax.lax.scan(f_, hidden_state, jnp.stack([trace1, trace2], axis=1))
        return outputs_stack


    def robustness(self, signal, **kwargs):
        """
        Computes the robustness value. Extracts the last entry along time_dim of robustness trace.

        Args:
            signal: jnp.array or Expression. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return: jnp.array, same as input with the time_dim removed.
        """
        return self.__call__(signal, **kwargs)[-1]
        # return jnp.rollaxis(self.__call__(signal, **kwargs), time_dim)[-1]
    def _next_function(self):
        """ next function is the input subformulas. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return  "(" + str(self.subformula1) + ")" + " U " + "(" + str(self.subformula2) + ")"



class Expression:
    name: str
    value: jnp.array

    def __init__(self, name, value):
        self.value = value
        self.name = name

    def set_name(self, new_name):
        self.name = new_name

    def set_value(self, new_value):
        self.value = new_value

    def __neg__(self):
        return Expression('-' + self.name, -self.value)

    def __add__(self, other):
        if isinstance(other, Expression):
            return Expression(self.name + ' + ' + other.name, self.value + other.value)
        else:
            return Expression(self.name + ' + other', self.value + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Expression):
            return Expression(self.name + ' - ' + other.name, self.value - other.value)
        else:
            return Expression(self.name + " - other", self.value - other)

    def __rsub__(self, other):
        return self.__sub__(other)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular sub

    def __mul__(self, other):
        if isinstance(other, Expression):
            return Expression(self.name + ' × ' + other.name, self.value * other.value)
        else:
            return Expression(self.name + " * other", self.value * other)


    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(a, b):
        # This is the new form required by Python 3
        numerator = a
        denominator = b
        return Expression(numerator.name + '/' + denominator.name, numerator.value/denominator.value)


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

    def __str__(self):
        return str(self.name)

    def __call__(self):
        return self.value

class Predicate:
    name: str
    predicate_function: Callable

    def __init__(self, name, predicate_function=lambda x: x):
        self.name = name
        self.predicate_function = predicate_function

    def set_name(self, new_name):
        self.name = new_name

    def __neg__(self):
        return Predicate('- ' + self.name, lambda x: -self.predicate_function(x))

    def __add__(self, other):
        if isinstance(other, Predicate):
            return Predicate(self.name + ' + ' + other.name, lambda x: self.predicate_function(x) + other.predicate_function(x))
        else:
            raise ValueError("Type error. Must be Predicate")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Predicate):
            return Predicate(self.name + ' - ' + other.name, lambda x: self.predicate_function(x) - other.predicate_function(x))
        else:
            raise ValueError("Type error. Must be Predicate")

    def __rsub__(self, other):
        return self.__sub__(other)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular sub

    def __mul__(self, other):
        if isinstance(other, Predicate):
            return Predicate(self.name + ' x ' + other.name, lambda x: self.predicate_function(x) * other.predicate_function(x))
        else:
            raise ValueError("Type error. Must be Predicate")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(a, b):
        if isinstance(a, Predicate) and isinstance(b, Predicate):
            return Predicate(a.name + ' / ' + b.name, lambda x: a.predicate_function(x) / b.predicate_function(x))
        else:
            raise ValueError("Type error. Must be Predicate")

    # Comparators
    def __lt__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Predicate), "LHS of LessThan needs to be a string or Predicate"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs)

    def __le__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Predicate), "LHS of LessThan needs to be a string or Predicate"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return LessThan(lhs, rhs)

    def __gt__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Predicate), "LHS of GreaterThan needs to be a string or Predicate"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)

    def __ge__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Predicate), "LHS of GreaterThan needs to be a string or Predicate"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return GreaterThan(lhs, rhs)

    def __eq__(lhs, rhs):
        assert isinstance(lhs, str) | isinstance(lhs, Predicate), "LHS of Equal needs to be a string or Predicate"
        assert not isinstance(rhs, str), "RHS cannot be a string"
        return Equal(lhs, rhs)

    def __str__(self):
        return str(self.name)

    def __call__(self, signal, **kwargs):
        return self.predicate_function(signal)

def convert_to_input_values(inputs):
    '''Converts input into jnp.arrays'''
    if not isinstance(inputs, tuple):
        if isinstance(inputs, Expression):
            assert inputs.value is not None, "Input Expression does not have numerical values"
            # if Expression is not time reversed
            return inputs.value
        elif isinstance(inputs, jax.Array):
            return inputs
        else:
            raise ValueError("Not a invalid input trace")
    else:
        return (convert_to_input_values(inputs[0]), convert_to_input_values(inputs[1]))


class Eventually(STL_Formula):
    def __init__(self, subformula, interval=None):
        super().__init__()

        self.interval = interval
        self.subformula = subformula
        self._interval = [0, jnp.inf] if self.interval is None else self.interval

    def robustness_trace(self, signal, padding=None, large_number=1E9, **kwargs):
        time_dim = 0  # assuming signal is [time_dim,...]
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = -large_number
        offset = 0
        if self.interval is None:
            interval = [0,T-1]
        elif self.interval[1] == jnp.inf:
            interval = [self.interval[0], T-1]
            offset = self.interval[0]
        else:
            interval = self.interval
        signal_matrix = signal.reshape([T,1]) @ jnp.ones([1,T])
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = mask_value
        signal_pad = jnp.ones([interval[1]+1, T]) * pad_value
        signal_padded = jnp.concatenate([signal_matrix, signal_pad], axis=time_dim)
        subsignal_mask = jnp.tril(jnp.ones([T + interval[1]+1,T]))
        time_interval_mask = jnp.triu(jnp.ones([T + interval[1]+1,T]), -interval[-1]-offset) * jnp.tril(jnp.ones([T + interval[1]+1,T]), -interval[0])
        masked_signal_matrix = jnp.where(time_interval_mask * subsignal_mask, signal_padded, mask_value)
        return maxish(masked_signal_matrix, axis=time_dim, keepdims=False, **kwargs)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

    def __str__(self):
        return "♢ " + str(self._interval) + "( " + str(self.subformula) + " )"


class Always(STL_Formula):
    def __init__(self, subformula, interval=None):
        super().__init__()

        self.interval = interval
        self.subformula = subformula
        self._interval = [0, jnp.inf] if self.interval is None else self.interval

    def robustness_trace(self, signal, padding=None, large_number=1E9, **kwargs):
        time_dim = 0  # assuming signal is [time_dim,...]
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = large_number
        sign = 1.
        offset = 0
        # if self.interval is None:
        #     interval = [0,T-1]
        #     sign = -1.
        def true_func(_interval, T):
            return [_interval[0], T-1], -1., _interval[0]
        def false_func(_interval, T):
            return _interval, 1., 0
        operands = (self._interval, T,)
        interval, sign, offset = cond(self._interval[1] == jnp.inf, true_func, false_func, *operands)
        # if self._interval[1] == jnp.inf:
        #     interval = [self.interval[0], T-1]
        #     sign = -1.
        #     offset = self.interval[0]
        # else:
        #     interval = self.interval
        signal_matrix = signal.reshape([T,1]) @ jnp.ones([1,T])
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = mask_value
        signal_pad = jnp.concatenate([jnp.ones([interval[1], T]) * sign * pad_value, jnp.ones([1, T]) * pad_value], axis=time_dim)
        signal_padded = jnp.concatenate([signal_matrix, signal_pad], axis=time_dim)
        subsignal_mask = jnp.tril(jnp.ones([T + interval[1]+1,T]))

        time_interval_mask = jnp.triu(jnp.ones([T + interval[1]+1,T]), -interval[-1]-offset) * jnp.tril(jnp.ones([T + interval[1]+1,T]), -interval[0])
        masked_signal_matrix = jnp.where(time_interval_mask * subsignal_mask, signal_padded, mask_value)
        return minish(masked_signal_matrix, axis=time_dim, keepdims=False, **kwargs)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

    def __str__(self):
        return "◻ " + str(self._interval) + "( " + str(self.subformula) + " )"


class Until(STL_Formula):
    def __init__(self, subformula1, subformula2, interval=None):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.interval = interval
        self._interval = [0, jnp.inf] if self.interval is None else self.interval


    def robustness_trace(self, signal, padding=None, large_number=1E9, **kwargs):
        time_dim = 0  # assuming signal is [time_dim,...]
        if isinstance(signal, tuple):
            signal1, signal2 = signal
            assert signal1.shape[time_dim] == signal2.shape[time_dim]
            signal1 = self.subformula1(signal1, padding=padding, large_number=large_number, **kwargs)
            signal2 = self.subformula2(signal2, padding=padding, large_number=large_number, **kwargs)
            T = signal1.shape[time_dim]
        else:
            signal1 = self.subformula1(signal, padding=padding, large_number=large_number, **kwargs)
            signal2 = self.subformula2(signal, padding=padding, large_number=large_number, **kwargs)
            T = signal.shape[time_dim]

        mask_value = large_number
        if self.interval is None:
            interval = [0,T-1]
        elif self.interval[1] == jnp.inf:
            interval = [self.interval[0], T-1]
        else:
            interval = self.interval
        signal1_matrix = signal1.reshape([T,1]) @ jnp.ones([1,T])
        signal2_matrix = signal2.reshape([T,1]) @ jnp.ones([1,T])
        if padding == "last":
            signal1_pad = jnp.ones([interval[1]+1, T]) * signal1[-1]
            signal2_pad = jnp.ones([interval[1]+1, T]) * signal2[-1]
        elif padding == "mean":
            signal1_pad = jnp.ones([interval[1]+1, T]) * signal1.mean(time_dim)
            signal2_pad = jnp.ones([interval[1]+1, T]) * signal2.mean(time_dim)
        else:
            signal1_pad = jnp.ones([interval[1]+1, T]) * -mask_value
            signal2_pad = jnp.ones([interval[1]+1, T]) * -mask_value

        signal1_padded = jnp.concatenate([signal1_matrix, signal1_pad], axis=time_dim)
        signal2_padded = jnp.concatenate([signal2_matrix, signal2_pad], axis=time_dim)

        start_idx = interval[0]
        phi1_mask = jnp.stack([jnp.triu(jnp.ones([T + interval[1]+1,T]), -end_idx) * jnp.tril(jnp.ones([T + interval[1]+1,T])) for end_idx in range(interval[0], interval[-1]+1)], 0)
        phi2_mask = jnp.stack([jnp.triu(jnp.ones([T + interval[1]+1,T]), -end_idx) * jnp.tril(jnp.ones([T + interval[1]+1,T]), -end_idx) for end_idx in range(interval[0], interval[-1]+1)], 0)
        phi1_masked_signal = jnp.stack([jnp.where(m1, signal1_padded, mask_value) for m1 in phi1_mask], 0)
        phi2_masked_signal = jnp.stack([jnp.where(m2, signal2_padded, mask_value) for m2 in phi2_mask], 0)
        return maxish(jnp.stack([minish(jnp.stack([minish(s1, axis=0, keepdims=False, **kwargs), minish(s2, axis=0, keepdims=False, **kwargs)], axis=0), axis=0, keepdims=False, **kwargs) for (s1, s2) in zip(phi1_masked_signal, phi2_masked_signal)], axis=0), axis=0, keepdims=False, **kwargs)



    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return  "(" + str(self.subformula1) + ")" + " U " + str(self._interval) + "(" + str(self.subformula2) + ")"


class DifferentiableAlways(STL_Formula):
    def __init__(self, subformula, interval=None):
        super().__init__()

        self.interval = interval
        self.subformula = subformula
        # self._interval = [0, jnp.inf] if self.interval is None else self.interval

    def robustness_trace(self, signal, t_start, t_end, scale=1.0, padding=None, large_number=1E9, delta=1E-3, **kwargs):
        time_dim = 0  # assuming signal is [time_dim,...]
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = large_number
        signal_matrix = signal.reshape([T,1]) @ jnp.ones([1,T])
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = -mask_value
        signal_pad = jnp.ones([T, T]) * pad_value
        signal_padded = jnp.concatenate([signal_matrix, signal_pad], axis=time_dim)
        smooth_time_mask = smooth_mask(T, t_start, t_end, scale)
        padded_smooth_time_mask = jnp.zeros([2 * T, T])
        for t in range(T):
            padded_smooth_time_mask = padded_smooth_time_mask.at[t:t+T,t].set(smooth_time_mask)

        masked_signal_matrix = jnp.where(padded_smooth_time_mask > delta, signal_padded * padded_smooth_time_mask, mask_value)
        return minish(masked_signal_matrix, axis=time_dim, keepdims=False, **kwargs)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

    def __str__(self):
        return "◻ [a,b] ( " + str(self.subformula) + " )"


class DifferentiableEventually(STL_Formula):
    def __init__(self, subformula, interval=None):
        super().__init__()

        self.interval = interval
        self.subformula = subformula
        self._interval = [0, jnp.inf] if self.interval is None else self.interval

    def robustness_trace(self, signal, t_start, t_end, scale=1.0, padding=None, large_number=1E9, delta=1E-3, **kwargs):
        time_dim = 0  # assuming signal is [time_dim,...]
        signal = self.subformula(signal, padding=padding, large_number=large_number, **kwargs)
        T = signal.shape[time_dim]
        mask_value = -large_number
        signal_matrix = signal.reshape([T,1]) @ jnp.ones([1,T])
        if padding == "last":
            pad_value = signal[-1]
        elif padding == "mean":
            pad_value = signal.mean(time_dim)
        else:
            pad_value = mask_value
        signal_pad = jnp.ones([T, T]) * pad_value
        signal_padded = jnp.concatenate([signal_matrix, signal_pad], axis=time_dim)
        smooth_time_mask = smooth_mask(T, t_start, t_end, scale)
        padded_smooth_time_mask = jnp.zeros([2 * T, T])
        for t in range(T):
            padded_smooth_time_mask = padded_smooth_time_mask.at[t:t+T,t].set(smooth_time_mask)

        masked_signal_matrix = jnp.where(padded_smooth_time_mask > delta, signal_padded * padded_smooth_time_mask, mask_value)
        return maxish(masked_signal_matrix, axis=time_dim, keepdims=False, **kwargs)

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

    def __str__(self):
        return "♢ [a,b] ( " + str(self.subformula) + " )"