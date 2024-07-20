import jax
import jax.numpy as jnp
import warnings
from typing import Callable

warnings.simplefilter("default")


@jax.jit
def bar_plus(signal, p=2):
    return jax.nn.relu(signal) ** p


@jax.jit
def bar_minus(signal, p=2):
    return (-jax.nn.relu(-signal)) ** p


def M0(signal, eps, weights=None, axis=1, keepdims=True):
    if weights is None:
        weights = jnp.ones_like(signal)
    sum_w = weights.sum(axis, keepdims=keepdims)
    return (
        eps**sum_w + jnp.prod(signal**weights, axis=axis, keepdims=keepdims)
    ) ** (1 / sum_w)


def Mp(signal, eps, p, weights=None, axis=1, keepdims=True):
    if weights is None:
        weights = jnp.ones_like(signal)
    sum_w = weights.sum(axis, keepdims=keepdims)
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


def gmsr_min_turbo(signal, eps, p, weights=None, axis=1, keepdims=True):
    # TODO: (norrisg) make actually turbo (faster than normal `gmsr_min`)
    pos_idx = signal > 0.0
    neg_idx = ~pos_idx

    return jnp.where(
        neg_idx.sum(axis, keepdims=keepdims) > 0,
        eps**0.5
        - Mp(
            bar_minus(signal, 2),
            eps,
            p,
            weights=weights,
            axis=axis,
            keepdims=keepdims,
        )
        ** 0.5,
        M0(bar_plus(signal, 2), eps, weights=weights, axis=axis, keepdims=keepdims)
        ** 0.5
        - eps**0.5,
    )


def gmsr_max_turbo(signal, eps, p, weights=None, axis=1, keepdims=True):
    return -gmsr_min_turbo(
        -signal, eps, p, weights=weights, axis=axis, keepdims=keepdims
    )


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


def maxish(signal, axis, keepdims=True, approx_method="true", temperature=None, **kwargs):
    """
    Function to compute max(ish) along an axis.

    Args:
        signal: A jnp.array or an Expression
        axis: (Int) axis along to compute max(ish)
        keepdims: (Bool) whether to keep the original array size. Defaults to True
        approx_method: (String) argument to choose the type of max(ish) approximation. possible choices are "true", "logsumexp", "softmax", "gmsr" (https://arxiv.org/abs/2405.10996).
        temperature: Optional, required for approx_method not True.

    Returns:
        jnp.array corresponding to the maxish

    Raises:
        If Expression does not have a value, or invalid approx_method

    """

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


def minish(signal, axis, keepdims=True, approx_method="true", temperature=None, **kwargs):
    '''
    Same as maxish
    '''
    return -maxish(-signal, axis, keepdims, approx_method, temperature, **kwargs)


class STL_Formula:
    '''
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

    def robustness(self, signal, time_dim, **kwargs):
        """
        Computes the robustness value. Extracts the last entry along time_dim of robustness trace.

        Args:
            signal: jnp.array or Expression. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return: jnp.array, same as input with the time_dim removed.
        """

        kwargs["time_dim"] = time_dim
        return jnp.rollaxis(self.__call__(signal, **kwargs), time_dim)[-1]

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

    def eval(self, signal, time_dim, **kwargs):
        """
        Boolean of robustness

        Args:
            signal: jnp.array or Expression. Expected size [bs, time_dim, state_dim]
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return: jnp.array with True/False, same as input with the time_dim removed.
        """
        return self.robustness(signal, time_dim, **kwargs) > 0


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
            if not self.lhs.reverse:
                warnings.warn("Input Predicate \"{input_name}\" is not time reversed. Reversing the signal now...".format(input_name=self.lhs.name))
                return (c_val - jnp.flip(self.lhs(signal), self.lhs.time_dim)) * predicate_scale
            else:
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
            if not self.lhs.reverse:
                warnings.warn("Input Predicate \"{input_name}\" is not time reversed. Reversing the signal now...".format(input_name=self.lhs.name))
                return -(c_val - jnp.flip(self.lhs(trace), self.lhs.time_dim)) * predicate_scale
            else:
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
            if not self.lhs.reverse:
                warnings.warn("Input Predicate \"{input_name}\" is not time reversed. Reversing the signal now...".format(input_name=self.lhs.name))
                return -jnp.abs(c_val - jnp.flip(self.lhs(trace), self.lhs.time_dim)) * predicate_scale
            else:
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
        xx = Or.separate_and(self, inputs, **kwargs)
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
        trace1, trace2 = trace
        signal1 = self.subformula1(trace1, **kwargs)
        signal2 = self.subformula2(trace2, **kwargs)
        xx = jnp.stack([-signal1, signal2], axis=-1)      # [batch_size, time_dim, ..., 2]
        return maxish(xx, axis=-1, keepdims=False, **kwargs)   # [batch_size, time_dim, ...]

    def _next_function(self):
        """ next function is the input subformulas. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return "(" + str(self.subformula1) + ") ⇒ (" + str(self.subformula2) + ")"

class Temporal_Operator(STL_Formula):
    """
    Class to compute Eventually and Always. This builds a recurrent cell to perform dynamic programming

    Args:
        subformula: The subformula that the temporal operator is applied to.
        interval: The time interval that the temporal operator operates on. Default: None which means [0, jnp.inf]. Other options car be: [a, b] (b < jnp.inf), [a, jnp.inf] (a > 0)

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

        Args:
            signal: the input signal. Expected size [batch_size, time_dim, signal_dim]

        Returns:
            h0: initial hidden state is [batch_size, hidden_dim, signal_dim]

        Notes:
        Initializing the hidden state requires padding on the signal. Currently, the default is to extend the last value.
        TODO: have option on this padding

        """
        # Case 1, 2, 4
        # TODO: make this less hard-coded. Assumes signal is [bs, time_dim, signal_dim], and already reversed
        # pads with the signal value at the last time step.
        h0 = jnp.ones([signal.shape[0], self.hidden_dim, signal.shape[2]])*signal[:,:1,:]

        # Case 3: if self.interval is [a, jnp.inf), then the hidden state is a tuple (like in an LSTM)
        if (self._interval[1] == jnp.inf) & (self._interval[0] > 0):
            c0 = signal[:,:1,:]
            return (c0, h0)
        return h0

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


    def _run_cell(self, signal, time_dim=1, **kwargs):
        """
        Function to run a signal through a cell T times, where T is the length of the signal in the time dimension

        Args:
            signal: input signal, size = [bs, time_dim, ...]
            time_dim: axis corresponding to time_dim. Default: 1
            kwargs: Other arguments including time_dim, approx_method, temperature

        Return:
            outputs: list of outputs
            states: list of hidden_states
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


    def robustness_trace(self, signal, time_dim=1, **kwargs):
        """
        Function to compute robustness trace of a temporal STL formula
        First, compute the robustness trace of the subformula, and use that as the input for the recurrent computation

        Args:
            signal: input signal, size = [bs, time_dim, ...]
            time_dim: axis corresponding to time_dim. Default: 1
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: jnp.array. Same size as signal.
        """

        trace = self.subformula(signal, **kwargs)
        outputs, _ = self._run_cell(trace, time_dim, **kwargs)
        return jnp.concatenate(outputs, axis=time_dim)                              # [batch_size, time_dim, ...]

    def _next_function(self):
        """ next function is the input subformula. For visualization purposes """
        return [self.subformula]

class Always(Temporal_Operator):
    """
    The Always STL formula □_[a,b] subformula
    The robustness value is the minimum value of the input trace over a prespecified time interval

    Args:
        subformula: subformula that the Always operation is applied on
        interval: time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep size is.
    """
    def __init__(self, subformula, interval=None):
        super().__init__(subformula=subformula, interval=interval)

    def _cell(self, x, hidden_state, time_dim, **kwargs):
        """
        see Temporal_Operator._cell
        """
        # Case 1, interval = [0, inf]
        if self.interval is None:
            input_ = jnp.concatenate([hidden_state, x], axis=time_dim)                # [batch_size, rnn_dim+1, x_dim]
            output = minish(input_, time_dim, keepdims=True, **kwargs)       # [batch_size, 1, x_dim]
            return output, output

        # Case 3: self.interval is [a, np.inf)
        if (self._interval[1] == jnp.inf) & (self._interval[0] > 0):
            c, h = hidden_state
            ch = jnp.concatenate([c, h[:,:1,:]], axis=time_dim)                             # [batch_size, 2, x_dim]
            output = minish(ch, time_dim, keepdims=True, **kwargs)               # [batch_size, 1, x_dim]
            hidden_state_ = (output, self.M @ h + self.b * x)

        # Case 2 and 4: self.interval is [a, b]
        else:
            hidden_state_ = self.M @ hidden_state + self.b * x
            hx = jnp.concatenate([hidden_state, x], axis=time_dim)                             # [batch_size, rnn_dim+1, x_dim]
            input_ = hx[:,:self.steps,:]                               # [batch_size, self.steps, x_dim]
            output = minish(input_, time_dim, **kwargs)               # [batch_size, 1, x_dim]
        return output, hidden_state_

    def __str__(self):
        return "◻ " + str(self._interval) + "( " + str(self.subformula) + " )"


class Eventually(Temporal_Operator):
    """
    The Eventually STL formula ♢_[a,b] subformula
    The robustness value is the minimum value of the input trace over a prespecified time interval

    Args:
        subformula: subformula that the Eventually operation is applied on
        interval: time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep size is.
    """
    def __init__(self, subformula, interval=None):
        super().__init__(subformula=subformula, interval=interval)

    def _cell(self, x, hidden_state, time_dim, **kwargs):
        """
        see Temporal_Operator._cell
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
            hidden_state_ = (output, self.M @ h + self.b * x)

        # Case 2 and 4: self.interval is [a, b]
        else:
            hidden_state_ = self.M @ hidden_state + self.b * x
            hx = jnp.concatenate([hidden_state, x], axis=time_dim)                             # [batch_size, rnn_dim+1, x_dim]
            input_ = hx[:,:self.steps,:]                               # [batch_size, self.steps, x_dim]
            output = maxish(input_, time_dim, **kwargs)               # [batch_size, 1, x_dim]
        return output, hidden_state_

    def __str__(self):
        return "♢ " + str(self._interval) + "( " + str(self.subformula) + " )"


class Until(STL_Formula):
    """
    The Until STL operator U. Subformula1 U_[a,b] subformula2
    Arg:
        subformula1: subformula for lhs of the Until operation
        subformula2: subformula for rhs of the Until operation
        interval: time interval [a,b] where a, b are indices along the time dimension. It is up to the user to keep track of what the timestep is.
        overlap: If overlap=True, then the last time step that ϕ is true, ψ starts being true. That is, sₜ ⊧ ϕ and sₜ ⊧ ψ at a common time t. If overlap=False, when ϕ stops being true, ψ starts being true. That is sₜ ⊧ ϕ and sₜ+₁ ⊧ ψ, but sₜ ¬⊧ ψ
    """

    def __init__(self, subformula1, subformula2, interval=None, overlap=True):
        super().__init__()
        self.subformula1 = subformula1
        self.subformula2 = subformula2
        self.interval = interval
        if overlap == False:
            self.subformula2 = Eventually(subformula=subformula2, interval=[0,1])
        self.LARGE_NUMBER = 1E6

    def robustness_trace(self, signal, time_dim=1, **kwargs):
        """
        Computing robustness trace of subformula1 U subformula2  (see paper)

        Args:
            signal: input signal for the formula. If using Expressions to define the formula, then inputs a tuple of signals corresponding to each subformula. If using Predicates to define the formula, then inputs is just a single jnp.array. Not need for different signals for each subformula. Expected signal is size [batch_size, time_dim, x_dim]
            time_dim: axis for time_dim. Default: 1
            kwargs: Other arguments including time_dim, approx_method, temperature

        Returns:
            robustness_trace: jnp.array. Same size as signal.
        """


        # TODO (karenl7) this really assumes axis=1 is the time dimension. Can this be generalized?
        assert time_dim == 1, "Current implementation assumes time_dim=1"
        LARGE_NUMBER = self.LARGE_NUMBER
        assert signal[0].shape[time_dim] == signal[1].shape[time_dim]
        n_time_steps = signal[0].shape[time_dim] # TODO: WIP
        trace1 = self.subformula1(signal[0], **kwargs)
        trace2 = self.subformula2(signal[1], **kwargs)
        Alw = Always(subformula=Identity(name=str(self.subformula1)))
        # TODO (karenl7) this really assumes axis=1 is the time dimension. Can this be generalized?
        LHS = jnp.permute_dims(jnp.repeat(jnp.expand_dims(trace2, -1), n_time_steps, axis=-1), [0,3,2,1])  # [bs, sub_signal, state, t_prime]
        RHS = jnp.ones_like(LHS) * -LARGE_NUMBER  # [bs, sub_signal, state, t_prime]

        # Case 1, interval = [0, inf]
        if self.interval == None:
            for i in range(n_time_steps):
                # TODO (karenl7) this really assumes axis=1 is the time dimension. Can this be generalized?
                RHS = RHS.at[:,i:,:,i].set(Alw(trace1[:,i:,:]))

        # Case 2 and 4: self.interval is [a, b], a ≥ 0, b < ∞
        elif self.interval[1] < jnp.inf:
            a = self.interval[0]
            b = self.interval[1]
            for i in range(n_time_steps):
                end = i+b+1
                # TODO (karenl7) this really assumes axis=1 is the time dimension. Can this be generalized?
                RHS = RHS.at[:,i+a:end,:,i].set(Alw(trace1[:,i:end,:])[:,a:,:])

        # Case 3: self.interval is [a, np.inf), a ≂̸ 0
        else:
            a = self.interval[0]
            for i in range(n_time_steps):
                # TODO (karenl7) this really assumes axis=1 is the time dimension. Can this be generalized?
                RHS = RHS.at[:,i+a:,:,i].set(Alw(trace1[:,i:,:])[:,a:,:])

        # TODO (karenl7) this really assumes axis=1 is the time dimension. Can this be generalized?
        return maxish(minish(jnp.stack([LHS, RHS], axis=-1), axis=-1, keepdims=False, **kwargs), axis=-1, keepdims=False, **kwargs)


    def _next_function(self):
        """ next function is the input subformulas. For visualization purposes """
        return [self.subformula1, self.subformula2]

    def __str__(self):
        return  "(" + str(self.subformula1) + ")" + " U " + "(" + str(self.subformula2) + ")"




class Expression:
    name: str
    value: jnp.array
    reverse: bool
    time_dim: int

    def __init__(self, name, value, reverse, time_dim=1):
        self.value = value
        self.name = name
        self.reverse = reverse
        self.time_dim = time_dim

    def set_name(self, new_name):
        self.name = new_name

    def set_value(self, new_value):
        self.value = new_value

    def flip(self, dim):
        assert self.value is not None, "Expression does not have numerical values"
        self.value = jnp.flip(self.value, dim)
        self.reverse = not self.reverse
        return self.value

    def flip_time(self):
        assert self.value is not None, "Expression does not have numerical values"
        self.value = jnp.flip(self.value, self.time_dim)
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

    def __str__(self):
        return str(self.name)

    def __call__(self):
        return self.value

class Predicate:
    name: str
    predicate_function: Callable
    reverse: bool
    time_dim: int

    def __init__(self, name, predicate_function=lambda x: x, time_dim=1, reverse=False):
        self.name = name
        self.predicate_function = predicate_function
        self.reverse = reverse
        self.time_dim = time_dim

    def set_name(self, new_name):
        self.name = new_name

    def __neg__(self):
        return Predicate('- ' + self.name, lambda x: -self.predicate_function(x), self.reverse)

    def __add__(self, other):
        if isinstance(other, Predicate):
            if self.reverse == other.reverse:
                return Predicate(self.name + ' + ' + other.name, lambda x: self.predicate_function(x) + other.predicate_function(x), self.reverse)
            else:
                raise ValueError("reverse attribute do not match")
        else:
            raise ValueError("Type error. Must be Predicate")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Predicate):
            if self.reverse == other.reverse:
                return Predicate(self.name + ' - ' + other.name, lambda x: self.predicate_function(x) - other.predicate_function(x), self.reverse)
            else:
                raise ValueError("reverse attribute do not match")
        else:
            raise ValueError("Type error. Must be Predicate")

    def __rsub__(self, other):
        return self.__sub__(other)
        # No need for the case when "other" is an Expression, since that
        # case will be handled by the regular sub

    def __mul__(self, other):
        if isinstance(other, Predicate):
            if self.reverse == other.reverse:
                return Predicate(self.name + ' x ' + other.name, lambda x: self.predicate_function(x) * other.predicate_function(x), self.reverse)
            else:
                raise ValueError("reverse attribute do not match")
        else:
            raise ValueError("Type error. Must be Predicate")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(a, b):
        if isinstance(a, Predicate) and isinstance(b, Predicate):
            if a.reverse == b.reverse:
                return Predicate(a.name + ' / ' + b.name, lambda x: a.predicate_function(x) / b.predicate_function(x), a.reverse)
            else:
                raise ValueError("reverse attribute do not match")
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
    if not isinstance(inputs, tuple):
        if isinstance(inputs, Expression):
            assert inputs.value is not None, "Input Expression does not have numerical values"
            # if Expression is not time reversed
            if not inputs.reverse:
                # throw warning to the user
                warnings.warn("Input Expression \"{input_name}\" is not time reversed! stljax will time-reverse the inputs for you...".format(input_name=inputs.name))
                return jnp.flip(inputs.value, inputs.time_dim)
            else:
                return inputs.value
        elif isinstance(inputs, jax.Array):
            return inputs
        else:
            raise ValueError("Not a invalid input trace")
    else:
        return (convert_to_input_values(inputs[0]), convert_to_input_values(inputs[1]))