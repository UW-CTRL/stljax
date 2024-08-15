import jax
import jax.numpy as jnp
import numpy as np
from stljax.formula import *
from stljax.viz import *
import matplotlib.pyplot as plt
import timeit
import statistics
import pickle

if __name__ == "__main__":
    axis = 0
    pred_rev = Predicate('x', lambda x: x, axis, True)
    interval = [2, 5]
    mask = Always(pred_rev > 4, interval=interval)
    recurrent = AlwaysRecurrent(pred_rev > 4, interval=interval)



    def grad_mask(signal):
        return jax.vmap(jax.grad(lambda x: mask(x).mean()))(signal)
    def grad_recurrent(signal):
        return jax.vmap(jax.grad(lambda x: recurrent(x).mean()))(signal)

    def mask_(signal):
        return jax.vmap(lambda x: mask(x).mean())(signal)
    def recurrent_(signal):
        return jax.vmap(lambda x: recurrent(x).mean())(signal)


    @jax.jit
    def grad_mask_jit(signal):
        return jax.vmap(jax.grad(lambda x: mask(x).mean()))(signal)
    @jax.jit
    def grad_recurrent_jit(signal):
        return jax.vmap(jax.grad(lambda x: recurrent(x).mean()))(signal)
    @jax.jit
    def mask_jit(signal):
        return jax.vmap(lambda x: mask(x).mean())(signal)
    @jax.jit
    def recurrent_jit(signal):
        return jax.vmap(lambda x: recurrent(x).mean())(signal)

    # Number of loops per run
    loops = 100
    # Number of runs
    runs = 25
    T = 2

    bs = 256
    means = []
    stds = []
    data = {}

    # functions = ["mask_", "recurrent_", "grad_mask", "grad_recurrent", "mask_jit", "recurrent_jit", "grad_mask_jit", "grad_recurrent_jit"]
    functions = ["mask_jit", "recurrent_jit", "grad_mask_jit", "grad_recurrent_jit"]

    while T <= 1024:
        print("running ", T)
        signal = jnp.array(np.random.random([bs, T]))
        times = []
        data[str(T)] = {}
        for f in functions:
            print("timing ", f)
            times.append(timeit.repeat(f + "(signal)", globals=globals(), repeat=runs, number=loops))
            print("timing: ", statistics.mean(times[-1]), statistics.stdev(times[-1]))
            data[str(T)][f] = times
            with open('signal_length.pkl', 'wb') as f:
                pickle.dump(data, f)

        T *= 2