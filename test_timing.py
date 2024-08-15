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
    phi = Always(pred_rev > 4, interval=interval)
    psi = AlwaysRecurrent(pred_rev > 4, interval=interval)



    def grad_phi(signal):
        return jax.vmap(jax.grad(lambda x: phi(x).mean()))(signal)
    def grad_psi(signal):
        return jax.vmap(jax.grad(lambda x: psi(x).mean()))(signal)

    def phi_(signal):
        return jax.vmap(lambda x: phi(x).mean())(signal)
    def psi_(signal):
        return jax.vmap(lambda x: psi(x).mean())(signal)

    # Number of loops per run
    loops = 100
    # Number of runs
    runs = 25
    T = 2

    bs = 256
    means = []
    stds = []
    data = {}

    while T <= 1024:
        print("running ", T)
        signal = jnp.array(np.random.random([bs, T]))
        # %timeit jax.vmap(phi2, [0])(signal)
        # foo(signal)

        times1 = timeit.repeat("phi_(signal)", globals=globals(), repeat=runs, number=loops)
        print("phi", statistics.mean(times1), statistics.stdev(times1))

        times2 = timeit.repeat("psi_(signal)", globals=globals(), repeat=runs, number=loops)
        print("psi", statistics.mean(times2), statistics.stdev(times2))

        times3 = timeit.repeat("grad_phi(signal)", globals=globals(), repeat=runs, number=loops)
        print("grad_phi", statistics.mean(times3), statistics.stdev(times3))

        times4 = timeit.repeat("grad_psi(signal)", globals=globals(), repeat=runs, number=loops)
        print("grad_psi", statistics.mean(times4), statistics.stdev(times4))

        data[str(T)] = {"mask": times1, "recurrent": times2, "grad_mask": times3, "grad_recurrent": times4}
        means.append([statistics.mean(times1), statistics.mean(times2), statistics.mean(times3), statistics.mean(times4)])
        stds.append([statistics.stdev(times1), statistics.stdev(times2), statistics.stdev(times3), statistics.stdev(times4)])
        print(T, means[-1], stds[-1])

        T *= 2

        with open('signal_length.pkl', 'wb') as f:
            pickle.dump(data, f)
