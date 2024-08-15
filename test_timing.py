import jax
import jax.numpy as jnp
import numpy as np
from stljax.formula import *
from stljax.viz import *
import matplotlib.pyplot as plt
import timeit
import statistics
import pickle
import sys

if __name__ == "__main__":

    args = sys.argv[1:]
    filename = args[0]
    max_T = int(args[1])


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
    Ts = []
    data["functions"] = functions
    data["runs"] = runs
    data["loops"] = loops
    while T <= max_T:
        Ts.append(T)
        data['Ts'] = Ts
        print("running ", T)
        signal = jnp.array(np.random.random([bs, T]))
        times_list = []
        data[str(T)] = {}

        for f in functions:
            print("timing ", f)
            timeit.repeat(f + "(signal)", globals=globals(), repeat=1, number=1)
            times = timeit.repeat(f + "(signal)", globals=globals(), repeat=runs, number=loops)
            times_list.append(times)
            print("timing: ", statistics.mean(times), statistics.stdev(times))
            data[str(T)][f] = times
            with open(filename + '.pkl', 'wb') as f:
                pickle.dump(data, f)

        T *= 2


    # means = []
    # stds = []
    # for k in loaded_dict.keys():
    #     if k in ["Ts", "functions"]:
    #         break
    #     mus = []
    #     sts = []
    #     for f in loaded_dict[k].keys():
    #         mus.append(statistics.mean(loaded_dict[k][f])/loaded_dict["loops"])
    #         sts.append(statistics.stdev(loaded_dict[k][f])/loaded_dict["loops"])

    #     means.append(mus)
    #     stds.append(sts)
    # means = np.array(means)
    # stds = np.array(stds)

    # plt.plot(loaded_dict["Ts"], means * 1E3)
    # plt.yscale("log")
    # plt.legend(loaded_dict["functions"])
    # plt.grid()
    # plt.xlabel("signal length")
    # plt.ylabel("computation time [ms]")

