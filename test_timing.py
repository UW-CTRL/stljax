import numpy as np
from stljax.torch_src import *
import matplotlib.pyplot as plt
import timeit
import statistics
import pickle
import sys
import torch

if __name__ == "__main__":

    args = sys.argv[1:]
    filename = args[0]
    max_T = int(args[1])
    bs = int(args[2])


    mask = Always(GreaterThan('x', torch.tensor([0.])))
    recurrent = AlwaysRecurrent(GreaterThan('x', torch.tensor([0.])))

    def foo(signal):
        return mask(signal).mean()

    def mask_(signal):
        return torch.vmap(foo)(signal)

    def grad_mask(signal):
        return torch.vmap(torch.func.grad(foo))(signal)

    @torch.jit.script
    def foo_jit(signal):
        return mask(signal).mean()

    def mask_jit(signal):
        return torch.vmap(foo_jit)(signal)

    def grad_mask_jit(signal):
        return torch.vmap(torch.func.grad(foo_jit))(signal)


    def goo(signal):
        return recurrent(signal).mean()

    def recurrent_(signal):
        return torch.vmap(recurrent)(signal)

    def grad_recurrent(signal):
        return torch.vmap(torch.func.grad(goo))(signal)

    @torch.jit.script
    def goo_jit(signal):
        return recurrent(signal).mean()

    def recurrent_jit(signal):
        return torch.vmap(goo_jit)(signal)

    def grad_recurrent_jit(signal):
        return torch.vmap(torch.func.grad(goo_jit))(signal)

    # Number of loops per run
    loops = 100
    # Number of runs
    runs = 25
    T = 2

    # bs =
    means = []
    stds = []
    data = {}

    # functions = ["mask_", "recurrent_", "grad_mask", "grad_recurrent", "mask_jit", "recurrent_jit", "grad_mask_jit", "grad_recurrent_jit"]
    functions = ["mask_", "grad_mask", "mask_jit", "grad_mask_jit", "recurrent_", "grad_recurrent", "recurrent_jit", "grad_recurrent_jit"]
    Ts = []
    data["functions"] = functions
    data["runs"] = runs
    data["loops"] = loops
    while T <= max_T:
        Ts.append(T)
        data['Ts'] = Ts
        print("running ", T)
        signal = torch.rand([bs, T])
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

