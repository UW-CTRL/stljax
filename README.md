stljax
======

A toolbox to compute the robustness of STL formulas using computation graphs. This is the jax version of the [STLCG toolbox originally implemented in PyTorch](https://github.com/StanfordASL/stlcg/tree/dev).


## Installation

Requires Python 3.10+

Clone the repo. 

 Make a venv and activate it

`python3 -m venv stljax_venv`

`source stljax_venv/bin/activate`

Go into the `stljax` folder. Then to install:

`pip install -e .`


## Usage
`demo.ipynb` is an IPython jupyter notebook that showcases the basic functionality of the toolbox:
* Setting up signals for the formulas, including the use of Expressions
* Defining STL formulas and visualizing them
* Evaluating STL robustness, and robustness trace
* Gradient descent on STL parameters and signal parameters.


## (New) Features
stljax leverages to benefits of jax and automatic differentiation!

Aside from using jax as the backend, stljax is more recent and tidier implementation of stlcg which was originally implemented in PyTorch back ~2019.

- Removed the `distributed_mean` hack. jax keeps track of multiple max/min values and will distribute the gradients across all max/min values!
- Incorporation of the smooth max/min presented in [Optimization with Temporal and Logical Specifications via Generalized Mean-based Smooth Robustness Measures](https://arxiv.org/abs/2405.10996) by Samet Uzun, Purnanand Elango, Pierre-Loic Garoche, Behcet Acikmese
    - Use `approx_method="gmsr"` and `temperature=(eps, p)`


## Quick intro

### Defining STL formulas and computing robustness values

There are two ways to define an STL formula. Using either the `Expression` and `Predicate` classes.

#### Using `Expression`
With `Expression`, you are essentially defining a signal whose values are the output of a predicate function computed external to the STL robustness computation formula.
Essentially, you process your desired signal first (e.g., from a state trajectory, you compute velocity), and then you pass it directly into the formula.

A step-by-step break down:\
1. Suppose you have a `trajectory` that is an array of size `[bs, time_steps, state_dim]`  (not reversed in time)

2. Suppose we have a `get_velocity()`  function and a `get_acceleration()` function:\
 `velocity_value = get_velocity(trajectory)   # [bs, time_steps, 1]`\
 `acceleration_value = get_acceleration(trajectory)   # [bs, time_steps, 1]`

3. Then, we can define the following two `Expression` objects:\
`velocity_exp = Expression("velocity," value=velocity_value, reverse=False, time_dim=1)`\
`acceleration_exp = Expression("acceleration", value=acceleration_value, reverse=False, time_dim=1)`

4. With these two expressions, we can define an STL formula `ϕ = □ (velocity_exp > 5.0) ∨ ◊ (acceleration_exp > 5.0)` which is equivalent to `ϕ = Always(velocity_exp > 5.0) & Eventually(acceleration_exp > 5.0)`.

5. To compute the robustness trace of `ϕ`, we run `ϕ((velocity_exp, acceleration_exp))` where the input is a _tuple_ since the first part of the formula depends on velocity, and the second part depends on acceleration.

This means that the user needs to compute velocity and acceleration values _before_ calling `ϕ` to compute the robustness trace (or `ϕ.robustness((velocity_exp, acceleration_exp))` for the robustness value)

**NOTE**: Expressions are used to _define_ an STL formula. While you can, you don't necessarily need to use Expressions as inputs for computing robustness values. However it may be convenient as it will automatically handle reversing your signal (if needed). You can directly feed in a `jnp.array`. For example, `ϕ((velocity_array, acceleration_array))` also works. The downside is that the user needs to make sure the inputs are reversed as there is no way for `stljax` to know. The upside is that you can use `jax.jit` and make it run really fast (see demo notebook)!



##### Using `Predicate`
With `Predicate`, this is more true to the STL definition. You pass a predicate function when defining an STL formula, rather than passing the signal that would be the output of a predicate function.
Essentially, you pass your N-D input (e.g., state trajectory) directly into the formula when computing robustness values.


A step-by-step break down:\
1. Suppose you have a `trajectory` that is an array of size `[bs, time_steps, state_dim]`  (not reversed in time)

2. Suppose we have a `get_velocity()`  function and a `get_acceleration()` function:\
 `velocity_value = get_velocity(trajectory)   # [bs, time_steps, 1]`\
 `acceleration_value = get_acceleration(trajectory)   # [bs, time_steps, 1]`

3. Then, we can define the following two `Predicate` objects:\
`velocity_pred = Predicate("velocity", predicate_function=get_velocity, reverse=False, time_dim=1)`\
`acceleration_pred = Predicate("acceleration", predicate_function=get_acceleration, reverse=False, time_dim=1)`

4. With these two `Predicate` objects, we can define an STL formula `ϕ = □ (velocity_pred > 5.0) ∨ ◊ (acceleration_pred > 5.0)` which is equivalent to `ϕ = Always(velocity_pred > 5.0) & Eventually(acceleration_pred > 5.0)`.

5. To compute the robustness trace of `ϕ`, we run `ϕ(trajectory)` where the input is what all the predicate functions expect the input to be.



**In summary**:\
When using Predicates to define STL formulas, it will extract the velocity and acceleration values _inside_ the robustness computation. Whereas when using Expressions, you need to extract the velocity and acceleration _outside_ of the robustness computation.






## TODOs
- finalize expected signal dimensions. Should we be strict about the expected signal size: `[batch_size, time_dim, state_dim]` and `time_dim=-1`. Is `state_dim` even necessary?
- re-implement stlcg (PyTorch) with the latest version of PyTorch.


## Publications
Here are a list of publications that use stlcg/stljax. Please file an issue, or pull request to add your publication to the list.

K. Leung, and M. Pavone, "[Semi-Supervised Trajectory-Feedback Controller Synthesis for Signal Temporal Logic Specifications](https://arxiv.org/abs/2202.01997)," in American Control Conference, 2022.

K. Leung, N. Aréchiga, and M. Pavone, "[Backpropagation through STL specifications: Infusing logical structure into gradient-based methods](https://arxiv.org/abs/2008.00097)," International Journal of Robotics Research, 2022.

J. DeCastro, K. Leung, N. Aréchiga, and M. Pavone, "[Interpretable Policies from Formally-Specified Temporal Properties](http://asl.stanford.edu/wp-content/papercite-data/pdf/DeCastro.Leung.ea.ITSC20.pdf)," in Proc. IEEE Int. Conf. on Intelligent Transportation Systems, Rhodes, Greece, 2020.

K. Leung, N. Arechiga, and M. Pavone, "[Backpropagation for Parametric STL](http://asl.stanford.edu/wp-content/papercite-data/pdf/Leung.Arechiga.ea.ULAD19.pdf)," in IEEE Intelligent Vehicles Symposium: Workshop on Unsupervised Learning for Automated Driving, Paris, France, 2019.

When citing stlcg/stljax, please use the following Bibtex:
```
# journal paper
@Article{LeungArechigaEtAl2020,
  author       = {Leung, K. and Ar\'{e}chiga, N. and Pavone, M.},
  title        = {Backpropagation through signal temporal logic specifications: Infusing logical structure into gradient-based methods},
  booktitle    = {{Int. Journal of Robotics Research}},
  year         = {2022},
}

# conference paper
@Inproceedings{LeungArechigaEtAl2020,
  author       = {Leung, K. and Ar\'{e}chiga, N. and Pavone, M.},
  title        = {Backpropagation through signal temporal logic specifications: Infusing logical structure into gradient-based methods},
  booktitle    = {{Workshop on Algorithmic Foundations of Robotics}},
  year         = {2020},
}
```

## Feedback
If there are any issues with the code, please make file an issue, or make a pull request.

