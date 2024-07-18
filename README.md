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

(TODO: update demo)
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




## TODOs
- Make the demo notebook better
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

