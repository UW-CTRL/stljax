stljax
======

A toolbox to compute the robustness of STL formulas using computation graphs. A jax version of the [STLCG toolbox originally implemented in PyTorch](https://github.com/StanfordASL/stlcg/tree/dev)

## Installation


Clone the repo. 

 Make a venv

`python3 -m venv stljax`

Go into the `stljax` folder. Then to install:

`pip install -e .`


## Usage

(TODO: upload demo)
`demo.ipynb` is an IPython jupyter notebook that showcases the basic functionality of the toolbox:
* Setting up signals for the formulas, including the use of Expressions
* Defining STL formulas and visualizing them
* Evaluating STL robustness, and robustness trace
* Gradient descent on STL parameters and signal parameters. On CPU and GPU (if available)


## Publications
Here are a list of publications that use stlcg. Please file an issue, or pull request to add your publication to the list.

K. Leung, and M. Pavone, "[Semi-Supervised Trajectory-Feedback Controller Synthesis for Signal Temporal Logic Specifications](https://arxiv.org/abs/2202.01997)," in American Control Conference, 2022.

K. Leung, N. Aréchiga, and M. Pavone, "[Backpropagation through STL specifications: Infusing logical structure into gradient-based methods](https://arxiv.org/abs/2008.00097)," International Journal of Robotics Research, 2022.

J. DeCastro, K. Leung, N. Aréchiga, and M. Pavone, "[Interpretable Policies from Formally-Specified Temporal Properties](http://asl.stanford.edu/wp-content/papercite-data/pdf/DeCastro.Leung.ea.ITSC20.pdf)," in Proc. IEEE Int. Conf. on Intelligent Transportation Systems, Rhodes, Greece, 2020.

K. Leung, N. Arechiga, and M. Pavone, "[Backpropagation for Parametric STL](http://asl.stanford.edu/wp-content/papercite-data/pdf/Leung.Arechiga.ea.ULAD19.pdf)," in IEEE Intelligent Vehicles Symposium: Workshop on Unsupervised Learning for Automated Driving, Paris, France, 2019.

When citing stlcg, please use the following Bibtex:
```
@Inproceedings{LeungArechigaEtAl2020,
  author       = {Leung, K. and Ar\'{e}chiga, N. and Pavone, M.},
  title        = {Backpropagation through signal temporal logic specifications: Infusing logical structure into gradient-based methods},
  booktitle    = {{Workshop on Algorithmic Foundations of Robotics}},
  year         = {2020},

}

@Article{LeungArechigaEtAl2020,
  author       = {Leung, K. and Ar\'{e}chiga, N. and Pavone, M.},
  title        = {Backpropagation through signal temporal logic specifications: Infusing logical structure into gradient-based methods},
  booktitle    = {{Int. Journal of Robotics Research}},
  year         = {2022},
  note         = {In press}

}
```

## Feedback
If there are any issues with the code, please make file an issue, or make a pull request.

