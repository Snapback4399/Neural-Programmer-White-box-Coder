# Neural Programmer-White-box-Coder (NPWC)

NPWC is a research project that focuses on bridging the gap between declarative deep learning and symbolic AI by introducing a new architecture that combines the benefits of both approaches.

The architecture is based on a neural network, called a _neural interpreter_, that takes as input a declaration of the structure of a solution and produces as output a program that solves the task. Neural interpreters can be trained to perform tasks by optimizing the structure of the declaration using gradient-based approaches. In this way, they can learn to implement programs from examples, without any coding.

In this repository, we provide an implementation of the neural interpreter model and some tasks that can be solved with it.

# Repository Structure

```bash
Neural Interpreter (NPWC)
|
├── README.md
├── requirements.txt
├── setup.py
├── npwc
│   ├── __init__.py
│   ├── functions.py
│   ├── modules.py
│   ├── tasks.py
│   └── utils.py
└── tests
```

## Installation

The library is implemented in Python 3 and requires a few dependencies:

* [Pytorch](https://pytorch.org/);
* [Pytorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/); and finally,
* [Tqdm](https://tqdm.github.io/).

The code can be installed by cloning the repository and running the setup script.

```bash
git clone https://github.com/pritoms/Neural-Programmer-White-box-Coder.git
cd NPWC
pip install -e .
```

## Examples

We provide a few examples that demonstrate how to use the neural interpreter architecture to solve tasks:

* [Sorting digits](examples/sorting_digits.ipynb): a simple example that sorts a list of digits; and
* [Math problems](examples/math_problems.ipynb): a more complex example that solves mathematical problems from natural language questions.
