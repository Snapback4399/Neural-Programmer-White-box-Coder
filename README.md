# Neural Programmer White Box Coder (NPWC)

NPWC is a research project that focuses on bridging the gap between declarative deep learning and symbolic AI by introducing a new architecture that combines the benefits of both approaches.

The architecture is based on a neural network, called a _neural interpreter_, that takes as input a declaration of the structure of a solution and produces as output a program that solves the task. Neural interpreters can be trained to perform tasks by optimizing the structure of the declaration using gradient-based approaches. In this way, they can learn to implement programs from examples, without any coding.

In this repository, we provide an implementation of the neural interpreter model and some tasks that can be solved with it.

# Repository Structure

```bash
npwc
├── README.md
├── requirements.txt
├── setup.py
├── npwc
│   ├── __init__.py
│   ├── functions.py
│   ├── modules.py
│   ├── tasks.py
│   └── utils.py
```


`functions.py` contains various functions:
- `_validate_parameter(function, parameter_name, parameter_value)` validates function parameters;
- `one_hot()` one-hot encode an integer tensor in the way that converts it to a matrix of binary values;
- `_locally_connected_matmul()` multiplies two arrays using a locally connected approach (a variant of convolution);

`modules.py` includes implementations of modules:
- `LookupTableModule(npwc.Module)` is a lookup table module;
- `CompositionModule(npwc.Module)` is a module for function composition;
- `ConcatModule(npwc.Module)` concatenates output on all the input sequences of a module into a single vector;
- `SumModule(npwc.Module)` is a module for summation of vectors;
- `RNNModule(npwc.Module)` implements recurrent neural network module;
- `empty_module()` returns a module which gives a constant output of 1.;

`tasks.py` demonstrates some tasks a neural interpreter can perform (excerpt):
- `_build_dataset_XOR()` - dataset and objective function for XOR problem;
- `_build_dataset_ANDOR()` - dataset and objective function for AND/OR problem;
- `_build_dataset_SINE_REGRESSION()` - dataset and objective function for sine regression problem;
- `_build_dataset_COPY()` - dataset and objective function for copying task;
- `_build_dataset_LINKED_LIST()` - dataset and objective function for linked list task;
- `_build_dataset_RECURSION()` - dataset and objective function for recursion task;
- `_build_dataset_MERGE_SORT()` - dataset and objective function for merge sort task;

`utils.py` contains helpful utility functions:
- `image_grid(tensors, rows = -1, cols = 5, border = 1)` - given a list of tensors l, generate a single such that if l has shape (BATCH_SIZE, D, H, W) then the result has shape (cols * H + border, rows * W + border) and the batch is tiled along the first dimension so that the output looks like an image grid. It is assumed that all tensors in l have the same shape.;
- `print_model(label, model_desc)` displays any model either as a graph or as a flowchart depending on the settings;
- `plot_layout(obs, layout)` plots a layout as a digraph;

To replicate the environment for the project follow steps listed below.

Clone the project repository:

```bash
git clone https://github.com/pritoms/Neural-Programmer-White-box-Coder.git
cd Neural-Programmer-White-box-Coder
```

Bash snippet, to know the details of your system packages:

```bash
#!/bin/sh
apt list --installed python3  #or aptitude show python3
python3 --version
apt list --installed python-pip  #or aptitude show python-pip
pip --version
pip list
```

Set up virtual environment using `virtualenv`, and activate it:

```bash
apt update
apt install virtualenv
make venv
source venv/bin/activate
```

Install required dependecies in the new Python environment:

```bash
make requirements
```

Run tests:

```bash
make test
```

Start Jupyter server:

```bash
make run_jupyter
```



## Citation

The theoretical foundations of neural interpreters have been presented in this unpublished paper:

```markdown
@misc{NPWC,
    author = "Pritom Sarker",
    title = "Neural Programmer-White-box-Coder: Bridging the gap between declarative deep learning and symbolic AI",
    year = "2019-2020",
    howpublished = "\url{https://github.com/pritoms/Neural-Programmer-White-box-Coder}"} 
```


## Contribution

If you want to contribute, please contact the developer.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
