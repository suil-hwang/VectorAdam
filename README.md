# VectorAdam for Rotation Equivariant Geometry Optimization

Official PyTorch implementation of:

VectorAdam for Rotation Equivariant Geometry Optimization  
[Selena Ling](https://www.iszihan.github.io), [Nicholas Sharp](https://nmwsharp.com/), [Alec Jacobson](https://www.cs.toronto.edu/~jacobson/)  
NeurIPS 2022

## Setup

This repo now uses a `src/` package layout and is intended to be installed with editable mode.

1. Create or update the conda environment from `environments.yml`:

```bash
# first time
conda env create -f environments.yml

# or update existing env
conda env update -f environments.yml --prune
```

2. Activate the environment:

```bash
conda activate vectoradam
```

3. Install this repository in editable mode:

```bash
pip install -e .
```

`environments.yml` currently targets Python 3.10 and CUDA 11.8 on Windows.

## Package Usage

```python
from vectoradam import VectorAdam

optimizer = VectorAdam(
    [
        {"params": X, "axis": -1},
        {"params": Y, "axis": 1},
    ],
    lr=lr,
    betas=betas,
    eps=eps,
)
```

Utility functions are also exposed from the package:

```python
from vectoradam import create_circle, laplacian_uniform_2d, plot_mesh2d
```

## Demo

Demo files are in `demo/`:

- `demo/laplacian2d_demo.ipynb`
- `demo/laplacian2d_demo.py`

Run the script demo:

```bash
python demo/laplacian2d_demo.py --show
```

Optional flags:

```bash
python demo/laplacian2d_demo.py --steps 5 --device auto --save-dir demo/outputs
```
