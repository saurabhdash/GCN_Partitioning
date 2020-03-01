# GCN Partitioning
Graph Partitoning Using Graph Convolutional Networks as described in [GAP: Generalizable Approximate Graph Partitioning Framework](https://arxiv.org/abs/1903.00614) 

## Equations

![\frac{\partial z_{i \alpha}}{\partial y_{ij}} = A_{i \alpha} \left(\frac{\Gamma_{j} (1 - y_{\alpha j}) - y_{ij}(1 - y_{\alpha j})D_{i}}{\Gamma_{j}^{2}}\right)](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20z_%7Bi%20%5Calpha%7D%7D%7B%5Cpartial%20y_%7Bij%7D%7D%20%3D%20A_%7Bi%20%5Calpha%7D%20%5Cleft(%5Cfrac%7B%5CGamma_%7Bj%7D%20(1%20-%20y_%7B%5Calpha%20j%7D)%20-%20y_%7Bij%7D(1%20-%20y_%7B%5Calpha%20j%7D)D_%7Bi%7D%7D%7B%5CGamma_%7Bj%7D%5E%7B2%7D%7D%5Cright))


## Installation
Create a virtual environment using venv

```bash
python3 -m venv env
```

Source the virtual environment

```bash
source env/bin/activate
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.

```bash
pip install -r requirements.txt
```

## Usage
```bash
python TrialModel.py
```
## Limitations
Has only been tested on small custom graphs.

## License
[MIT](https://choosealicense.com/licenses/mit/)
