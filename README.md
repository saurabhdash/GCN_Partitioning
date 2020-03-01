# GCN Partitioning
Graph Partitoning Using Graph Convolutional Networks as described in [GAP: Generalizable Approximate Graph Partitioning Framework](https://arxiv.org/abs/1903.00614) 

## Loss Backward Equations
To handle large graphs, the loss function is implemented using sparse torch tensors using a custom loss class.

If ![Z = (Y / \Gamma)(1 - Y)^{T} \circ A ](https://render.githubusercontent.com/render/math?math=Z%20%3D%20(Y%20%2F%20%5CGamma)(1%20-%20Y)%5E%7BT%7D%20%5Ccirc%20A%20)

where Y_{ij} is the probability of node i being in partition j.

![L = \sum_{A_{lm} \neq 0} Z_{lm} ](https://render.githubusercontent.com/render/math?math=L%20%3D%20%5Csum_%7BA_%7Blm%7D%20%5Cneq%200%7D%20Z_%7Blm%7D%20)

Then the gradients can be calculated by the equations: 

![\frac{\partial z_{i \alpha}}{\partial y_{ij}} = A_{i \alpha} \left(\frac{\Gamma_{j} (1 - y_{\alpha j}) - y_{ij}(1 - y_{\alpha j})D_{i}}{\Gamma_{j}^{2}}\right)](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20z_%7Bi%20%5Calpha%7D%7D%7B%5Cpartial%20y_%7Bij%7D%7D%20%3D%20A_%7Bi%20%5Calpha%7D%20%5Cleft(%5Cfrac%7B%5CGamma_%7Bj%7D%20(1%20-%20y_%7B%5Calpha%20j%7D)%20-%20y_%7Bij%7D(1%20-%20y_%7B%5Calpha%20j%7D)D_%7Bi%7D%7D%7B%5CGamma_%7Bj%7D%5E%7B2%7D%7D%5Cright))

![\frac{\partial z_{\alpha i}}{\partial y_{ij}} = A_{\alpha i} \left(\frac{\Gamma_{j} (- y_{\alpha j}) - y_{\alpha j}(1 - y_{ij})D_{i}}{\Gamma_{j}^{2}}\right)](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20z_%7B%5Calpha%20i%7D%7D%7B%5Cpartial%20y_%7Bij%7D%7D%20%3D%20A_%7B%5Calpha%20i%7D%20%5Cleft(%5Cfrac%7B%5CGamma_%7Bj%7D%20(-%20y_%7B%5Calpha%20j%7D)%20-%20y_%7B%5Calpha%20j%7D(1%20-%20y_%7Bij%7D)D_%7Bi%7D%7D%7B%5CGamma_%7Bj%7D%5E%7B2%7D%7D%5Cright))

![\frac{\partial z_{i^{'} \alpha}}{\partial y_{ij}} = A_{i^{'} \alpha} \left(\frac{(1 - y_{\alpha j}) y_{i^{'}j}D_{i}}{\Gamma_{j}^{2}}\right) \;\;\; i^{'}, \alpha \neq i](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20z_%7Bi%5E%7B'%7D%20%5Calpha%7D%7D%7B%5Cpartial%20y_%7Bij%7D%7D%20%3D%20A_%7Bi%5E%7B'%7D%20%5Calpha%7D%20%5Cleft(%5Cfrac%7B(1%20-%20y_%7B%5Calpha%20j%7D)%20y_%7Bi%5E%7B'%7Dj%7DD_%7Bi%7D%7D%7B%5CGamma_%7Bj%7D%5E%7B2%7D%7D%5Cright)%20%5C%3B%5C%3B%5C%3B%20i%5E%7B'%7D%2C%20%5Calpha%20%5Cneq%20i)

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
