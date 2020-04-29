# GCN Partitioning
Graph Partitoning Using Graph Convolutional Networks as described in [GAP: Generalizable Approximate Graph Partitioning Framework](https://arxiv.org/abs/1903.00614) 


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
To run the algorithm place the Hypergraphs in ./hgr_files folder:
```bash
python MinCutSingleHGR.py --circuit=[Circuit] --beta=1 --parts=2
```
[circuit] can be fract, industry2 etc.

To check the generalization ability of GAP, run 
```bash
python MinCutMultiHGR.py --circuit=[Circuit] --beta=1
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
