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

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install -r requirements.txt
```

## Usage
```bash
python TrialModel.py
```
## Limitations
Can only handle small graphs. Custom Sparse backward method in utils.py needs debugging.


## License
[MIT](https://choosealicense.com/licenses/mit/)
