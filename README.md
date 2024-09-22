# Graph Auto Encoder Implementation using PyG

### Requirements
- torch
- torch_geometric
To spin up a conda env: run `conda create -n <env-name> --file requirements.txt` at the root level of the project

### To run
run the training loop with `python main.py --mode train --dataset cora`

currently, for `--dataset`, the available options are `cora`, `pubmed`, and `citeseer`