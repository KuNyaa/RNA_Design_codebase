# Branch #1: mRNA Design

This repository contains our mRNA design system. We leverage DyNet for gradient computation, and we have provided sampled target proteins of different lengths to test the system. 

## Data
The data for this project are sampled target proteins of various lengths. You can find these data files in the `data` directory, named as `protein*.txt`.

## Setup

This system utilizes DyNet for gradient computations. To install DyNet, please use the following pip command:

```bash
pip install git+https://github.com/clab/dynet#egg=dynet
```

## Usage

The entry point to the system is through `main_*.py` files. These scripts are designed to accept protein sequences as input and then generate the corresponding mRNA sequences.

Here is an example of how to use the `main_PG.py` script to design mRNA based on the protein sequence provided in `protein10.txt`:

```bash
cat ./data/protein10.txt | python main_PG.py
```

Please replace `protein10.txt` with the filename that corresponds to the protein sequence you want to use, and `main_PG.py` with the script suitable for your specific needs.