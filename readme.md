# MADG: Motif-Level Anomaly Detection in Dynamic Graphs

This repository contains the source code of the TIFS paper "Motif-Level Anomaly Detection in Dynamic Graphs".

## 📖 Abstract

As many real-world networks evolve over time, such as social networks, user-item networks, and IP-IP networks, anomaly detection for dynamic graphs has attracted growing attention. Most existing studies focus on detecting anomalous nodes or edges but fail to detect anomalous motif instances. In this paper, we propose MADG, a general Motif-level Anomaly Detection framework for dynamic Graphs, which can identify the anomaly in different motifs. Motifs are specific subgraph structures that frequently occur in a network and have been widely used in network analysis. In order to learn discriminative motif-level representations and leverage the temporal information from the dynamic graph, we design motif-augmented GCN and temporal self-attention. We first use motif-augmented GCN to model the topological structure among nodes and motif instances to learn their representations at each snapshot. Then, we feed the representations of multiple snapshots into the self-attention layer with relative temporal encoding in order to capture the evolutionary patterns. 


## 🚀 Quick Start

### Prerequisites

- Python ≥ 3.9 (validated on 3.12.3)
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MADG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

Run the main experiment:
```bash
python main.py
```

#### Command Line Arguments

```bash
python main.py --help
```

Available options:
- `--dataset`: Dataset name (uci/dnc/alpha/otc) [default: uci]
- `--hidden_dim`: Hidden embedding dimension [default: 64]
- `--epoch`: Training epochs [default: 10]
- `--num_heads`: Number of attention heads [default: 2]
- `--n_layers`: Number of GCN layers [default: 2]
- `--lr`: Learning rate [default: 5e-5]
- `--p`: Anomalous rate [default: 0.1]
- `--motif`: Motif type (triangle/circle) [default: triangle]
- `--dropout`: Dropout rate [default: 0.1]
- `--device`: Device (cuda/cpu) [default: cuda]

#### Example Usage

```bash
# Run on UCI dataset with triangle motifs
python main.py --dataset uci --motif triangle --epoch 50

# Run on Bitcoin Alpha dataset with circle motifs
python main.py --dataset alpha --motif circle --hidden_dim 128 --epoch 100
```

## 📁 Project Structure

```
MADG/
├── main.py                    # Main training script
├── model.py                   # MADG model implementation
├── layers.py                  # Graph neural network layers
├── Graph.py                   # Graph data structures and snapshots
├── utils.py                   # Utility functions
├── save_motif_list.py         # Motif list generation
├── generate anomaly_data.py   # Anomaly data generation
├── requirements.txt           # Dependencies
├── raw_data/                  # Raw datasets
│   ├── email-dnc.edges
│   ├── soc-sign-bitcoinalpha.csv
│   ├── soc-sign-bitcoinotc.csv
│   └── ...
└── motif_list/               # Pre-computed motif lists
    ├── alpha/
    ├── dnc/
    ├── otc/
    └── uci/
```


## 📄 Citation

Please cite our paper if it is helpful to your work:

```latex
@article{yuan2023motif,
  title={Motif-level Anomaly Detection in Dynamic Graphs},
  author={Yuan, Zirui and Shao, Minglai and Yan, Qiben},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2023},
  publisher={IEEE}
}
```
