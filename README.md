# Selective Long Range Connection GNNs
This repository contains the code for my seminar papar "Enhancing Message Passing Neural Networks with Selective Long-Range Connections to Mitigate Over-Squashing"

## Usage
To run the code, you need to install the requirements first. This can be done by running
```bash
conda env create -f environment.yml
conda activate slrc-gnn
```
Then, you can run the code by running
```bash
python main.py
```

To configure the experiment, you can change the parameters in the call of `main.py`. The following parameters are available:
```bash
main.py [-h] [--type {GCN,GGNN,GIN,GAT}] [--max_epochs MAX_EPOCHS] [--depth DEPTH] [--last_layer {REGULAR,FULLY_ADJACENT,K_HOP}] [--max_samples MAX_SAMPLES]
               [--learning_rate LEARNING_RATE] [--stop {TRAIN,TEST}]

options:
  -h, --help            show this help message and exit
  --type {GCN,GGNN,GIN,GAT}
                        GNN type to use
  --max_epochs MAX_EPOCHS
                        Maximum number of epochs
  --depth DEPTH         Depth of trees from NeighborsMatch dataset
  --last_layer {REGULAR,FULLY_ADJACENT,K_HOP}
                        Last layer type
  --max_samples MAX_SAMPLES
                        Maximum number of samples to use from NeighborsMatch dataset
  --learning_rate LEARNING_RATE
                        Learning rate
  --stop {TRAIN,TEST}   Stop criterion
```

## Idea
The project addresses the issue of over-squashing in processing large-scale graph data by establishing selective long-range connections in the network's final layer. This approach balances global information flow and local feature distinctiveness, improving MPNNs’ applicability and accuracy in complex scenarios. The selective long-range connections are chosen by a distance measure between nodes. The distance measure is based on the shortest path between nodes in the graph.

## Results
The raw training results of the experiments can be found in the `results` folder. For a detailed analysis of the results, please refer to the seminar paper in the file `seminar-paper.pdf`. 





