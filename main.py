from experiment import Experiment
from common import Task, GNN_TYPE, STOP, LAST_LAYER
import argparse as ap
import time

def get_fake_args(
        task=Task.NEIGHBORS_MATCH,
        type=GNN_TYPE.GCN,
        dim=32,
        depth=3,
        num_layers=None,
        train_fraction=0.8,
        max_epochs=50000,
        eval_every=100,
        batch_size=1024,
        accum_grad=1,
        patience=20,
        stop=STOP.TRAIN,
        loader_workers=0,
        last_layer=LAST_LAYER.REGULAR,
        no_layer_norm=False,
        no_activation=False,
        no_residual=False,
        unroll=False,
        max_samples=32000,
        learning_rate=0.001,
        weight_decay=0.0
):
    return {
        'task': task,
        'type': type,
        'dim': dim,
        'depth': depth,
        'num_layers': num_layers,
        'train_fraction': train_fraction,
        'max_epochs': max_epochs,
        'eval_every': eval_every,
        'batch_size': batch_size,
        'accum_grad': accum_grad,
        'stop': stop,
        'patience': patience,
        'loader_workers': loader_workers,
        'last_layer': last_layer,
        'no_layer_norm': no_layer_norm,
        'no_activation': no_activation,
        'no_residual': no_residual,
        'unroll': unroll,
        'max_samples': max_samples,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    }


def start(params):
    experiment = Experiment(params)
    experiment.run()

def run(params):
    start_time = time.time()
    start(params)
    end_time = time.time()
    print(f"Total time: {end_time - start_time}")  

if __name__ == '__main__':
    args = ap.ArgumentParser()
    args.add_argument('--type', type=str, default='GCN', choices=['GCN', 'GGNN', 'GIN', 'GAT'], help='GNN type to use')
    args.add_argument('--max_epochs', type=int, default=50000, help='Maximum number of epochs')
    args.add_argument('--depth', type=int, default=3, help='Depth of trees from NeighborsMatch dataset')
    args.add_argument('--last_layer', type=str, default='REGULAR', choices=['REGULAR', 'FULLY_ADJACENT', 'K_HOP'], help='Last layer type')
    args.add_argument('--max_samples', type=int, default=32000, help='Maximum number of samples to use from NeighborsMatch dataset')
    args.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args.add_argument('--stop', type=str, default='TRAIN', choices=['TRAIN', 'TEST'], help='Stop criterion')
    args = args.parse_args()

    params = get_fake_args(
            type=GNN_TYPE.from_string(args.type),
            max_epochs=args.max_epochs,
            depth=args.depth,
            last_layer=LAST_LAYER.from_string(args.last_layer),
            max_samples=args.max_samples,
            learning_rate=args.learning_rate,
            stop=STOP.from_string(args.stop)
    )
    run(params)