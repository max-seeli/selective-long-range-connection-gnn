from experiment import Experiment
from common import GNN_TYPE, STOP, LAST_LAYER
from task import Task
import argparse as ap
import time

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
    args.add_argument('--task', type=str, default='NEIGHBORS_MATCH', choices=['NEIGHBORS_MATCH', 'ZINC'], help='Task to run')
    args.add_argument('--type', type=str, default='GCN', choices=['GCN', 'GGNN', 'GIN', 'GAT'], help='GNN type to use')
    args.add_argument('--dim', type=int, default=32, help='Hidden dimension')
    args.add_argument('--depth', type=int, default=3, help='Depth of trees from NeighborsMatch dataset (ignored if task is not NeighborsMatch)')
    args.add_argument('--num_layers', type=int, default=3, help='Number of layers')
    args.add_argument('--train_fraction', type=float, default=0.8, help='Fraction of training examples')
    args.add_argument('--max_epochs', type=int, default=50000, help='Maximum number of epochs')
    args.add_argument('--eval_every', type=int, default=100, help='Evaluate every X epochs')
    args.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    args.add_argument('--accum_grad', type=int, default=1, help='Number of gradient accumulation steps')
    args.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    args.add_argument('--stop', type=str, default='TRAIN', choices=['TRAIN', 'TEST'], help='Stop criterion')
    args.add_argument('--loader_workers', type=int, default=0, help='Number of workers for data loader')
    args.add_argument('--last_layer', type=str, default='REGULAR', choices=['REGULAR', 'FULLY_ADJACENT', 'K_HOP'], help='Last layer type')
    args.add_argument('--k_hop', type=int, default=3, help='Number of hops for K-Hop layer (ignored if last layer is not K_HOP)')
    args.add_argument('--no_layer_norm', action='store_true', help='Do not use layer normalization')
    args.add_argument('--no_activation', action='store_true', help='Do not use activation function')
    args.add_argument('--no_residual', action='store_true', help='Do not use residual connections')
    args.add_argument('--unroll', action='store_true', help='Unroll the GNN')
    args.add_argument('--max_samples', type=int, default=32000, help='Maximum number of samples to use from NeighborsMatch dataset')
    args.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    args.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    args = args.parse_args()

    task = Task.from_string(args.task)
    type = GNN_TYPE.from_string(args.type)
    last_layer = LAST_LAYER.from_string(args.last_layer)
    stop = STOP.from_string(args.stop)

    params = {
        'task': task,
        'type': type,
        'dim': args.dim,
        'depth': args.depth,
        'num_layers': args.num_layers,
        'train_fraction': args.train_fraction,
        'max_epochs': args.max_epochs,
        'eval_every': args.eval_every,
        'batch_size': args.batch_size,
        'accum_grad': args.accum_grad,
        'patience': args.patience,
        'stop': stop,
        'loader_workers': args.loader_workers,
        'last_layer': last_layer,
        'k_hop': args.k_hop,
        'no_layer_norm': args.no_layer_norm,
        'no_activation': args.no_activation,
        'no_residual': args.no_residual,
        'unroll': args.unroll,
        'max_samples': args.max_samples,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
    }
    run(params)