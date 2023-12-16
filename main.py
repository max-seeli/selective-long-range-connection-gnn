from attrdict import AttrDict

from experiment import Experiment
from common import Task, GNN_TYPE, STOP


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
        last_layer_fully_adjacent=False,
        no_layer_norm=False,
        no_activation=False,
        no_residual=False,
        unroll=False,
):
    return AttrDict({
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
        'last_layer_fully_adjacent': last_layer_fully_adjacent,
        'no_layer_norm': no_layer_norm,
        'no_activation': no_activation,
        'no_residual': no_residual,
        'unroll': unroll,
    })


def start(params):
    experiment = Experiment(params)
    experiment.run()