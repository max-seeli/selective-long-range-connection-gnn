from attrdict import AttrDict

from experiment import Experiment
from common import Task, GNN_TYPE, STOP, LAST_LAYER


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
        'last_layer': last_layer,
        'no_layer_norm': no_layer_norm,
        'no_activation': no_activation,
        'no_residual': no_residual,
        'unroll': unroll,
        'max_samples': max_samples,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay
    })


def start(params):
    experiment = Experiment(params)
    experiment.run()