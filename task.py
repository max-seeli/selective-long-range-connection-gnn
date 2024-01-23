from enum import Enum, auto

from models.tree_neighbors_model import TreeNeighborsModel
from tasks.tree_neighbors_match import TreeNeighborsMatch


class Task(Enum):
    NEIGHBORS_MATCH = auto()

    @staticmethod
    def from_string(s):
        try:
            return Task[s]
        except KeyError:
            raise ValueError()

    def get_dataset(self, depth, train_fraction, max_samples, k_hop):
        if self is Task.NEIGHBORS_MATCH:
            dataset = TreeNeighborsMatch(depth)
        else:
            dataset = None

        return dataset.generate_data(train_fraction, max_samples, k_hop)

    def get_model(self, args, dataset_args):
        if self is Task.NEIGHBORS_MATCH:
            return TreeNeighborsModel(
                gnn_type=args['type'],
                num_layers=args['num_layers'],
                dim0=dataset_args['dim0'],
                h_dim=args['dim'],
                out_dim=dataset_args['out_dim'],
                last_layer=args['last_layer'],
                unroll=args['unroll'],
                layer_norm=not args['no_layer_norm'],
                use_activation=not args['no_activation'],
                use_residual=not args['no_residual']
            )
        else:
            return None