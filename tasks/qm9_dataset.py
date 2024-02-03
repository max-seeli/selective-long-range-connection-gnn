from torch_geometric.datasets import QM9
from torch.nn import L1Loss

import slrc
from tqdm import tqdm

class Qm9(object):

    def __init__(self):
        super(Qm9, self).__init__()
        
        self.criterion = L1Loss()
        
        self.data = QM9(root='/tmp/QM9', pre_transform=self.data_preprocessing)
        
    @staticmethod
    def data_preprocessing(data):
        data.x = data.x.float()
        return data

    def get_dataset(self, max_samples, gen_k_hop):
        X = self.data
        X = [data for data in X]
        if max_samples is not None and len(X) > max_samples:
            X = X[:max_samples]

        if gen_k_hop['do_generation']:
            for data in tqdm(X, desc='Creating k-hop graphs', unit='graphs'):
                data.k_hop_edge_index = slrc.create_k_hop_graph(data, k=gen_k_hop['k_hop']).edge_index
        

        dim_in, dim_out = self.get_dims()
        return X, {'dim0': dim_in, 'out_dim': dim_out, 'criterion': self.criterion}

    def get_dims(self):
        in_dim = self.data.num_node_features
        out_dim = 19
        return in_dim, out_dim
