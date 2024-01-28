from torch_geometric.datasets import ZINC
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

import slrc
from tqdm import tqdm

class Zinc(object):

    def __init__(self):
        super(Zinc, self).__init__()
        
        self.criterion = F.mse_loss
        
        self.train = ZINC(root='/tmp/ZINC', subset=True, split='train', pre_transform=self.data_preprocessing)
        self.val = ZINC(root='/tmp/ZINC', subset=True, split='val', pre_transform=self.data_preprocessing)
        self.test = ZINC(root='/tmp/ZINC', subset=True, split='test', pre_transform=self.data_preprocessing)

    @staticmethod
    def data_preprocessing(data):
        data.x = data.x.float()
        return data

    def get_dataset(self, max_samples, gen_k_hop):
        X = self.train + self.val + self.test
        X = [data for data in X]
        if max_samples is not None and len(X) > max_samples:
            X = X[:max_samples]

        if gen_k_hop['do_generation']:
            for data in tqdm(X, desc='Creating k-hop graphs', unit='graphs'):
                data.k_hop_edge_index = slrc.create_k_hop_graph(data, k=gen_k_hop['k_hop']).edge_index
        

        dim_in, dim_out = self.get_dims()
        return X, {'dim0': dim_in, 'out_dim': dim_out, 'criterion': self.criterion}

    def get_dims(self):
        in_dim = self.train.num_node_features
        out_dim = 1
        return in_dim, out_dim
