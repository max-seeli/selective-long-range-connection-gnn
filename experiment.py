import torch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import KFold

import numpy as np
import random

from common import STOP, LAST_LAYER


class Experiment():
    def __init__(self, args):


        self.task = args['task']
        self.max_epochs = args['max_epochs']
        self.batch_size = args['batch_size']
        self.accum_grad = args['accum_grad']
        self.eval_every = args['eval_every']
        self.loader_workers = args['loader_workers']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.stopping_criterion = args['stop']
        self.patience = args['patience']
        self.learning_rate = args['learning_rate']
        self.weight_decay = args['weight_decay']

        seed = 11
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        gen_k_hop = {
            'do_generation': args['last_layer'] == LAST_LAYER.K_HOP,
            'k_hop': args['k_hop']
        }
        self.X, dataset_args = \
            self.task.get_dataset(args['depth'], args['max_samples'], gen_k_hop)
        
        self.k_fold = KFold(n_splits=args['k_fold'], shuffle=True, random_state=seed)

        self.criterion = dataset_args['criterion']

        self.gen_model = lambda: self.task.get_model(args, dataset_args).to(self.device)

        print(f'Starting experiment')
        self.print_args(args)
        print(f'Number of observations: {len(self.X)}')

    def print_args(self, args):
        for key, value in args.items():
            print(f"{key}: {value}")
        print()

    def run(self):
        print('Starting training')

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(self.k_fold.split(self.X)):
            X_train = [self.X[i] for i in train_idx]
            X_test = [self.X[i] for i in test_idx]

            model = self.gen_model()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            mode = 'max' if self.task.is_classification else 'min'
            scheduler = ReduceLROnPlateau(optimizer, mode=mode, threshold_mode='abs', factor=0.5, patience=10)
        

            best_test = 0.0 if self.task.is_classification else np.inf
            best_train = 0.0 if self.task.is_classification else np.inf
            best_epoch = 0
            epochs_no_improve = 0
            for epoch in range(1, (self.max_epochs // self.eval_every) + 1):
                model.train()
                loader = DataLoader(X_train * self.eval_every, batch_size=self.batch_size, shuffle=True,
                                    pin_memory=True, num_workers=self.loader_workers)

                total_loss = 0
                total_num_examples = 0
                train_correct = 0
                optimizer.zero_grad()
                for i, batch in enumerate(loader):
                    batch = batch.to(self.device)
                    out = model(batch)
                    loss = self.criterion(input=out, target=batch.y)
                    total_num_examples += batch.num_graphs
                    total_loss += (loss.item() * batch.num_graphs)
                    if self.task.is_classification:
                        _, train_pred = out.max(dim=1)
                        train_correct += train_pred.eq(batch.y).sum().item()

                    loss = loss / self.accum_grad
                    loss.backward()
                    if (i + 1) % self.accum_grad == 0:
                        optimizer.step()
                        optimizer.zero_grad()

                avg_training_loss = total_loss / total_num_examples
                if self.task.is_classification:
                    train_perf = train_correct / total_num_examples
                    comp = lambda x, y, t: x > y + t 
                else:
                    train_perf = avg_training_loss
                    comp = lambda x, y, t: x < y - t
                test_perf = self.eval(model, X_test)

                scheduler.step(train_perf)
                cur_lr = [g["lr"] for g in optimizer.param_groups]

                new_best_str = ''
                stopping_threshold = 0.0001
                if self.stopping_criterion is STOP.TEST:
                    if comp(test_perf, best_test, stopping_threshold):
                        best_test = test_perf
                        best_train = train_perf
                        best_epoch = epoch
                        epochs_no_improve = 0
                        new_best_str = ' (new best test)'
                    else:
                        epochs_no_improve += 1
                elif self.stopping_criterion is STOP.TRAIN:
                    if comp(train_perf, best_train, stopping_threshold):
                        best_train = train_perf
                        best_test = test_perf
                        best_epoch = epoch
                        epochs_no_improve = 0
                        new_best_str = ' (new best train)'
                    else:
                        epochs_no_improve += 1

                if self.task.is_classification:
                    print(f'Epoch {epoch * self.eval_every} @ {fold + 1}, LR: {cur_lr}: Train loss: {avg_training_loss:.7f}, Train acc: {train_perf:.4f}, Test accuracy: {test_perf:.4f}{new_best_str}')
                else:
                    print(f'Epoch {epoch * self.eval_every} @ {fold + 1}, LR: {cur_lr}: Train loss: {avg_training_loss:.7f}, Test loss: {test_perf:.7f}{new_best_str}')

                if epochs_no_improve >= self.patience:
                    print(
                        f'{self.patience} * {self.eval_every} epochs without {self.stopping_criterion} improvement, stopping. ')
                    break
            print(f'Best train perf: {best_train}, epoch: {best_epoch * self.eval_every}')
            print(f'Fold {fold + 1} completed')

            fold_results.append((best_train, best_test, best_epoch * self.eval_every))

        print(f'Average train perf: {np.mean([x[0] for x in fold_results])} +/- {np.std([x[0] for x in fold_results])}')
        print(f'Average test perf: {np.mean([x[1] for x in fold_results])} +/- {np.std([x[1] for x in fold_results])}')
        print(f'Average epoch: {np.mean([x[2] for x in fold_results])} +/- {np.std([x[2] for x in fold_results])}')
        return fold_results

    def eval(self, model, X_test):
        model.eval()
        with torch.no_grad():
            loader = DataLoader(X_test, batch_size=self.batch_size, shuffle=False,
                                pin_memory=True, num_workers=self.loader_workers)

            if self.task.is_classification:
                total_correct = 0
                total_examples = 0
                for batch in loader:
                    batch = batch.to(self.device)
                    _, pred = model(batch).max(dim=1)
                    total_correct += pred.eq(batch.y).sum().item()
                    total_examples += batch.y.size(0)
                acc = total_correct / total_examples
                return acc
            else:
                total_loss = 0
                total_num_examples = 0
                for batch in loader:
                    batch = batch.to(self.device)
                    out = model(batch)
                    loss = self.criterion(input=out, target=batch.y)
                    total_num_examples += batch.num_graphs
                    total_loss += (loss.item() * batch.num_graphs)
                avg_loss = total_loss / total_num_examples
                return avg_loss
