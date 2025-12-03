import torch

class SchedulerCallback:
    def __init__(self, optimizer, patience=5, higher_is_better=False, factor=0.1, max_reductions=0):
        self.optimizer = optimizer
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.factor = factor
        self.max_reductions = max_reductions
        self.best_metric = -float('inf') if higher_is_better else float('inf')
        self.wait = 0
        self.reduction_count = 0

    def __call__(self, metric):
        improve = (metric > self.best_metric) if self.higher_is_better else (metric < self.best_metric)
        if improve:
            self.best_metric = metric; self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.reduction_count < self.max_reductions:
                    for g in self.optimizer.param_groups:
                        g['lr'] *= self.factor
                    self.reduction_count += 1
                    self.wait = 0
                else:
                    return True
        return False
