import numpy as np
from NNets import FFNetAsync

class FFNetEnsemble(object):
    '''An interface that abstracts the talking with an asynchronous Agent.'''
    def __init__(self, model, N=5):
        self.nets = [FFNetAsync(model) for i in range(N)]

    def predict_on_batch(self, batch):
        for net in self.nets:
            net.predict_on_batch(batch)
        stack = np.stack([net.collect() for net in self.nets])
        return np.mean(stack, axis=0)

    def train_on_batch(self, batch, labels):
        for net in self.nets:
            net.train_on_batch(batch, labels)
        return 0