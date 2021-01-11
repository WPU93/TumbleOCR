import torch
import torch.utils.data
import torchvision
import numpy as np
import math
class recSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None,pow=1):
        self.dataset = dataset
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        weights = [1.0 / (label_to_count[self._get_label(idx)]**pow)
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        print("=> dict of label in sampler:",label_to_count)
    def _get_label(self,idx):
        path = self.dataset.get_path(idx)
        label = "synth" if "synth" in path.lower() else "real"
        return label

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

