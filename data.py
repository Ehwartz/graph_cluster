import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import torchvision
import random


class TempDataset(Dataset):
    def __init__(self, data, targets):
        super(TempDataset, self).__init__()
        self.data = data
        self.targets = targets

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)


class MNIST(Dataset):
    def __init__(self, root, flatten, train=True):
        super(MNIST, self).__init__()

        mds = torchvision.datasets.MNIST(root=root, download=True, train=train)

        if flatten:
            self.data = mds.data.view(-1, 784) / 255
        else:
            self.data = mds.data / 255
        self.n = self.data.size(0)
        self.targets = torch.zeros(size=[self.data.size(0), 10])
        self.targets[torch.arange(self.n), mds.targets] = 1
        self.labels = mds.targets.clone().detach()
        del mds

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)

    def filter(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return TempDataset(data=self.data[indices], targets=self.targets[indices])

    def filter_indices(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return indices


class CIFAR(Dataset):
    def __init__(self, root, train=True):
        super(CIFAR, self).__init__()

        cds = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.data = (torch.from_numpy(cds.data) / 255).permute(0, 3, 1, 2)
        self.n = self.data.size(0)
        self.targets = torch.zeros(size=[self.data.size(0), 10])
        self.targets[torch.arange(self.n), cds.targets] = 1
        self.labels = torch.tensor(cds.targets)

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)

    def filter(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return TempDataset(data=self.data[indices], targets=self.targets[indices])


class MixedDataset(Dataset):
    def __init__(self, dataset, mixed_labels):
        super(MixedDataset, self).__init__()
        self.data = dataset.data
        self.n = self.data.size(0)
        dim = len(mixed_labels)
        self.true_targets = dataset.targets
        self.targets = torch.zeros(size=[self.data.size(0), dim])
        self.labels = torch.argmax(dataset.targets, dim=-1)

        for d in range(dim):
            indices = []
            for label in mixed_labels[d]:
                indices.append(torch.where(self.labels == label)[0])
            indices = torch.concat(indices, dim=0)
            self.targets[indices, d] = 1

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)

    def filter(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return TempDataset(data=self.data[indices], targets=self.targets[indices])


class PartialDataset(Dataset):
    def __init__(self, dataset, partial_rate, targets_root=None):
        super(PartialDataset, self).__init__()
        self.data = dataset.data
        self.labels = dataset
        self.n = self.data.size(0)
        self.labels = torch.argmax(dataset.targets, dim=-1)
        if targets_root:
            self.targets = torch.load(targets_root)
        else:
            self.targets = dataset.targets
            partial_indices = torch.where(torch.rand(size=self.targets.size()) < partial_rate)
            self.targets[partial_indices] = 1

    def __getitem__(self, item):
        return self.data[item], self.targets[item], item

    def __len__(self):
        return self.data.size(0)

    def filter(self, labels: list):
        ls = []
        for label in labels:
            ls.append(torch.where(self.labels == label)[0])
        indices = torch.concat(ls, dim=0)
        return TempDataset(data=self.data[indices], targets=self.targets[indices])


def generate_unlearn_set(dataset, n, ul_labels, rm_labels, model, remain=False):
    uld = dataset.filter(ul_labels)
    rmd = dataset.filter(rm_labels)
    uln = len(uld)
    indices = list(range(uln))
    random.shuffle(indices)
    ulset = TempDataset(data=uld.data[indices[:n]], targets=uld.targets[indices[:n]])
    if remain:
        rmn = len(rmd)
        indices = list(range(rmn))
        random.shuffle(indices)
        rmset = TempDataset(data=rmd.data[indices], targets=model(rmd.data[indices]).detach())
        return ulset, rmset
    else:
        return ulset

if __name__ == '__main__':
    from torch.utils.data import RandomSampler, SubsetRandomSampler

    # dataset = MNIST(root='./datasets', train=True, flatten=True)
    # n = len(dataset)
    # r = torch.randint(0, n, size=[4])
    # print(r)
    # print(dataset[r])
    print(list(range(10))[:5])
