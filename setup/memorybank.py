import torch
import numpy as np
import faiss

# paper code
class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n  # length of dataset
        self.dim = dim
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def reset(self):
        self.ptr = 0

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')

    def update(self, features, targets):
        b = features.size(0)

        assert (b + self.ptr <= self.n)

        self.features[self.ptr:self.ptr + b].copy_(features.detach())
        self.targets[self.ptr:self.ptr + b].copy_(targets.detach())
        self.ptr += b

    def mine_nearest_neighbors(self, k, calculate_accuracy=True):
        # use faiss library for efficient knn mining
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        # to use GPU
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, k + 1)  # Sample included in search
        # features are the query_vector (# items in dataset x dimension), so we have queries = num imgs
        # index.search returns the indices: row i contains the IDs of the neighbors of query vector i, sorted by increasing distance
        # and corresponding squared distances
        # IDs are the vector ordinals - the first vector gets 0, the second 1, etc.

        # evaluate
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1, 1), k, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy

        else:
            return indices

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1, -1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, self.C),
                                    yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, (ims, lbls) in enumerate(loader):
        images = ims.cuda(non_blocking=True)
        targets = lbls.cuda(non_blocking=True)
        output = model(images)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))