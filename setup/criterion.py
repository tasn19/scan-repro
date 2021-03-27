import torch
import torch.nn as nn
import torch.nn.functional as F

# loss_i,j = -log( exp(sim(z_i, z_j)/tau) / sum [1] exp(sim(z_i, z_k)/tau) )
# Based on https://www.egnyte.com/blog/2020/07/understanding-simclr-a-framework-for-contrastive-learning/
class SimCLR_loss(nn.Module):
    def __init__(self, batch_size, temp=0.1):
        super().__init__()
        self.batch_size = batch_size  # need?
        self.register_buffer("temperature", torch.tensor(temp))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        print('NT-Xent loss', loss)
        return loss