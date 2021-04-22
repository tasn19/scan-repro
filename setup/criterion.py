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
        #print('NT-Xent loss', loss)
        return loss

# Formula definition in paper
class SCAN_loss(nn.Module):
  def __init__(self, entropy_weight):
    super(SCAN_loss, self).__init__()
    self.softmax = nn.Softmax(dim=1)
    self.entropy_weight = entropy_weight

  def forward(self, anchors, neighbors):
    # clustering fnc terminates in softmax fnc to perform soft assignment over clusters
    batch, numClasses = anchors.size()
    prob_anchor = self.softmax(anchors)
    prob_neighbor = self.softmax(neighbors)

    # dot product
    dp = torch.bmm(prob_anchor.view(batch, 1, numClasses), prob_neighbor.view(batch, numClasses, 1)).squeeze() # returns [1xbatch]
    # sum over all images in batch (sum over all neighbors(log(dot product)))  # THIS PART DFRNT
    s = sum(torch.log(dp))
    # take negative sum spread over dataset (divide by num imgs)
    term1 = -(s/batch)

    # Include entropy term to avoid clustering fnc from from assigning all samples to single cluster
    prob_anchor_mean = torch.mean(prob_anchor, 0) # mean probability of sample being assigned to cluster
    s1 = sum(prob_anchor_mean * torch.log(prob_anchor_mean))
    entropy_term = self.entropy_weight * s1

    # Total loss
    loss = term1 + entropy_term
    return loss

class CE_loss(nn.Module):
  def __init__(self, threshold):
    super(CE_loss, self).__init__()
    self.softmax = nn.Softmax(dim=1)
    self.threshold = threshold

  def forward(self, images, augmented_images):
    images_prob = self.softmax(images)
    batch, cls = images_prob.size()
    maxprob, lbl = torch.max(images_prob, 1)
    mask = maxprob > self.threshold # create mask for probabilities higher than threshold
    # find classified labels with probability greater than threshold
    maskedlbl = torch.masked_select(lbl, mask)

    # Apply weight to cross-entropy loss to compensate for imbalance between confident samples across clusters
    # class weights are inversely proportional to num occurrences in batch after thresholding
    idx, counts = torch.unique(maskedlbl, return_counts = True)
    n = maskedlbl.size(0)
    occurence = counts.int()/n # occurence of label over total labels with prob above threshold
    freq = 1/occurence
    weight = torch.ones(cls).to(device)
    weight[idx] = freq

    b,c = augmented_images.size()
    ce_input = torch.masked_select(augmented_images, mask.view(b, 1)).view(n, c)
    ce_loss = F.cross_entropy(ce_input, maskedlbl, weight=weight, reduction='mean')
    return ce_loss