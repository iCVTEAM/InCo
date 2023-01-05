import torch
import torch.nn as nn
import torch.distributed as dist
from .gather import GatherLayer


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        #self.device = device
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        reordered_z = torch.cat((z.narrow(0, N//2, N//2),\
                z.narrow(0, 0, N//2)), 0)
        sim = self.similarity_f(z.unsqueeze(1), reordered_z.unsqueeze(0)) / self.temperature
        #print(sim.shape, reordered_z.shape)
        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)
        #print(sim.shape, sim_i_j.shape, sim_j_i.shape)
        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.cuda()).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        #print(sim.shape,self.mask.shape, positive_samples.shape, negative_samples.shape, labels.shape, logits.shape)
        #torch.Size([128, 128]) torch.Size([128, 128]) torch.Size([128, 1]) torch.Size([128, 126]) torch.Size([128]) torch.Size([128, 127])
        #print(labels.shape, logits.shape) # torch.Size([128]) torch.Size([128, 127])])])
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
