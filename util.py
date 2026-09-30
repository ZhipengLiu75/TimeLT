import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()

    def compute_centroids(self, features, labels,num_classes):
        centroids = []
        stds = []
        for c in range(num_classes):
            class_feats = features[labels == c]
            centroid = class_feats.mean(dim=0)
            std = class_feats.std(dim=0, unbiased=False)
            centroids.append(centroid)
            stds.append(std)
        return torch.stack(centroids), torch.stack(stds) 



    def compute_class_boundaries(self, centroids, stds,num_classes):
        boundaries = []
        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    boundary = centroids[i]
                else:
                    sigma_i = stds[i].norm()
                    sigma_j = stds[j].norm()
                    alpha = sigma_j / (sigma_i + sigma_j + 1e-8)
                    boundary = (1 - alpha) * centroids[i] + alpha * centroids[j]
                boundaries.append(boundary)
        return torch.stack(boundaries)

    def forward(self, features, labels):
        unique_classes = torch.unique(labels)
        class_mapping = {old.item(): new for new, old in enumerate(unique_classes)}
        labels = torch.tensor([class_mapping[l.item()] for l in labels])
        num_classes = torch.max(labels).item() + 1

        features = F.normalize(features, dim=1)
        centroids, STD = self.compute_centroids(features, labels,num_classes)
        centroids = F.normalize(centroids, dim=1)

        boundaries = self.compute_class_boundaries(centroids, STD,num_classes).reshape(
            num_classes, num_classes, -1
        )
        boundaries = F.normalize(boundaries, dim=2)

        feat_c = centroids[labels]
        sim_close = F.cosine_similarity(features, feat_c, dim=1)

        loss_to = 1 - sim_close.mean()

        loss_boundary = []
        loss_from = []
        for i in range(features.size(0)):
            c = labels[i].item()

            boundary_others = torch.cat([
                boundaries[c, :c],
                boundaries[c, c + 1:]
            ], dim=0)
            sims_boundary = F.cosine_similarity(features[i].unsqueeze(0), boundary_others)
            loss_margin = (sims_boundary + 1).mean()
            loss_boundary.append(loss_margin)

            other_centroids = torch.cat([
                centroids[:c],
                centroids[c + 1:]
            ], dim=0)
            sims_centroids = F.cosine_similarity(features[i].unsqueeze(0), other_centroids)
            loss_from.append((sims_centroids + 1))

        loss_from=torch.mean(torch.cat(loss_from,dim=-1))
        loss_boundary=torch.mean(torch.stack(loss_boundary))
        return loss_boundary,(loss_to + loss_from)

