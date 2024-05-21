import sys

import torch
import numpy as np
from torch.nn import functional as F
import torch.distributions as dist
from scipy.spatial.distance import cdist
import geomloss

def calc_prompt_similarity(examples, model=None, measure='wasserstein', otloss=None):
    # extract the prompt
    p = []
    k = []
    a = []

    # select learned prompt
    pt = int(model.prompt.e_pool_size / model.prompt.n_tasks)
    s = int(model.prompt.task_count * pt)
    f = int((model.prompt.task_count + 1) * pt)
    for name, param in model.prompt.named_parameters():
        if 'e_p' in name:
            p.append(param[:f].detach().clone())

    with torch.no_grad():
        # embed the batch data
        # the most time-consuming step which increases with the buffer size
        data_tensor = model.feat.patch_embed(examples.cuda()).detach()

        o_prompt_tensor = torch.stack(p, dim=0).sum(2) # n_layer, task_per_pool * task_count, p_length, dim
        prompt_tensor = o_prompt_tensor.view(-1, 768)

        if measure == 'wasserstein':
            avg_similarities = compute_wasserstein_distance(data_tensor, prompt_tensor, otloss)
            avg_similarities = avg_similarities / avg_similarities.sum() # weight will degrade
        elif measure == 'kl':
            input_data_avg = torch.mean(data_tensor, dim=1)  # [batch_size, dimension]
            input_probs = F.softmax(input_data_avg, dim=-1)
            prompt_probs = F.softmax(torch.mean(prompt_tensor, dim=0), dim=-1)
            dist_input = dist.Categorical(probs=input_probs)
            dist_prompt = dist.Categorical(probs=prompt_probs.unsqueeze(0).repeat(data_tensor.shape[0], 1))
            avg_similarities = dist.kl_divergence(dist_input, dist_prompt)

        elif measure == 'cosine':
            # normalised along the last dimension
            prompt_norm = F.normalize(prompt_tensor, p=2, dim=-1)
            data_norm = F.normalize(data_tensor, p=2, dim=-1)
            # use einsum to calculate the cosine similarity
            cos_sim = torch.matmul(data_norm, prompt_norm.transpose(0, 1).unsqueeze(0))  # (B, N, pool_size * p_length)
            avg_similarities = cos_sim.mean(dim=[1, 2])  # (B,)

        elif measure == 'all':
            # normalised along the last dimension
            prompt_norm = F.normalize(prompt_tensor, p=2, dim=-1)
            data_norm = F.normalize(data_tensor, p=2, dim=-1)
            # use einsum to calculate the cosine similarity
            cos_sim = torch.matmul(data_norm, prompt_norm.transpose(0, 1).unsqueeze(0))  # (B, N, pool_size * p_length)
            cos_avg_similarities = cos_sim.mean(dim=[1, 2])  # (B,)

            avg_sim = compute_wasserstein_distance(data_tensor, prompt_tensor, otloss)
            wd_avg_similarities = avg_sim / avg_sim.sum() # weight will degrade
            avg_similarities = cos_avg_similarities.cpu() + wd_avg_similarities.cpu()

        elif measure == 'euclidean':
            avg_similarities = []
            for i in range(data_tensor.shape[0]):
                avg_similarities.append(torch.cdist(prompt_tensor, data_tensor[i], p=2).mean())
            avg_similarities = torch.tensor(avg_similarities)

    torch.cuda.empty_cache()
    return avg_similarities

def cost_func(a, b, p=2, metric='cosine'):
    """ a, b in shape: (B, N, D) or (N, D)
    """
    assert type(a)==torch.Tensor and type(b)==torch.Tensor, 'inputs should be torch.Tensor'
    if metric=='euclidean' and p==1:
        return geomloss.utils.distances(a, b)
    elif metric=='euclidean' and p==2:
        return geomloss.utils.squared_distances(a, b)
    else:
        if a.dim() == 3:
            x_norm = a / a.norm(dim=2)[:, :, None]
            y_norm = b / b.norm(dim=2)[:, :, None]
            M = 1 - torch.bmm(x_norm, y_norm.transpose(-1, -2))
        elif a.dim() == 2:
            x_norm = a / a.norm(dim=1)[:, None]
            y_norm = b / b.norm(dim=1)[:, None]
            M = 1 - torch.mm(x_norm, y_norm.transpose(0, 1))
        M = pow(M, p)
        return M
def compute_wasserstein_distance(tensor_a, tensor_b, otloss):

    distances = torch.zeros(tensor_a.size(0))

    for i in range(tensor_a.size(0)):
        distances[i] = otloss(tensor_a[i].unsqueeze(0), tensor_b.unsqueeze(0))
    return distances

class Streamprompt:
    def __init__(self, measure=None):
        self.measure = measure
        if self.measure in ['wasserstein', 'all', 'selfattn']:
            p = 2
            entreg = .1  # entropy regularization factor for Sinkhorn
            metric = 'cosine'
            self.OTLoss = geomloss.SamplesLoss(
                loss='sinkhorn', p=p,
                cost=lambda a, b: cost_func(a, b, p=p, metric=metric),
                blur=entreg ** (1 / p), backend='tensorized')
        else:
            self.OTLoss = None

    def select(self, examples, model, size, sim=None):
        if sim is None:
            avg_similarities = calc_prompt_similarity(examples, model=model, measure=self.measure, otloss=self.OTLoss)
        else:
            avg_similarities = sim
        _, indices = torch.sort(avg_similarities, descending=True)
        quarter_size = size // 2
        retained_indices = indices[len(indices) // 2 - quarter_size:len(indices) // 2 + quarter_size]

        return retained_indices, avg_similarities[retained_indices].tolist()

