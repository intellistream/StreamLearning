import numpy as np
import torch

def coreset_selection(x, model, size):
    orig_mode = model.training
    model.eval()
    with torch.no_grad():
        embedding = model(x.cuda(), train=False)
    model.train(orig_mode)

    m = np.shape(embedding)[0]
    min_dist = np.tile(float("inf"), m)
    idxs = []

    for i in range(size):
        idx = min_dist.argmax()
        idxs.append(idx)
        dist_new_ctr = torch.cdist(embedding, embedding[[idx], :])
        for j in range(m):
            min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

    return idxs