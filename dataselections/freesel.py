import torch
import random
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from functools import partial
from collections import namedtuple
from kmeans_pytorch import kmeans


def extract_feature_pipeline(x, y, model, size):
    # ============ setting config ... ============
    Args = namedtuple('Args', ['arch', 'threshold', 'centroid_num', 'sample_num', 'kmeans_dist_type'])
    args = Args(arch="vit", threshold=0.5, centroid_num=1, sample_num=size, kmeans_dist_type="euclidean")
    # ============ building network ... ============
    model.cuda()
    model.eval()
    # ============ building batch dataset ... ============
    new_dataset = TensorDataset(x, y)
    data_loader_train = DataLoader(new_dataset, batch_size=x.shape[0], shuffle=False)
    all_features = {}
    # ============ extract features ... ============
    train_features, train_ids = extract_features(model, data_loader_train, args)

    for i in range(len(train_features)):
        all_features[train_ids[i]] = train_features[i]

    return all_features

def filter_features(dense_features, args, attn=None):
    # input: (n, c, k, k)
    # output: list n: [c1, c2, ...,]
    filtered_features = []
    count = 0

    bs = dense_features.shape[0]
    if "vit" not in args.arch:
        dense_features = dense_features.permute(0, 2, 3, 1)
        dense_features = dense_features.reshape(bs, dense_features.shape[1]*dense_features.shape[2], dense_features.shape[3])  # (n, k*k,c )

    dense_features_norm = torch.norm(dense_features, p=2, dim=2)  # (n, k*k)

    if attn is None:
        mask = dense_features_norm > args.threshold # (n, k*k)
    else:
        assert 0 <= args.threshold <= 1
        # attn: (bs, wh)
        attn_sort, idx_sort = torch.sort(attn, dim=1, descending=False)
        attn_cum = torch.cumsum(attn_sort, dim=1)  # (bs, wh)
        mask = attn_cum > (1-args.threshold)
        for b in range(bs):
            mask[b][idx_sort[b]] = mask[b].clone()

    for b in range(bs):
        mask_i = mask[b]  # (k*k, )
        dense_features_i = dense_features[b]  # (k*k, c)
        if torch.sum(mask_i) > 0:
            dense_features_i = dense_features_i[mask_i]
        else:
            max_id = torch.max(dense_features_norm[b], dim=0)[1]
            dense_features_i = dense_features_i[max_id].unsqueeze(0)  # (1, c)

        if args.centroid_num is not None and args.centroid_num < dense_features_i.shape[0]:
            if args.centroid_num > 1:
                cluster_ids_x, cluster_centers = kmeans(
                    X=dense_features_i, num_clusters=args.sample_num, distance=args.kmeans_dist_type, iter_limit=100, device=torch.device('cuda:0')
                )
            else:
                if args.kmeans_dist_type == "cosine":
                    dense_features_i_ = F.normalize(dense_features_i, p=2, dim=1)
                else:
                    dense_features_i_ = dense_features_i
                cluster_centers = torch.mean(dense_features_i_, dim=0, keepdims=True)
            count += cluster_centers.shape[0]
            filtered_features.append(cluster_centers.cuda())
        else:
            filtered_features.append(dense_features_i)
            count += dense_features_i.shape[0]

    return filtered_features, count

@torch.no_grad()
def extract_features(model, data_loader, args):

    train_ids = []
    train_features = []
    feature_num = 0
    for samples, index in data_loader:
        samples = samples.cuda(non_blocking=True)
        if "vit" not in args.arch:
            feats, dense_feats = model(samples)
            dense_feats = dense_feats["layer4"].clone()
            dense_feats, count = filter_features(dense_feats, args)
        else:
            dense_feats = model.get_intermediate_layers(samples, n=2)
            dense_feats = dense_feats[0]
            dense_feats = dense_feats[:, 1:]
            attn = model.get_last_selfattention(samples)  # (bs, nh, wh+1, wh+1)
            attn = torch.mean(attn, dim=1)[:, 0, 1:]  # (bs, wh)
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            dense_feats, count = filter_features(dense_feats, args, attn)

        feature_num += count

        train_features.extend(dense_feats)
        train_ids.extend(index)

    return train_features, train_ids


def merge_features(all_features):
    merged_features = list(all_features.values())
    merged_features = torch.cat(merged_features, dim=0)
    id2idx = {}
    idx = 0
    count = 0
    merged_ids = []
    for id in all_features:
        id2idx[count] = torch.arange(idx, idx+all_features[id].shape[0])
        merged_ids.append(id)
        idx = idx + all_features[id].shape[0]
        count += 1
    return merged_features, merged_ids, id2idx

def prob_seed_dense(all_features, id2idx, sample_num, dist_func, init_ids=[]):
    if len(init_ids) >= sample_num:
        print("Initial samples are enough")
        return init_ids

    feature_num = all_features.shape[0]
    total_num = len(id2idx)
    if total_num <= sample_num:
        print("Not enough features")
        return list(range(total_num))

    idx2id = []
    for id in id2idx:
        idxs = id2idx[id]
        idx2id.extend([id]*idxs.shape[0])
    assert len(idx2id) == feature_num

    if len(init_ids) == 0:
        sample_ids = random.sample(range(total_num), 1)
    else:
        sample_ids = init_ids

    distances = torch.zeros(feature_num).cuda() + 1e20
    # print(torch.max(distances, dim=0)[0])

    for i, init_id in enumerate(sample_ids):
        distances = update_distance_dense(distances, all_features, all_features[id2idx[init_id]], dist_func)
        if i % 100 == 1:
            print(i, torch.max(distances, dim=0)[0], "random")
            print(all_features.shape, all_features[id2idx[init_id]].shape)

    while len(sample_ids) < sample_num:
        prob = distances ** 2 / torch.sum(distances ** 2)
        prob = prob.cpu().numpy()
        new_featid = np.random.choice(distances.shape[0], p=prob)

        # new_featid = torch.max(distances, dim=0)[1]
        new_id = idx2id[new_featid]
        distances = update_distance_dense(distances, all_features, all_features[id2idx[new_id]], dist_func)
        sample_ids.append(new_id)
        if len(sample_ids) % 100 == 1:
            print(len(sample_ids))
            print(len(sample_ids), torch.max(distances, dim=0)[0], "prob")
            print(all_features.shape, all_features[id2idx[new_id]].shape)
    assert len(set(sample_ids)) == sample_num
    return sample_ids

def update_distance_dense(distances, all_features, cfeatures, dist_func):
    # all_features: (n, c)
    # cfeatures: (r, c)
    new_dist = dist_func(all_features, cfeatures)  # (n, r)
    new_dist = torch.min(new_dist, dim=1)[0]  # (n, )
    distances = torch.where(distances < new_dist, distances, new_dist)
    return distances

def get_distance(p1, p2, type, slice=1000):
    if len(p1.shape) > 1:
        if len(p2.shape) == 1:
            # p1 (n, dim)
            # p2 (dim)
            p2 = p2.unsqueeze(0)  # (1, dim)
            if type == "cosine":
                p1 = F.normalize(p1, p=2, dim=1)
                p2 = F.normalize(p2, p=2, dim=1)
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    dist_ = 1 - torch.sum(p1[slice*i:slice*(i+1)]*p2, dim=1)  # (slice, )
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            elif type == "euclidean":
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    dist_ = torch.norm(p1[slice*i:slice*(i+1)]-p2, p=2, dim=1)  # (slice, )
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            else:
                raise NotImplementedError
        else:
            # p1 (n, dim)
            # p2 (m, dim)
            if type == "cosine":
                p1 = p1.unsqueeze(1)  # (n, 1, dim)
                p2 = p2.unsqueeze(0)  # (1, m, dim)
                p2 = F.normalize(p2, p=2, dim=2)
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    p1_slice = F.normalize(p1[slice*i:slice*(i+1)], p=2, dim=2)
                    dist_ = 1 - torch.sum(p1_slice * p2, dim=2)  # (slice, m)
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)

            elif type == "euclidean":
                p1 = p1.unsqueeze(1)  # (n, 1, dim)
                p2 = p2.unsqueeze(0)  # (1, m, dim)
                dist = []
                iter = p1.shape[0] // slice + 1
                for i in range(iter):
                    dist_ = torch.norm(p1[slice*i:slice*(i+1)] - p2, p=2, dim=2)  # (slice, m)
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            else:
                raise NotImplementedError
    else:
        # p1 (dim, )
        # p2 (dim, )
        if type == "cosine":
            dist = 1 - torch.sum(p1 * p2)
        elif type == "euclidean":
            dist = torch.norm(p1 - p2, p=2)
        else:
            raise NotImplementedError
    return dist

def freesel_selection(x, y, model, size):
    train_features = extract_feature_pipeline(x, y, model, size)
    merged_features, merged_ids, id2idx = merge_features(train_features)
    selected_sample = prob_seed_dense(merged_features, id2idx, size,
                                      partial(get_distance, type="cosine"))
    # selected_ids = []
    # for idx in selected_sample:
    #     id = merged_ids[idx]
    #     selected_ids.append(int(id))
    # selected_ids.sort()
    return selected_sample