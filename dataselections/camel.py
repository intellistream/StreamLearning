import torch
import heapq
import torchvision.models as models
import torch.nn.functional as F
from collections import defaultdict

def delta_cover(examples, labels, subset_examples, subset_labels, subset_indices, remaining_indices, d_max):
    """
    Calculate the δ-cover of a dataset.
    :param examples: A list of data points (as tensors).
    :param labels: A list of class labels corresponding to the data points.
    :param subset_indices: The indices of the examples that form the subset S.
    :param delta: The threshold distance value for the δ-cover.
    :return: The δ-cover value and whether the subset S is a δ-cover of dataset B.
    """
    sum_min_distances = 0
    valid_points_count = 0

    unique_labels = labels.unique()

    delta_gain = defaultdict(list)
    for y in unique_labels:
        # Mask to select only examples of class y
        class_mask = labels == y
        class_examples = examples[class_mask]

        # Mask to select only subset examples of class y
        class_subset_mask = subset_labels == y
        class_subset = subset_examples[class_subset_mask]

        subset_with_p = subset_indices + remaining_indices
        class_subset_indices = [subset_with_p[i] for i in range(len(subset_indices))
                                if class_subset_mask[i]]
        class_remaining_indices = [subset_with_p[i] for i in range(len(subset_indices), len(subset_with_p))
                                   if class_subset_mask[i]]

        # Skip the class if there are no representatives in the subset
        if class_subset.nelement() == 0:
            continue

        # Compute all pairwise distances (broadcasting)
        class_examples = class_examples.view(class_examples.size(0), -1)
        class_subset = class_subset.view(class_subset.size(0), -1)
        # print(class_examples.shape, class_subset.shape)

        distances = torch.cdist(class_examples.float(), class_subset.float(), p=2)
        # print(f'distances.shape {distances.shape}')
        # Find the minimum distance for each example in the class
        min_distances = None
        if len(class_subset_indices) > 0:
            min_distances, _ = torch.min(distances[:, :len(class_subset_indices)], dim=1)

        if len(class_remaining_indices) > 0:
            for i in range(len(class_remaining_indices)):
                r_distances = distances[(len(class_subset_indices)+i)]
                if min_distances is None:
                    min_distances = torch.sum(r_distances)
                else:
                    # print(len(class_examples), min_distances)
                    # print(f'Compare distances shape: {min_distances.shape}, {r_distances.shape}')
                    min_distances = torch.sum(torch.min(min_distances, r_distances))
                delta_gain[class_remaining_indices[i]].append(min_distances)

        # Sum these minimum distances and count the valid points
        sum_min_distances += torch.sum(min_distances)
        valid_points_count += class_examples.size(0)

    delta_gain = dict(delta_gain)
    similarity_gains = {}
    for index, tensor_list in delta_gain.items():
        similarity_gains[index] = d_max - sum(tensor_list) / len(examples)

    return similarity_gains

def calculate_weights(examples, labels, subset):
    """
    Calculate the weights w_j for each element in the subset S.

    :param examples The input dataset as a list of inputs xi.
    :param labels The label dataset as a list of labels yi.
    :param subset The indices of the subset S.

    :returns: The weights w_j for each element in S.
    """
    # Initialize the weights for each element in S to zero
    weights = [0] * len(subset)

    # Create a mapping from labels to their corresponding indices in B
    label_to_indices = {}
    for i, y in enumerate(labels):
        key_y = y.cpu().item()
        if key_y not in label_to_indices:
            label_to_indices[key_y] = []
        label_to_indices[key_y].append(i)
    # print(label_to_indices)

    # Iterate over each element in the subset S
    for j_index, j in enumerate(subset):
        yj = labels[j].cpu().item()
        # Consider only elements with the same label as the current element from S
        for i in label_to_indices[yj]:
            xi, yi = examples[i], labels[i]
            # Find the closest point in S to xi
            closest_index_in_s = min((index for index in subset if labels[index] == yi),
                                     key=lambda index: torch.norm(examples[index] - xi), default=None)
            # If the closest point in S to xi is the current point from S, increment its weight
            if closest_index_in_s == j:
                weights[j_index] += 1

    return weights

def similarity_function(examples):
    """
    Calculate the similarity function F_B(S) for the subset S, optimized for large-scale data.

    Parameters:
    examples (torch.Tensor): The dataset B as a tensor of inputs xi.
    delta_S (float): The δ-cover of S, previously calculated.

    Returns:
    float: The value of the similarity function F_B(S).
    """
    # Compute all pairwise distances. Note that for very large datasets, this needs to be batched
    # to avoid out-of-memory errors. We assume examples is a 2D tensor where the rows are examples.
    examples = examples.view(examples.size(0), -1)
    pairwise_distances = torch.cdist(examples.float(), examples.float(), p=2)

    # Since the diagonal of the distance matrix is zero, we fill it with negative infinity to ensure
    # they are not considered when taking the max.
    pairwise_distances.fill_diagonal_(-float('inf'))

    # Get the maximum value from the pairwise distances matrix
    d_max = torch.max(pairwise_distances).item()

    # Calculate the similarity function value
    return d_max


def camel_selection(examples, labels, model, size, desc='None'):
    """
    Select a coreset S from dataset B using a greedy algorithm based on the similarity function F_B.

    :param examples The dataset B as a list of inputs xi.
    :param labels The dataset B as a list of labels yi.
    :param model The model for extracting features.
    :param size The size of the coreset to select.
    :param desc The description of the current process.

    :returns: The subset S of indices and their corresponding weights.
    """
    subset = []  # Initialize the subset S
    weights = []  # Initialize the weights vector w
    # batch_select_num = examples.shape[0] // 50
    batch_select_num = 1
    if "Buffer" in desc:
        # batch_select_num = 1
        batch_select_num = examples.shape[0] // 1

    d_max = similarity_function(examples)
    remaining_indices = [i for i in range(len(examples))]
    # updated_examples = deepcopy(examples)
    if size == examples.shape[0]:
        return remaining_indices, []
    orig_mode = model.training
    model.eval()
    with torch.no_grad():
        features, _, _ = model(examples.float().cuda(), train=False)
        features = F.normalize(features, p=2, dim=-1)
    model.train(orig_mode)

    while len(subset) < size:
        # Find a sample p with the largest similarity gain
        similarity_gains = delta_cover(examples=features,
                                       labels=labels,
                                       subset_examples=features[subset + remaining_indices],
                                       subset_labels=labels[subset + remaining_indices],
                                       subset_indices=subset,
                                       remaining_indices=remaining_indices,
                                       d_max=d_max)
        k_largest = heapq.nlargest(batch_select_num, similarity_gains, key=similarity_gains.get)
        for p in k_largest:
            subset.append(p)
            remaining_indices.remove(p)

    # Calculate the sample weights
    # weights = calculate_weights(examples, subset)
    return subset
