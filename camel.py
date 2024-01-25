import numpy as np
from typing import List, Tuple

class Camel:
    def __init__(self, B: List[Tuple[np.ndarray, any]], ratio: float, budge_size: int):
        self.ratio = ratio
        self.buffer = []  # Initialize an empty buffer
        self.budge_size = budge_size  # Set the budget size

    def delta_cover(self, B: List[Tuple[np.ndarray, any]], S: List[int], delta: float) -> float:
        """
        Calculate the δ-cover of a dataset.

        Parameters:
        B (List[Tuple[np.ndarray, any]]): The dataset B as a list of tuples (xi, yi).
        S (List[int]): The indices of the subset S.
        delta (float): The δ value for the δ-cover.

        Returns:
        float: The δ-cover of the subset S.
        """
        n = len(B)
        sum_min_distances = 0
        for y in range(n):
            xi, _ = B[y]
            min_distance = min([np.linalg.norm(xi - B[j][0]) for j in S])
            sum_min_distances += min_distance

        return sum_min_distances / n

    def calculate_weights(self, B: List[Tuple[np.ndarray, any]], S: List[int]) -> List[int]:
        """
        Calculate the weights w_j for each element in the subset S.

        Parameters:
        B (List[Tuple[np.ndarray, any]]): The dataset B as a list of tuples (xi, yi).
        S (List[int]): The indices of the subset S.

        Returns:
        List[int]: The weights w_j for each element in S.
        """
        # Initialize the weights for each element in S to zero
        weights = [0] * len(S)

        # Create a mapping from labels to their corresponding indices in B
        label_to_indices = {}
        for i, (_, y) in enumerate(B):
            if y not in label_to_indices:
                label_to_indices[y] = []
            label_to_indices[y].append(i)

        # Iterate over each element in the subset S
        for j_index, j in enumerate(S):
            _, yj = B[j]
            # Consider only elements with the same label as the current element from S
            for i in label_to_indices[yj]:
                xi, yi = B[i]
                # Find the closest point in S to xi
                closest_index_in_s = min((index for index in S if B[index][1] == yi),
                                         key=lambda index: np.linalg.norm(B[index][0] - xi), default=None)
                # If the closest point in S to xi is the current point from S, increment its weight
                if closest_index_in_s == j:
                    weights[j_index] += 1

        return weights

    def similarity_function(self, B: List[Tuple[np.ndarray, any]], S: List[int], delta_S: float) -> float:
        """
        Calculate the similarity function F_B(S) for the subset S.

        Parameters:
        B (List[Tuple[np.ndarray, any]]): The dataset B as a list of tuples (xi, yi).
        S (List[int]): The indices of the subset S.
        delta_S (float): The δ-cover of S, previously calculated.

        Returns:
        float: The value of the similarity function F_B(S).
        """
        # Calculate d_max as the maximum distance between all pairs in B
        d_max = max(np.linalg.norm(B[i][0] - self.B[j][0]) for i in range(len(B)) for j in range(len(B)) if i != j)

        # Calculate the similarity function value
        return d_max - delta_S

    def coreset_selection(self, B: List[Tuple[np.ndarray, any]], m: int) -> Tuple[List[int], List[int]]:
        """
        Select a coreset S from dataset B using a greedy algorithm based on the similarity function F_B.

        Parameters:
        B (List[Tuple[np.ndarray, any]]): The dataset B as a list of tuples (xi, yi).
        m (int): The size of the coreset to select.

        Returns:
        Tuple[List[int], List[int]]: The subset S of indices and their corresponding weights.
        """
        S = []  # Initialize the subset S
        weights = []  # Initialize the weights vector w

        while len(S) < m:
            # Find a sample p with the largest similarity gain
            similarity_gains = {}
            for i, _ in enumerate(B):
                if i not in S:
                    S_with_p = S + [i]
                    delta_S_with_p = delta_cover(B, S_with_p, delta=1.0)
                    similarity_gains[i] = similarity_function(B, S_with_p, delta_S_with_p)
            p = max(similarity_gains, key=similarity_gains.get)
            # Add the sample p to coreset
            S.append(B[p])

        # Calculate the sample weights
        weights = calculate_weights(B, S)

        return S, weights

    def buffer_merge_reduce(self, new_batch: List[Tuple[np.ndarray, any]]) -> None:
        """
        Merge the new batch with the existing buffer and then reduce if needed.
        The buffer and dataset B are updated in place.

        Parameters:
        new_batch (List[Tuple[np.ndarray, any]]): The new batch of data points.
        """
        # Merge step
        self.buffer.extend(new_batch)

        # Reduce step
        if len(self.buffer) > self.budget_size:
            # Select a new coreset from the combined buffer
            new_coreset, _ = self.coreset_selection(self.buffer, self.budget_size)
            self.buffer = new_coreset  # Update the buffer with the new coreset indices

# Example use:
# B = [(np.random.rand(3), i) for i in range(10)]  # Assuming a dataset B is given as before
# camel = Camel(B)
# coreset, weights = camel.coreset_selection(5)  # Select a coreset of size 5
# coreset, weights
