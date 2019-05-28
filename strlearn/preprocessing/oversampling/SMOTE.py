import numpy as np
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from itertools import cycle
from collections import defaultdict

INF = np.inf


def generate_syntetic_sample(base_sample, neighbor, max_rand=1.0, min_rand=0):
    num_attrs = base_sample.shape[0]

    randomly_generated_sample = np.zeros(num_attrs)
    for attr in range(num_attrs):
        dif = neighbor[attr] - base_sample[attr]
        gap = np.random.uniform(low=min_rand, high=max_rand)
        randomly_generated_sample[attr] = base_sample[attr] + gap * dif
    return randomly_generated_sample


def calculate_samples_count_based_on_percentage(minority_class_size, precentage):
    return int(np.ceil(minority_class_size * precentage / 100))


def get_shuffled_indicies(X):
    class_indices = np.arange(len(X))
    np.random.shuffle(class_indices)
    return class_indices


def smote(X, k=3, sampling_percentage=100, dont_restrict=False):
    samples_to_create = calculate_samples_count_based_on_percentage(len(X), sampling_percentage)

    synthetics = []

    nnarray = kneighbors_graph(X, n_neighbors=k)
    used_neighbors = defaultdict(list)

    minority_samples_indices_iter = cycle(get_shuffled_indicies(X))

    for _ in range(samples_to_create):
        idx = next(minority_samples_indices_iter)

        if dont_restrict:
            sample_neighbors = find_neighbor_in_graph(nnarray, idx)
        else:
            sample_neighbors = find_neighbor_in_graph(nnarray, idx, used_neighbors[idx])

        if len(sample_neighbors) != 0:
            randomly_chosen_neighbor_idx = np.random.choice(sample_neighbors)
            used_neighbors[idx].append(randomly_chosen_neighbor_idx)

            randomly_chosen_neighbor_sample = X[randomly_chosen_neighbor_idx]

            synthetics.append(generate_syntetic_sample(X[idx], randomly_chosen_neighbor_sample))

    return np.array(synthetics)


def smote_borderline1(X_minority, X_majority, k1=3, k2=3, sampling_percentage=100, dont_restrict=False):
    samples_to_create = calculate_samples_count_based_on_percentage(len(X_minority), sampling_percentage)
    all_samples = np.concatenate((X_minority, X_majority))

    neighbors = kneighbors_graph(all_samples, n_neighbors=k1)
    danger_samples_indices = find_danger_samples(X_minority, k1, neighbors)

    synthetics = []

    danger_samples_indices_iter = cycle(danger_samples_indices)
    used_neighbors = defaultdict(list)

    minority_neighbors = kneighbors_graph(X_minority, n_neighbors=k2)

    for _ in range(samples_to_create):
        idx = next(danger_samples_indices_iter)
        sample = X_minority[idx]
        if dont_restrict:
            sample_minority_neighbors = find_neighbor_in_graph(minority_neighbors, idx)
        else:
            sample_minority_neighbors = find_neighbor_in_graph(minority_neighbors, idx, used_neighbors[idx])

        if len(sample_minority_neighbors) != 0:
            randomly_chosen_neighbor_idx = np.random.choice(sample_minority_neighbors)
            randomly_chosen_neighbor_sample = all_samples[randomly_chosen_neighbor_idx]

            used_neighbors[idx].append(randomly_chosen_neighbor_idx)

            synthetics.append(generate_syntetic_sample(sample, randomly_chosen_neighbor_sample))

    return np.array(synthetics)


def find_danger_samples(X_minority, k1, neighbors):
    danger_samples_indices = []
    for idx, sample in enumerate(X_minority):
        sample_neighbors = neighbors[idx].nonzero()[1]
        majority_neighbors = [neighbor_idx for neighbor_idx in sample_neighbors if neighbor_idx >= len(X_minority)]
        majority_neighbors_count = len(majority_neighbors)
        if majority_neighbors_count >= len(sample_neighbors) / 2 and majority_neighbors_count != k1:
            danger_samples_indices.append(idx)
    return danger_samples_indices


def find_neighbor_in_graph(graph, index, exclusions=None):
    if exclusions is None:
        exclusions = []

    neighbors = graph[index].nonzero()[1]
    return np.array([neighbor for neighbor in neighbors if neighbor not in exclusions])


def find_neighbors(dataset, sample_idx, k):
    neighbors_graph = kneighbors_graph(dataset, n_neighbors=k)
    return neighbors_graph[sample_idx].nonzero()[1]


def find_minority_neighbors_in_indices(indices, X_minority_length):
    return [neighbor_idx for neighbor_idx in indices if neighbor_idx < X_minority_length]


def smote_borderline2(X_minority, X_majority, k1=3, k2=3, sampling_percentage=100, dont_restrict=False):
    samples_to_create = calculate_samples_count_based_on_percentage(len(X_minority), sampling_percentage)
    all_samples = np.concatenate((X_minority, X_majority))

    neighbors = kneighbors_graph(all_samples, n_neighbors=k1)
    danger_samples_indices = find_danger_samples(X_minority, k1, neighbors)

    synthetics = []

    danger_samples_indices_iter = cycle(danger_samples_indices)
    used_neighbors = defaultdict(list)

    all_neighbors = kneighbors_graph(np.concatenate([X_minority, X_majority]), n_neighbors=k2)

    for _ in range(samples_to_create):
        idx = next(danger_samples_indices_iter)
        sample = X_minority[idx]

        if dont_restrict:
            sample_neighbors = find_neighbor_in_graph(all_neighbors, idx)
        else:
            sample_neighbors = find_neighbor_in_graph(all_neighbors, idx, used_neighbors[idx])

        if len(sample_neighbors) != 0:
            randomly_chosen_neighbor_idx = np.random.choice(sample_neighbors)
            randomly_chosen_neighbor_sample = all_samples[randomly_chosen_neighbor_idx]

            used_neighbors[idx].append(randomly_chosen_neighbor_idx)

            neighbor_is_majority = randomly_chosen_neighbor_idx > len(X_minority)
            if neighbor_is_majority:
                synthetic = generate_syntetic_sample(sample, randomly_chosen_neighbor_sample, 0.5)
            else:
                synthetic = generate_syntetic_sample(sample, randomly_chosen_neighbor_sample)

            synthetics.append(synthetic)

    return np.array(synthetics)


def smote_safe_level(X_minority, X_majority, k=3, sampling_percentage=100, dont_restrict=False):
    samples_to_create = calculate_samples_count_based_on_percentage(len(X_minority), sampling_percentage)

    all_samples = np.concatenate((X_minority, X_majority))

    all_neighbors_graph = kneighbors_graph(all_samples, n_neighbors=k)
    minority_neighbors_graph = kneighbors_graph(X_minority, n_neighbors=k)
    used_neighbors = defaultdict(list)

    minority_samples_iterator = cycle(get_shuffled_indicies(X_minority))

    synthetics = []
    for _ in range(samples_to_create):
        # seed sample
        idx = next(minority_samples_iterator)

        # calculate safe level for seed sample
        all_neighbors_indices = find_neighbor_in_graph(all_neighbors_graph, idx)
        all_neighbors = all_samples[all_neighbors_indices]
        minority_neighbors = [neighbor for neighbor in all_neighbors if neighbor in X_minority]
        safe_level = len(minority_neighbors) / k

        # pick random minorty neighbor
        if dont_restrict:
            minority_neighbors = find_neighbor_in_graph(minority_neighbors_graph, idx)
        else:
            minority_neighbors = find_neighbor_in_graph(minority_neighbors_graph, idx, used_neighbors[idx])

        if len(minority_neighbors) != 0:

            minority_neighbor_idx = np.random.choice(minority_neighbors)
            used_neighbors[idx].append(minority_neighbor_idx)

            # calculate safe level for minority sample
            minority_neighbor_all_neighbors = all_samples[
                find_neighbor_in_graph(all_neighbors_graph, minority_neighbor_idx)]
            minority_neighbor_minority_neighbors = [neigh for neigh in minority_neighbor_all_neighbors if
                                                    neigh in X_minority]
            minority_neighbor_safe_level = len(minority_neighbor_minority_neighbors) / k

            if minority_neighbor_safe_level != 0:
                safe_level_ratio = safe_level / minority_neighbor_safe_level
            else:
                safe_level_ratio = INF

            if not (safe_level_ratio == INF and safe_level == 0):
                if safe_level_ratio == INF and safe_level != 0:
                    generated_sample = generate_syntetic_sample(X_minority[idx], X_minority[minority_neighbor_idx], 0)
                elif safe_level_ratio == 1:
                    generated_sample = generate_syntetic_sample(X_minority[idx], X_minority[minority_neighbor_idx])
                elif safe_level_ratio > 1:
                    generated_sample = generate_syntetic_sample(X_minority[idx], X_minority[minority_neighbor_idx],
                                                                1 / safe_level_ratio)
                elif safe_level_ratio < 1:
                    generated_sample = generate_syntetic_sample(X_minority[idx], X_minority[minority_neighbor_idx], 1,
                                                                1 - safe_level_ratio)
                synthetics.append(generated_sample)
    return np.array(synthetics)
