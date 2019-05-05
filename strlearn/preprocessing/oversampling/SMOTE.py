import numpy as np
from sklearn.neighbors import kneighbors_graph, NearestNeighbors

def generate_syntetic_sample(base_sample, neighbor, max_rand=1, min_rand=0):
    num_attrs = base_sample.shape[0]

    randomly_generated_sample = np.zeros(num_attrs)
    for attr in range(num_attrs):
        dif = neighbor[attr] - base_sample[attr]
        gap = np.random.uniform(low=min_rand, high=max_rand)
        randomly_generated_sample[attr] = base_sample[attr] + gap * dif
    return randomly_generated_sample


def smote(X, sampling_ratio, k):
    if sampling_ratio < 1:
        samples_to_take = int(np.floor(sampling_ratio * len(X)))
        X_indices = np.arange(len(X))
        sampled_X_indices = np.random.choice(X_indices, samples_to_take, replace=False)
        X = X[sampled_X_indices]
        sampling_ratio = 1

    sampling_iterations = int(np.floor(sampling_ratio))
    numattrs = X.shape[1]
    synthetics = []

    nnarray = kneighbors_graph(X, n_neighbors=k)

    while sampling_ratio != 0:
        for idx, sample in enumerate(X):
            sample_neighbors = nnarray[idx].nonzero()[1]
            randomly_choosen_neighbor_idx = np.random.choice(sample_neighbors)
            randomly_choosen_neighbor = X[randomly_choosen_neighbor_idx]

            synthetics.append(generate_syntetic_sample(sample, randomly_choosen_neighbor))
        sampling_ratio = sampling_ratio - 1

    return np.array(synthetics)


def smote_borderline1(X_minority, X_majority, k, s):
    all_samples = np.concatenate((X_minority, X_majority))

    neighbors = kneighbors_graph(all_samples, n_neighbors=k)

    danger_samples_indices = []
    for idx, sample in enumerate(X_minority):
        sample_neighbors = neighbors[idx].nonzero()[1]
        majority_neighbors = [neighbor_idx for neighbor_idx in sample_neighbors if neighbor_idx >= len(X_minority)]
        #         print(f'{idx} {sample} - {majority_neighbors} {len(majority_neighbors)}')
        majority_neighbors_count = len(majority_neighbors)
        if majority_neighbors_count >= len(sample_neighbors) / 2:
            danger_samples_indices.append(idx)

    synthetics = []

    minority_neighbors = kneighbors_graph(X_minority, n_neighbors=k)
    for iteration in range(s):
        for idx in danger_samples_indices:
            sample = X_minority[idx]
            sample_minority_neighbors = minority_neighbors[idx].nonzero()[1]
            randomly_choosen_neighbor_idx = np.random.choice(sample_minority_neighbors)
            randomly_choosen_neighbor = all_samples[randomly_choosen_neighbor_idx]

            synthetics.append(generate_syntetic_sample(sample, randomly_choosen_neighbor))

    return np.array(synthetics)


def find_neighbors(dataset, sample_idx, k):
    neighbors_graph = kneighbors_graph(dataset, n_neighbors=k)
    return neighbors_graph[sample_idx].nonzero()[1]


def find_minority_neighbors_in_indices(indices, X_minority_length):
    return [neighbor_idx for neighbor_idx in indices if neighbor_idx < X_minority_length]


def smote_borderline2(X_minority, X_majority, k, s):
    all_samples = np.concatenate((X_minority, X_majority))

    neighbors = kneighbors_graph(all_samples, n_neighbors=k)

    danger_samples_indices = []
    for idx, sample in enumerate(X_minority):
        sample_neighbors = neighbors[idx].nonzero()[1]
        majority_neighbors = [neighbor_idx for neighbor_idx in sample_neighbors if neighbor_idx >= len(X_minority)]
        majority_neighbors_count = len(majority_neighbors)
        if majority_neighbors_count >= len(sample_neighbors) / 2:
            danger_samples_indices.append(idx)

    synthetics = []

    minority_neighbors = kneighbors_graph(X_minority, n_neighbors=k)
    majority_neighbors_finder = NearestNeighbors(n_neighbors=1)
    majority_neighbors_finder.fit(X_majority)
    for iteration in range(s):
        for idx in danger_samples_indices:
            sample = X_minority[idx]
            sample_minority_neighbors = minority_neighbors[idx].nonzero()[1]
            randomly_choosen_neighbor_idx = np.random.choice(sample_minority_neighbors)
            randomly_choosen_neighbor = all_samples[randomly_choosen_neighbor_idx]

            sample_majority_closest_neighbor_idx = \
                majority_neighbors_finder.kneighbors([sample], return_distance=False)[0][0]
            sample_majority_closest_neighbor = X_majority[sample_majority_closest_neighbor_idx]

            synthetics.append(generate_syntetic_sample(sample, sample_majority_closest_neighbor, 0.5))
            synthetics.append(generate_syntetic_sample(sample, randomly_choosen_neighbor))

    return np.array(synthetics)


def smote_safe_level(X_minority, X_majority, k, s):
    INF = np.inf

    all_samples = np.concatenate((X_minority, X_majority))

    neighbors_graph = kneighbors_graph(all_samples, n_neighbors=k)
    minority_neighbors_graph = kneighbors_graph(X_minority, n_neighbors=k)

    synthetics = []
    for idx, sample in enumerate(X_minority):
        minority_neighbors_indices = find_neighbors(X_minority, idx, k)
        all_neighbors_indices = find_neighbors(all_samples, idx, k)

        minority_neighbors_of_all_neighbors = find_minority_neighbors_in_indices(all_neighbors_indices, len(X_minority))
        sample_safe_level = len(minority_neighbors_of_all_neighbors)

        random_minority_neighbor_idx = np.random.choice(minority_neighbors_indices)

        random_minority_neighbor_all_neighbors_indices = find_neighbors(all_samples, random_minority_neighbor_idx, k)
        random_minority_neighbor_safe_level = len(
            find_minority_neighbors_in_indices(random_minority_neighbor_all_neighbors_indices, len(X_minority)))

        if random_minority_neighbor_safe_level != 0:
            safe_level_ratio = sample_safe_level / random_minority_neighbor_safe_level
        else:
            safe_level_ratio = INF

        if not (safe_level_ratio == INF and sample_safe_level == 0):
            if safe_level_ratio == INF and sample_safe_level != 0:
                generated_sample = generate_syntetic_sample(sample, X_minority[random_minority_neighbor_idx], 0)
            elif safe_level_ratio == 1:
                generated_sample = generate_syntetic_sample(sample, X_minority[random_minority_neighbor_idx])
            elif safe_level_ratio > 1:
                generated_sample = generate_syntetic_sample(sample, X_minority[random_minority_neighbor_idx],
                                                            1 / safe_level_ratio)
            elif safe_level_ratio < 1:
                generated_sample = generate_syntetic_sample(sample, X_minority[random_minority_neighbor_idx], 1,
                                                            1 - safe_level_ratio)
            synthetics.append(generated_sample)
    return np.array(synthetics)
