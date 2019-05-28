import numpy as np
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder


def split_minority_majority(x, y):
    classes, counts = np.unique(y, return_counts=True)
    sorted_indicies = np.argsort(counts)
    sorted_classes = classes[sorted_indicies]

    minority_class = sorted_classes[0]
    other_classes = sorted_classes[1:]

    minority_class_indicies = np.isin(y, minority_class)
    other_classes_indicies = np.isin(y, other_classes)

    return x[minority_class_indicies], y[minority_class_indicies], x[other_classes_indicies], y[
        other_classes_indicies]


def load_arff(path):
    data = arff.loadarff(path)
    ds, header = data
    attribute_names = ds.dtype.names[0:-1]
    class_name = ds.dtype.names[-1]
    x = ds[list(attribute_names)]
    y = LabelEncoder().fit_transform(ds[class_name])
    return np.array(x.tolist()), y
