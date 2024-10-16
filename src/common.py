from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


def split_into_windows(ts: np.ndarray, wsize: int, stride: int = 1) -> np.ndarray:
    """Splits a time series into overlapping or non-overlapping windows of a specified size.

    Parameters:
        ts (np.ndarray): The input time series data.
        wsize (int): The size of each window.
        stride (int, optional): The stride or step size for window sliding.

    Returns:
        np.ndarray: An array containing the split windows of the time series data.

    Raises:
        AssertionError: If window size or stride is non-positive.
    """
    assert 0 < wsize and 0 < stride, 'Window size and stride must be positive'

    return np.stack([ts[i-wsize:i] for i in range(wsize, len(ts), stride)])


def extract_segments(time_series: np.ndarray, change_points: np.ndarray) -> List[np.ndarray]:
    """
    Extracts segments from a time series based on given change points.

    Parameters:
        time_series (np.ndarray): 1-D numpy array representing the time series data.
        change_points (np.ndarray): 1-D numpy array containing the indices where the time series data changes.

    Returns:
        List[np.ndarray]: A list of numpy arrays, each representing a segment of the time series data between the change points.
    """
    segments: List[np.ndarray] = []

    change_points = [0] + list(change_points) + [len(time_series)]

    #  Extract segments based on change points
    for left, right in zip(change_points, change_points[1:]):
        segment = time_series[left: right]
        segments.append(segment)

    return segments


# TODO: Write documentation for this
def evaluate(segment_features: List[np.ndarray], agg = np.mean) -> Tuple[np.ndarray, np.ndarray]:
    num_segments = len(segment_features)

    similarity = np.zeros((num_segments, num_segments))
    distance = np.zeros((num_segments, num_segments))

    for i, features in enumerate(segment_features):
        for j, other_features in enumerate(segment_features):
            if i == j:
                filter = np.eye(len(features), len(other_features), dtype=np.bool_) == False
            else:
                filter = np.ones((len(features), len(other_features)), dtype=np.bool_)

            similarity[i, j] = agg(cosine_similarity(features, other_features)[filter])
            distance[i, j] = agg(euclidean_distances(features, other_features)[filter])

    return similarity, distance


def calculate_intra(matrix: np.ndarray) -> float:
    """
    Calculate the average of the diagonal elements of a square matrix.

    Args:
    matrix (numpy.ndarray): A square matrix represented as a NumPy array.

    Returns:
    float: The average of the diagonal elements.
    """
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1], 'The input should be a square matrix'

    diagonal_elements = np.diag(matrix)
    return diagonal_elements.mean()


# TODO: Write documentation for this
def calculate_inter(matrix: np.ndarray) -> float:
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1], 'The input should be a square matrix'

    elements = np.diag(matrix, k=1)
    return elements.mean()

