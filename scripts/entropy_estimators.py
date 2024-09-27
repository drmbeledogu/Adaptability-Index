# Adapted from https://github.com/gregversteeg/NPEET

import numpy as np
import numpy.linalg as la
from numpy import log
from scipy.special import digamma, gamma
from sklearn.neighbors import BallTree, KDTree
from collections import Counter

# CONTINUOUS ESTIMATORS
def entropy(x: list, k: int=3, base: float=2, metric:str="chebyshev") -> float:
    """
    The classic K-L k-nearest neighbor continuous entropy estimator
    x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples

    Args:
        x (list): list of continously valued vectors
        k (int, optional): Number of nearest neaighbors. Defaults to 3.
        base (float, optional): Base units. Defaults to 2.
        metric (str, optional): Distance metric. Defaults to "chebyshev".

    Returns:
        float: Entropy
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = build_tree(x, metric=metric)
    nn = query_neighbors(tree, x, k)
    if metric != "chebyshev":
        unit_ball_volume = (np.pi**(n_features/2))/gamma(1+n_features/2)
    else:
        unit_ball_volume = 2**n_features
    const = digamma(n_elements) - digamma(k) + log(unit_ball_volume)
    return (const + n_features * np.log(nn).mean()) / log(base)

def centropy(x: list, y: list, k: int=3, base: float=2, metric: str="chebyshev") -> float:
    """
    The classic K-L k-nearest neighbor continuous entropy estimator for the
    entropy of X conditioned on Y

    Args:
        x (list): list of continously valued vectors to be conditioned on another variable
        y (list): list of continously valued vectors that are doing the conditioning
        k (int, optional): Number of nearest neaighbors. Defaults to 3.
        base (float, optional): Base units. Defaults to 2.
        metric (str, optional): Distance metric. Defaults to "chebyshev".

    Returns:
        float: Conditional entropy
    """
    xy = np.c_[x, y]
    entropy_union_xy = entropy(xy, k=k, base=base, metric=metric)
    entropy_y = entropy(y, k=k, base=base, metric=metric)
    return entropy_union_xy - entropy_y

def mi(x: list, y: list, z: list|None=None, k: int=3, base: float=2, alpha: float=0) -> float:
    """
    Mutual information of x and y (conditioned on z if z is not None)
    x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples

    Args:
        x (list): List of continously valued vectors
        y (list): List of continously valued vectors
        z (list | None, optional): List of continously valued vectors to do the conditioning. Defaults to None.
        k (int, optional): Nearest neighbors. Defaults to 3.
        base (float, optional): Base units. Defaults to 2.
        alpha (float, optional): LNC correction term. Defaults to 0.

    Returns:
        float: Mutual information
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = (
            avgdigamma(x, dvec),
            avgdigamma(y, dvec),
            digamma(k),
            digamma(len(x)),
        )
        if alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = (
            avgdigamma(xz, dvec),
            avgdigamma(yz, dvec),
            avgdigamma(z, dvec),
            digamma(k),
        )
    return (-a - b + c + d) / log(base)

def lnc_correction(tree: KDTree | BallTree, points: np.ndarray, k: int, alpha: float) -> float:
    """
    LNC correction for mutual information between two continuous variables

    Args:
        tree (KDTree | BallTree): Nearest neighbor tree for joint space
        points (np.ndarray): Points in joint space
        k (int): Number of nearest neighbors
        alpha (float): Corrrection effect

    Returns:
        float: Correction term
    """
    e = 0
    n_sample = points.shape[0]
    for point in points:
        # Find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k + 1, return_distance=False)[0]
        knn_points = points[knn]
        # Substract mean of k-nearest neighbor points
        knn_points = knn_points - knn_points[0]
        # Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
        covr = knn_points.T @ knn_points / k
        _, v = la.eig(covr)
        # Calculate PCA-bounding box using eigen vectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        # Calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

        # Perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect) / n_sample
    return e


# DISCRETE ESTIMATORS
def entropyd(x: list, base: float=2) -> float:
    """
    Discrete entropy estimator
    x is a list of samples

    Args:
        x (list): List of samples
        base (float, optional): Base unit. Defaults to 2.

    Returns:
        float: Entropy
    """
    counter = Counter(x)
    unique_counts = list(counter.values())
    probs = [count / len(x) for count in unique_counts]
    entropy = -np.sum([prob * np.log(prob) for prob in probs]) / log(base)
    return entropy

def midd(x: list, y: list, base: float=2) -> float:
    """
    Computes the mutual information between two discrete variables

    Args:
        x (list): List of samples
        y (list): List of samples
        base (float, optional): Base unit. Defaults to 2.

    Returns:
        float: Mutual Information
    """
    assert len(x) == len(y), "Arrays should have same length"
    return entropyd(x, base) - centropyd(x, y, base)

def cmidd(x: list, y: list, z: list, base: float=2) -> float:
    """
    Computes the mutual information between two discrete variables, x and y
    conditioned on a third discrete variable, z

    Args:
        x (list): List of samples
        y (list): List of samples
        z (list): List of samples
        base (float, optional): Base unit. Defaults to 2.

    Returns:
        float: Conditional mutual information
    """
    assert len(x) == len(y) == len(z), "Arrays should have same length"
    xz = np.c_[x, z]
    yz = np.c_[y, z]
    xyz = np.c_[x, y, z]
    return (
        entropyd(xz, base)
        + entropyd(yz, base)
        - entropyd(xyz, base)
        - entropyd(z, base)
    )

def centropyd(x: list, y: list, base: float=2) -> float:
    """
    Computes entropy for discrete variable x, conditioned on discrete variable y

    Args:
        x (list): List of samples to compute entropy
        y (list): List of samples to use for conditioning
        base (float, optional): base unit. Defaults to 2.

    Returns:
        float: Conditional entropy
    """
    xy = list(zip(x, y))
    return entropyd(xy, base) - entropyd(y, base)

# MIXED ESTIMATORS
def micd(x: list, y: list, k: int=3, base: float=2, metric: str="chebyshev") -> float:
    """
    Computes the mutual information between a continous variable, x,
    and a discrete variable, y. Methodology implements work here
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357

    Args:
        x (list): List of continuously valued vectors
        y (list): List of samples
        k (int, optional): Nearest neighbors. Defaults to 3.
        base (float, optional): Base unit. Defaults to 2.
        metric (str, optional): Metric for distance calculation. Defaults to "chebyshev".

    Returns:
        float: Mutual information
    """
    x = np.array(x)
    counter = Counter(y)
    unique_values = list(counter.keys())
    unique_counts = list(counter.values())

    if metric != "chebyshev":
        metric = "euclidean"

    #Calculate m's
    total_tree = build_tree(x)
    ms = []
    for unique_val in unique_values:
        indices = [i for i, y in enumerate(y) if y == unique_val]
        filtered_x = np.array([x[i] for i in indices])
        filtered_tree = build_tree(filtered_x, metric=metric)
        nn = query_neighbors(filtered_tree, filtered_x, k)
        m = count_neighbors(total_tree, filtered_x, nn)
        ms.extend(m)

    #Calculate digamma pieces
    psi_N = digamma(len(x))
    psi_k = digamma(k)
    avg_psi_Nx = np.dot(unique_counts, digamma(unique_counts))/sum(unique_counts)
    avg_psi_m = np.mean(digamma(ms))

    return (psi_N - avg_psi_Nx + psi_k - avg_psi_m)/log(base)

def cmicd(x: list, y: list, z: list, k: int=3, base: float=2, metric: str="euclidean") -> float:
    """
    Computes the mutual information between a continous variable, x,
    a discrete variable, y, coinditioned on a discrete variable, z.

    Args:
        x (list): List of continously valued vectors
        y (list): List of samples
        z (list): List of samples
        k (int, optional): Nearest neighbors. Defaults to 3.
        base (float, optional): Base unit. Defaults to 2.
        metric (str, optional): Metric for distance calculation. Defaults to "euclidean".

    Returns:
        float: Conditional mutual information
    """
    counter = Counter(z)
    unique_values = list(counter.keys())
    unique_counts = list(counter.values())

    mi_xy_given_z = 0
    for unique_val, unique_count in zip(unique_values, unique_counts):
        indices = [i for i, z in enumerate(z) if z == unique_val]
        filtered_x = [x[i] for i in indices]
        filtered_y = [y[i] for i in indices]
        mi_xy_given_z+=(unique_count/sum(unique_counts))*micd(filtered_x, filtered_y, k=k, base=base, metric=metric)
    
    return mi_xy_given_z

def midc(x: list, y: list, k: int=3, base: float=2, metric: str="euclidean") -> float:
    """
    Computes the mutual information between a discrete variable, x,
    and a continuous variable, y. Methodology implements work here
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0087357

    Args:
        x (list): List of samples
        y (list): List of continuously valued vectors
        k (int, optional): Nearest neighbors. Defaults to 3.
        base (float, optional): Base unit. Defaults to 2.
        metric (str, optional): Metric for distance calculation. Defaults to "chebyshev".

    Returns:
        float: Mutual information
    """
    return micd(y, x, k, base, metric)


def centropycd(x: list, y: list, k: int=3, base: float=2, metric: str="euclidean") -> float:
    """
    Computes the entropy of a continous variable, x, conditioned on
    discrete variable, y.
     
    Args:
        x (list): List of continuously valued vectors
        y (list): List of samples
        k (int, optional): Nearest neighbors. Defaults to 3.
        base (float, optional): Base unit. Defaults to 2.
        metric (str, optional): Metric for distance calculation. Defaults to "chebyshev".

    Returns:
        float: Mutual information
    """
    return entropy(x, k, base, metric) - micd(x, y, k, base, metric)


def centropydc(x: list, y: list, k: int=3, base: float=2, metric: str="euclidean") -> float:
    """
    Computes the entropy of a discrete variable, x, conditioned on
    continuous variable, y.
     
    Args:
        x (list): List of continuously valued vectors
        y (list): List of samples
        k (int, optional): Nearest neighbors. Defaults to 3.
        base (float, optional): Base unit. Defaults to 2.
        metric (str, optional): Metric for distance calculation. Defaults to "chebyshev".

    Returns:
        float: Mutual information
    """
    return centropycd(y, x, k=k, base=base, metric=metric)

# UTILITY FUNCTIONS
def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]

def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)

def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))

def build_tree(points, metric="chebyshev"):
    if metric != "chebyshev":
        metric = "euclidean"
    if points.shape[1] >= 20:
        return BallTree(points, metric=metric)
    return KDTree(points, metric=metric)
