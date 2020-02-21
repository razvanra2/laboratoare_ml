import numpy as np
from zipfile import ZipFile
from random import randint
import urllib.request
# Plotting stuff
import matplotlib.pyplot as plt
import matplotlib.markers
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
import scipy
from functools import reduce
import operator as op
#from scipy.special import comb
#from sklearn.metrics.cluster import adjusted_rand_score


# @param ['Atom', 'Chainlink', 'EngyTime', 'GolfBall', 'Hepta', 'Lsun', 'Target', 'TwoDiamonds', 'WingNut']
DATASET_NAME = 'WingNut'

# Numărul de clustere
K = 2  # @param {type: "slider", min: 2, max: 10}


def getArchive():
    """ Checks if FCPS.zip is present in the local directory, if not,
    downloads it.

    Returns:
        A ZipFile object for the FCPS archive
    """

    archive_url = ("https://github.com/cs-pub-ro/ML/raw/master/lab1/FCPS.zip")
    local_archive = "FCPS.zip"

    from os import path
    if not path.isfile(local_archive):
        import urllib
        print("Downloading...")
        urllib.request.urlretrieve(archive_url, filename=local_archive)
        assert(path.isfile(local_archive))
        print("Got the archive")

    return ZipFile(local_archive)


def getDataSet(archive, dataSetName):
    """ Get a dataset from the FCPS.zip

    Args:
        archive (ZipFile): Object for the FCPS
        dataSetName (String): The dataset name from the FCPS

    Returns:
        A tuple (Xs, labels)
        Xs (numpy array): rows are the elements and the cols are the features
        labels (numpy array): labels associated with Xs

    """

    path = "FCPS/01FCPSdata/" + dataSetName

    lrnFile = path + ".lrn"
    with archive.open(lrnFile, "r") as f:
        N = int(f.readline().decode("UTF-8").split()[1])
        D = int(f.readline().decode("UTF-8").split()[1])
        f.readline()
        f.readline()
        Xs = np.zeros([N, D-1])
        for i in range(N):
            data = f.readline().decode("UTF-8").strip().split("\t")
            assert (len(data) == D)
            assert (int(data[0]) == (i + 1))
            Xs[i] = np.array(list(map(float, data[1:])))

    clsFile = path + ".cls"
    with archive.open(clsFile, "r") as f:
        labels = np.zeros(N).astype("uint")

        line = f.readline().decode("UTF-8")
        while line.startswith("%"):
            line = f.readline().decode("UTF-8")

        i = 0
        while line and i < N:
            data = line.strip().split("\t")
            assert (len(data) == 2)
            assert (int(data[0]) == (i + 1))
            labels[i] = int(data[1])
            line = f.readline().decode("UTF-8")
            i = i + 1

        assert (i == N)

    return Xs, labels


def plotClusters(Xs, labels, centroids, clusters):
    """ Plot the data with the true labels alongside the centroids and the
    predicted cluster.
    If the elements from the dataset are not 2 or 3 dimensional then print
    the index, predicted cluster and true label.

    Args:
        Xs (numpy array): dataset
        labels (numpy array): real/true labels
        centroids (numpy array): positions for the centroids
        clusters (numpy array): predicted labels
    """

    labelsNo = np.max(labels)
    K = centroids.shape[0]

    markers = []

    while len(markers) < labelsNo:
        markers.extend(list(matplotlib.markers.MarkerStyle.filled_markers))

    colors = plt.cm.rainbow(np.linspace(0, 1, K+1))
    if Xs.shape[1] == 2:
        x = Xs[:, 0]
        y = Xs[:, 1]
        for (_x, _y, _c, _l) in zip(x, y, clusters, labels):
            plt.scatter(_x, _y, s=500, c=[colors[_c]], marker=markers[_l])
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    s=800, c=[colors[K]], marker=markers[labelsNo])
        plt.show()
    elif Xs.shape[1] == 3:
        x = Xs[:, 0]
        y = Xs[:, 1]
        z = Xs[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for (_x, _y, _z, _c, _l) in zip(x, y, z, clusters, labels):
            ax.scatter(_x, _y, _z, s=200, c=[colors[_c]], marker=markers[_l])
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
                   s=400, c=[colors[K]], marker=markers[labelsNo])
        plt.show()
    else:
        for i in range(Xs.shape[0]):
            print(f"{i} : {clusters[i]} ~ {labels[i]}")

Xs, labels = getDataSet(getArchive(), DATASET_NAME)

def initkMeanspp(Xs, K):
    (N, D) = Xs.shape
    centroids = np.zeros([K, D])
    centroids[0] = Xs[randint(0, N - 1)]

    for k in range(1, K):
        D2 = scipy.array([min([scipy.inner(c-x, c-x) for c in centroids]) for x in Xs])
        probs = D2/D2.sum()
        cumprobs = probs.cumsum()

        r = 0.5 # heuristic

        i = -1
        
        for _, p in enumerate(cumprobs):
            if r < p:
                i = _
                break
        
        centroids[k] = Xs[i]

    return centroids

def dist(a, b, ax = 1):
    return np.linalg.norm(a - b, axis=ax)


def kMeans(K, Xs):
    (N, D) = Xs.shape
 
    # centroids = init_kaufman(Xs, K)
    centroids = initkMeanspp(Xs, K)
    old_centroids = np.zeros([K, D])
    clusters = np.zeros(N).astype("uint")  # id of cluster for each example

    err = dist(centroids, old_centroids)

    while err.all() != 0:
        for i in range(N):
            distances = dist(Xs[i], centroids)
            clusters[i] = np.argmin(distances)
        old_centroids = deepcopy(centroids)

        for i in range(K):
            points = [Xs[j] for j in range(len(Xs)) if clusters[j] == i]
            centroids[i] = np.mean(points, axis=0)
        err = dist(centroids, old_centroids)

    return clusters, centroids

clusters, centroids = kMeans(K, Xs)
plotClusters(Xs, labels, centroids, clusters)

def randIndex(clusters, labels):

    K = clusters.shape[0]
    TP = FP = FN = TN = 0

    for i in range(K):
        for j in range(K):
            if clusters[i] == clusters[j] and labels[i] == labels[j]:
                TP = TP + 1
            if clusters[i] == clusters[j] and labels[i] != labels[j]:
                FP = FP + 1
            if clusters[i] != clusters[j] and labels[i] == labels[j]:
                FN = FN + 1
            if clusters[i] != clusters[j] and labels[i] != labels[j]:
                TN = TN + 1

    randIndex = (TP + TN) / (TP + FP + FN + TN);
    return randIndex

print("randIndex:", randIndex(clusters, labels))
