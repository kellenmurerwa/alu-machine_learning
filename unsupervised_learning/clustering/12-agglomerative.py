#!/usr/bin/env python3
"""Performs agglomerative clustering on a dataset."""
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

def agglomerative(X, dist):
    """Performs agglomerative clustering with Ward linkage."""
    # Perform Ward linkage
    linkage_matrix = sch.linkage(X, method='ward')
    
    # Form flat clusters
    clss = sch.fcluster(linkage_matrix, t=dist, criterion='distance')
    
    # Plot the dendrogram
    plt.figure()
    sch.dendrogram(linkage_matrix, color_threshold=dist)
    plt.show()
    
    return clss
