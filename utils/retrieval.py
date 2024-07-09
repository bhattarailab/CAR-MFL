import sys

import numpy as np
from sklearn.cluster import KMeans

from .utils import gentsnePlot


class ModalityRetrival:
    def __init__(self, img_vec, txt_vec, idxs, num_clusters):
        self.img_vec = img_vec
        self.txt_vec = txt_vec
        self.idxs = idxs
        self.num_clusters = num_clusters
    
    def runKmeans(self):
        self.img_cluster = KMeans(n_clusters=self.num_clusters, random_state=0).fit(self.img_vec)
        self.txt_cluster = KMeans(n_clusters=self.num_clusters, random_state=0).fit(self.txt_vec)
        self.img_centriods = self.img_cluster.cluster_centers_
        self.txt_centriods = self.txt_cluster.cluster_centers_
        self.img_assignment = self.img_cluster.predict(self.img_vec)
        self.txt_assignment = self.txt_cluster.predict(self.txt_vec)
    
    def genPlot(self, modality):
        if modality == "image":
            gentsnePlot(self.img_cluster, self.img_assignment, self.img_centriods, "image.png")
        elif modality == "text":
            gentsnePlot(self.txt_cluster, self.txt_assignment, self.txt_centriods, "text.png")

    