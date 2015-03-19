#!/usr/bin/env python
# Author: Alex Ratner <ajratner@stanford.edu>
# Created: 2015-02-15
# Functions for sampling random features and clustering
import numpy as np
from scipy import sparse
from scipy.io import savemat
from collections import Counter
from feature_matrix import load_all, get_db_conn
from multiprocessing import Pool
from time import time
from sklearn.metrics.pairwise import euclidean_distances
from unseen import prob_unseen
import re


##### CLUSTERING / SAMPLING FEATURES #####

# csr sparse matrix dot product
def dot(a,b):
  return (a*b.T)[0,0]

# Take the union of two binary vectors represented as np.array \in \{0,1\}^n
def union_binary(X):
  X = X.sum(0)
  X[X>0] = 1
  return X


# get the fingerprints vector from a vector of e.g. cluster assignments
# 0-indexing => F_i = fingerprint[i-1]
def get_fingerprint(v):
  counts = [c[1] for c in Counter(list(v)).most_common()]
  max_count = max(counts)
  f_dict = Counter(counts)
  return np.array([f_dict[i+1] for i in range(max_count)], dtype=int)
  

# we utilize dot-product expansion form to take advantage of sparse matrix ops and precompute
#   d(x,y) = sqrt(x^2 - 2xy + y^2)
# inputs = [x, y, dot(x,x), dot(y,y)]
def euclidean_dist(inputs):
  x = inputs[0]
  y = inputs[1]
  x_sq = inputs[2]
  y_sq = inputs[3]
  return np.sqrt(x_sq - 2*dot(x,y) + y_sq)


# distance of x from a cluster centroid (as computed here) where we measure relative to x
def binary_cluster_dist(inputs):
  x = inputs[0]
  c = inputs[1]
  return 1.0 - dot(x,c)/x.sum()
  

# given a feature - example matrix, returns a set of cluster assignments 
# for each of the features, where > 0 => feature used
def cluster(X, gamma=1.0, omega=0.5, B=100, pool_size=8):

  # assume we get a examples-by-features matrix and transpose
  X = X.transpose().tocsr()
  d,n = X.shape
  print "X: (%s features, %s examples)" % X.shape

  # initialize the sampling order, cluster assignment vec, etc
  clusters = np.zeros(X.shape[0], dtype=int)  # cluster assignments
  C = []                                          # cluster 'centroids'
  C_sq = []                                       # precomputing c.dot(c)
  r = np.random.permutation(X.shape[0])           # feature sampling order
  n_clusters = 0

  # loop through the batches
  pool = Pool(processes = pool_size)
  n_batches = np.ceil(d/float(B))
  t0 = time()
  p_unseens_gt = []
  p_unseens_v = []
  fingerprints = []
  for b in range(20):
  #for b in range(int(np.ceil(X.shape[0]/B))):
    print "[INFO] Starting batch %s / %s [t = %ds]" % (b, int(n_batches), time()-t0)

    # loop through the features in the batch & add to clusters
    for i in r[b*B:(b+1)*B]:
      f = X[i]
      #f_sq = dot(f,f)
      
      # find the closest cluster for this feature vector
      #dists = pool.map(euclidean_dist, [(f,C[i],f_sq,C_sq[i]) for i in range(n_clusters)])
      dists = pool.map(binary_cluster_dist, [(f,C[i]) for i in range(n_clusters)])

      # select whether to assign to existing cluster or to create new one based on thresh
      if len(dists) > 0:
        c_min = np.argmin(dists)
        if dists[c_min] < gamma:
          clusters[i] = c_min
          C[c_min] = union_binary(X[clusters==c_min])
          #C_sq[c_min] = dot(C[c_min],C[c_min])
        else:
          clusters[i] = n_clusters + 1
          n_clusters += 1
          C.append(X[i])
          #C_sq.append(dot(X[i],X[i]))
      else:
        clusters[i] = n_clusters + 1
        n_clusters += 1
        C.append(X[i])
        #C_sq.append(dot(X[i],X[i]))

    # calculate the support size / P(unseen) estimate for the batch
    fingerprint = get_fingerprint(filter(lambda c : c > 0, clusters))
    fingerprints.append(fingerprint)
    p_unseen_gt = fingerprint[0]/float((b+1)*B)
    print "[INFO] P(unseen) using Good-Turing: %s" % (p_unseen_gt,)
    p_unseens_gt.append(p_unseen_gt)
    try:
      p_unseen_v = prob_unseen(fingerprint)
    except:
      p_unseen_v = None
    print "[INFO] P(unseen) using Valiant: %s" % (p_unseen_v,)
    p_unseens_v.append(p_unseen_v)
    
    # stopping criterion
    #if p_unseen < omega:
    #  break

  # save fingerprints for output to MATLAB
  max_len = max([len(fp) for fp in fingerprints])
  fp_padded = []
  for fp in fingerprints:
    fp_padded.append(np.concatenate((fp, np.zeros(max_len-len(fp)))))
  savemat('fingerprints.mat', {'F':fp_padded})

  # print some statistics about the clusters
  print "[INFO] Cluster statistics:"
  print "[INFO] Total # of features used: %d" % ((b+1)*B,)
  print "[INFO] Total # of clusters used: %d" % (len(set(list(clusters)))-1,)
  print "[INFO] Avg. dist. (last batch): %s" % (np.mean(dists),)
  #print "[INFO] Final fingerprint: \n%s" % (list(fingerprint),)
  return clusters, p_unseens_v, p_unseens_gt


# upload clusters to db
def upload_new_features(entity):
  F, e_idx, f_idx = load_all(entity)
  clusters, p_unseens_v, p_unseens_gt = cluster(F)
  mc = max(clusters)
  conn = get_db_conn()
  cur = conn.cursor()
  for c in range(1,int(mc)+1):
    old_fs = "','".join([re.sub("\"","",f_idx.reverse_lookup[i]) for i in clusters if i == c])
    sql = "UPDATE %s_features SET feature=%s,used=True WHERE feature IN ('%s')" % (entity,c,old_fs)
    cur.execute(sql)
  conn.commit()
  conn.close()



##### COMMAND-LINE EXECUTION #####
if __name__ == '__main__':

  # HYPERPARAMETERS
  GAMMA = 0.5
  OMEGA = 0.5
  B = 100

  # execution script
  F, e_idx, f_idx = load_all("genepheno")
  clusters = cluster(F, gamma=GAMMA, omega=OMEGA, B=B)
