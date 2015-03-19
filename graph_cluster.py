#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Alex Ratner <ajratner@stanford.edu>
# Created: 2015-02-27
import numpy as np
from scipy import sparse
import re
from itertools import chain
from time import time
from scipy.cluster.hierarchy import maxdists, fcluster
from scipy.spatial.distance import squareform
from fastcluster import linkage
from util.db_load import *


##### FEATURE CLUSTERING VIA GRAPH REPRESENTATION #####

# Given sentence-feature matrix X (m by d), we calculate the weighted edge matrix 
# E (d by d, symmetric)
# >>  E_ij = \sum_k X_ki * X_kj
# >>       = number of sentences features i and j co-occur in
# Input should be X in COO or CSR sparse matrix format, outputs CSR sparse matrix
def get_edge_matrix(X):
  X = X.tocsr()
  return X.T*X

# Given edge matrix E (d by d symmetric), we calculate the substitution matrix
# W (d by d, symmetric)
# >>  W_ij = (\sum_{k \in N(i) \cap N(j) \backslash {i,j} \omega_k * E_ik * E_kj) * 1{E_ij = 0}
# >>       = the weighted sum of the product of edge weights to shared neighbors
# >>       = OR zero if features i and j occur in the same sentence
# >>  where \omega_k is determined by the specified weighting scheme
# Input and output in CSR sparse matrix format
# TODO: profile this and see if i can make faster / lower memory?
def get_subs_matrix(E, N, weighting='idf'):

  # make sure matrix is in proper format
  E = E.tocsr()
  d,d2 = E.shape
  if d != d2:
    raise ValueError("Symmetric matrix required")
  
  # compute weighting matrix (diagonal, d by d)
  if weighting == 'log-idf':
    D = (1/np.log(d))*sparse.diags(np.log(float(d)/E.diagonal()), 0, format="csr")
  elif weighting == 'idf':
    D = sparse.diags(1.0/E.diagonal(), 0, format="csr")
  else:
    D = sparse.identity(d)

  # compute W_hat
  E = E.sqrt()
  W = (1.0/N)*E.T*D*E

  # threshold
  # NOTE: W will be returned in csc format...
  # NOTE: this part is the slowest / highest-memory point, can be refactored if needed
  W[E > 0] = 0
  W.eliminate_zeros()
  return W

# convert subs matrix W -> condensed distance matrix
# NOTE: scalar + sparse matrix addition not implemented in scipy.sparse...
def to_cdm(W):
  X = 1.0 - W.todense()
  X = 0.5*(X.T+X)  # ensure symmetric
  np.fill_diagonal(X, 0)
  return squareform(X)

# Cluster features given substitute / neighborhood matrix W
# Use HAC for now- fastcluster implementation is O(n^2), simple implementation...
def cluster_features(Y):
  Z = linkage(Y, method='ward', preserve_input=False)
  T = fcluster(Z, 1.0, criterion='inconsistent')
  print "%d clusters found!" % (len(set(T)),)
  return T, Z

# save tsv output of feature-cluster mapppings
# TO UPDATE DB:
# 1. move cluster_mappings.tsv to e.g. /lfs/local/0/ajratner
# 2. COPY feature_clusters FROM ... DELIMITER '\t'
# 3. UPDATE genepheno_features SET feature = fc.cluster FROM feature_clusters fc WHERE genepheno_features.feature = fc.feature;
def cluster_mappings_out(fpath_out, T, f_idx):
  with open(fpath_out, 'wb') as f:
    for i,t in enumerate(T):
      f.write("%s\t%s\n" % (f_idx.reverse_lookup[i], t))

# TODO TODO
# - Write SQL / shell scripts for loading cluster mappings -> DB (load mappings to db as separate table, then do UPDATE statement?)
# - RUN everything!
# - compare w/ (1) random clustering, (2) different clusterings, (3) different tfidf metrics

# current pipeline
def run_pipeline(fpath_out='cluster_mappings.tsv', return_data=False):

  t0 = time()
  print "Loading GP Relation-Feature matrix X..."
  X, e_idx, f_idx = load_gp_relation_feature_matrix()
  N, d = X.shape
  print "[Done in %.02fs]" % (time()-t0,)

  t0 = time()
  print "Calculating edge matrix E..."
  E = get_edge_matrix(X)
  print "[Done in %.02fs]" % (time()-t0,)

  t0 = time()
  print "Calculating substitutions matrix W..."
  W = get_subs_matrix(E, N)
  print "[Done in %.02fs]" % (time()-t0,)

  t0 = time()
  print "Converting to compressed distance matrix Y..."
  Y = to_cdm(W)
  print "[Done in %.02fs]" % (time()-t0,)

  t0 = time()
  print "Calculating linkage matrix Z..."
  T, Z = cluster_features(Y)
  print "[Done in %.02fs]" % (time()-t0,)

  print "Printing TSV cluster mapping to %s & finishing." % (fpath_out,)
  cluster_mappings_out(fpath_out, T, f_idx)
  if return_data:
    return e_idx, f_idx, X, E, W, Z, T
  else:
    return True

##### TESTING #####
X = sparse.csr_matrix(np.array([
[0, 1, 1, 0],
[1, 0, 1, 0],
[0, 1, 0, 1],
[0, 0, 1, 1],
[0, 1, 1, 0]]))
