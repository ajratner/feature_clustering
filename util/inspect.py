#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Alex Ratner <ajratner@stanford.edu>
# Created: 2015-02-27
import numpy as np
from scipy import sparse
import re
from itertools import chain
from time import time
from db_load import get_db_conn


##### TOOLS FOR DATA INSPECTION #####

# Inspect cluster
def analyze_cluster_connectivity(c, T, X, E, W):
  c_idx = [i for i,t in enumerate(T) if t == c]
  print "Avg. # of node edges in W:"
  print np.mean([W.getcol(i).nnz for i in c_idx])
  print "% of nodes with > 0 edges in W:"
  print sum([1 if W.getcol(i).nnz > 0 else 0 for i in c_idx]) / float(len(c_idx))
  print "Avg. # of edges in E:"
  print np.mean([E.getcol(i).nnz for i in c_idx])
  print "% of nodes with > 0 edges in E:"
  print sum([1 if E.getcol(i).nnz > 0 else 0 for i in c_idx]) / float(len(c_idx))
  
# Connection matrix sparsity stats
def edge_matrix_sparsity_stats(W):
  d, d2 = W.shape
  if d != d2:
    raise ValueError("Symmetric matrix required")
  print "% of entries non-zero:"
  print W.nnz/float(d**2)
  if type(W) == sparse.csr.csr_matrix:
    print "% of non-zero rows:"
    print sum([1 if W.getrow(i).nnz > 0 else 0 for i in range(d)]) / float(d)
    print "Avg. # of node edges:"
    print np.mean([W.getrow(i).nnz for i in range(d)])
  elif type(W) == sparse.csc.csc_matrix:
    print "% of non-zero rows:"
    print sum([1 if W.getcol(i).nnz > 0 else 0 for i in range(d)]) / float(d)
    print "Avg. # of node edges:"
    print np.mean([W.getcol(i).nnz for i in range(d)])
  else:
    raise ValueError("Unsupported matrix type")

# Get pairwise maxes i.e. for every feature i, get the feature j with highest substitution
# score wrt i
def get_pairwise_maxes(W):
  row_maxes = []
  for i in range(W.shape[0]):
    row = W.getrow(i)
    r = row.indices[row.data.argmax()]
    row_maxes.append([(i,r), W[i,r]])
  return sorted(row_maxes, key = lambda x : -x[1])

def load_relation_sentence(doc_id, relation_id, cur):
  cur.execute("SELECT sent_id_1 FROM genepheno_relations WHERE doc_id=%s AND relation_id=%s", (doc_id, relation_id))
  sent_id = cur.fetchone()[0]
  cur.execute("SELECT words FROM sentences_input WHERE doc_id=%s AND sent_id=%s", (doc_id, sent_id))
  return re.sub(r'\|\^\|', ' ', cur.fetchone()[0])

def inspect_feature_pair(p, X, e_idx, f_idx, X_row_type='relation'):
  conn = get_db_conn()
  cur = conn.cursor()
  i = p[0]
  j = p[1]

  # Print feature labels
  f1 = f_idx.reverse_lookup[i]
  f2 = f_idx.reverse_lookup[j]
  print "Feature %d:  %s\nFeature %d:  %s" % (p[0], f1, p[1], f2)

  # print some examples of sentences for each
  for k in [i,j]:
    print "\nFeature %s example sentences:" % (k,)
    feature_sentences = X.tocsc()[:,k].nonzero()[0]
    for fs in feature_sentences[:3]:
      doc_id, sent_id = re.split(r',', e_idx.reverse_lookup[fs])
      cur.execute("SELECT words FROM sentences_input WHERE doc_id=%s AND sent_id=%s", (doc_id, sent_id))
      sent = re.sub(r'\|\^\|', ' ', cur.fetchone()[0])
      print "SENTENCE (%s:%s): %s" % (doc_id, sent_id, sent)
  cur.close()
  conn.close()
