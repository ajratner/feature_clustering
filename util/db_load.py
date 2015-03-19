#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Alex Ratner <ajratner@stanford.edu>
# Created: 2015-02-27
import numpy as np
from scipy import sparse
from scipy.io import savemat
import psycopg2
from time import time



#### GENERAL FUNCTIONS FOR *SIMPLE* DB CONNECTION ####

# connect to the db
def get_db_conn():
  return psycopg2.connect("user=senwu host=raiders4 port=6432 dbname=genomics_recall")



##### UTILITIES FOR LOADING THE SENTENCE-FEATURE MATRIX X #####

# simple dictionary class for indexing
class Index:
  
  def __init__(self):
    self.table = {}
    self.reverse_lookup = {}
    
  def get_index(self, x):
    if self.table.has_key(x):
      return self.table[x]
    else:
      k = len(self.table)
      self.table[x] = k
      self.reverse_lookup[k] = x
      return k

  def __len__(self):
    return len(self.table)
    
# load all the features into a sparse matrix
def load_feature_matrix(sql):
  
  # Get data & index by (e, feature)
  e_idx = Index()
  f_idx = Index()
  data_rows = []
  data_cols = []
  conn = get_db_conn()
  cur = conn.cursor()
  cur.execute(sql)
  for row in cur:
    ex_id = "%s,%s" % (row[0], row[1])
    data_rows.append(e_idx.get_index(ex_id))
    data_cols.append(f_idx.get_index(row[2]))
  cur.close()
  conn.close()
      
  # Load into sparse matrix- COO format for conversion to CSC / CSR
  F = sparse.coo_matrix(([1]*len(data_rows), (data_rows, data_cols)), dtype=np.double)
  return F, e_idx, f_idx

def load_gp_relation_feature_matrix():
  sql = "SELECT * FROM genepheno_features"
  return load_feature_matrix(sql)

def load_gp_sentence_feature_matrix():
  sql = """
  SELECT 
    f.doc_id, r.sent_id_1, f.feature
  FROM
    genepheno_features f, genepheno_relations r
  WHERE
    f.doc_id = r.doc_id AND f.relation_id = r.relation_id"""
  return load_feature_matrix(sql)

def subsample_fm(F, subsample=0.1):
  idx = range(F.shape[0])
  np.random.shuffle(idx)
  k = int(subsample*F.shape[0])
  return F.tocsr()[idx[:k]].tocoo()
