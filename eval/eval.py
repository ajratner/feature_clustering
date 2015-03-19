#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Alex Ratner <ajratner@stanford.edu>
# Created: 2015-02-27
import numpy as np
import psycopg2
import re
from itertools import chain
from time import time
from tabulate import tabulate

# connect to the db
def get_db_conn():
  return psycopg2.connect("user=senwu host=raiders4 port=6432 dbname=genomics_recall")

# get evaluation metrics
def evaluate_run(entity):
  conn = get_db_conn()
  cur = conn.cursor()
  table = "%s_is_correct_inference_bucketed" % (entity,)
  res = [["Expectation", "Accuracy (%, Test Set)", "Total (Test Set)"]]
  for b in range(10):
    sql = "SELECT COUNT(*) FROM %s WHERE bucket=%d AND is_correct IS NOT NULL" % (table,b)
    cur.execute(sql)
    num_holdout_total = int(cur.fetchone()[0])
    sql = "SELECT COUNT(*) FROM %s WHERE bucket=%d AND is_correct IS TRUE" % (table,b)
    cur.execute(sql)
    num_holdout_true = int(cur.fetchone()[0])
    res.append([
      b/10.0, 
      "%.02f" % (100.0 * num_holdout_true / float(num_holdout_total),) if num_holdout_total > 0 else "N/A",
      num_holdout_total])
  print "\n%s stats:" % (entity,)
  print tabulate(res, headers="firstrow")
  cur.close()
  conn.close()

# evaluate all
def evaluate_all():
  evaluate_run("genepheno_relations")
  evaluate_run("gene_mentions")
  evaluate_run("pheno_mentions")


### COMMAND LINE ###
if __name__ == '__main__':
  evaluate_all()
