import os
import sys
import random
import time
from multiprocessing import Pool
import kernel

node = []
cluster = []

score = list()

num = 100

def f(x):
  if random.random() <= kernel.kernel(x[0], x[1]):
    return x[0]
  else:
    return x[1]

# Usage: python cluster.py features.txt
# Reture: the mapping from feature to cluster
#   featrueA clusterA
#   featrueB clusterA
        
if __name__ == "__main__":
  fin = open(sys.argv[1], 'r')
  for line in fin:
    node.append(line.strip())
    cluster.append(line.strip())
  fin.close()
  num = min(num, len(cluster))
  pool = Pool(processes = 80)
  iter = 0
  for i in range(num):
    start_time = time.time()
    x = random.sample(cluster, 1)[0]
    iter = iter + 1
    cluster = pool.map(f, [(x, y) for y in cluster]) 
    print "[INFO]: Epoch", str(iter), str(time.time() - start_time)
      
  fout = open(sys.argv[2], 'w')
  for i in range(len(node)):
    fout.write(node[i] + "\t" + cluster[i] + "\n")
    
