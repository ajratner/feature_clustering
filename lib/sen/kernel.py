import os
import sys
import math


FEAT_SEP = "%|%"
WORD_SEP = "@|@"
WORD_PRE = "~!"

dd = dict()
fin = open("dict.txt", 'r')
for line in fin:
  dd[line.strip().split('\t')[1].lower()] = line.strip().split('\t')[0].lower()
fin.close()


def _transformation(x):
  ret = []
  for word in x:
    if word.startswith('~!'):
      if word[2:].lower() in dd:
        ret.append(word[2:].lower())
        continue
      try:
        int(word[2:])
        ret.append(word[2:])
      except:
        ret.append(word)
    else:
      ret.append(word)
  return ret


def _get_sub_tree(x):
  ret = set()
  if len(x.split(FEAT_SEP)) == 3:
    l = _transformation(x.split(FEAT_SEP)[1].split(WORD_SEP))
    r = _transformation(x.split(FEAT_SEP)[2].split(WORD_SEP))
    for i in range(len(l)):
      ret.add(" ".join(l[:i + 1]))
    for i in range(len(l)):
      ret.add(" ".join(reversed(l[-1 - i:])))
    for i in range(len(r)):
      ret.add(" ".join(r[:i + 1]))
    for i in range(len(r)):
      ret.add(" ".join(reversed(r[-1 - i:])))
    ret.add(" ".join(l) + '(^.^)' + " ".join(r))
  else:
    l = _transformation(x.split(FEAT_SEP)[1].split(WORD_SEP))
    for i in range(len(l)):
      ret.add(" ".join(l[:i + 1]))
    for i in range(len(l)):
      ret.add(" ".join(reversed(l[-1 - i:])))
    ret.add(" ".join(l))

  return ret

def kernel(x, y):
  if x.startswith('LEN') and y.startswith('LEN'):
    l1 = x.split(FEAT_SEP)[1].split(WORD_SEP)
    l2 = y.split(FEAT_SEP)[1].split(WORD_SEP)
    s = 1.0
    for i in range(len(l1)):
      s += (float(l1[i]) - float(l2[i])) * (float(l1[i]) - float(l2[i]))
    return 1.0 / (math.sqrt(s))
  if x.startswith('LEN') or y.startswith('LEN'):
    return 0.0
  try:
    s1 = _get_sub_tree(x)
  except:
    s1 = set()
  try:
    s2 = _get_sub_tree(y)
  except:
    s2 = set()
  try:
    return float(len(s1.intersection(s2))) / float((len(s1.union(s2))))
  except:
    return 0.0


