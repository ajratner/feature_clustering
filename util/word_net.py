#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Alex Ratner <ajratner@stanford.edu>
# Created: 2015-02-27
from nltk.corpus import wordnet as wn
import re
from itertools import chain

# given two feature labels, find the similarity between them using wordnet
def wn_sim(f1, f2):
  
  # extract words from the feature labels
  f1w = list(chain.from_iterable([re.findall(r'[a-z]+', ws, flags=re.I) for ws in re.findall(r'\[.*?\]',f1)]))
  f2w = list(chain.from_iterable([re.findall(r'[a-z]+', ws, flags=re.I) for ws in re.findall(r'\[.*?\]',f2)]))

  # only consider single word ones for now
  # TODO: go beyond this obviously...
  if len(f1w) != 1 or len(f2w) != 1:
    return 0.0

  # look up wordnet synset similarity
  synsets_1 = wn.synsets(f1w[0])
  synsets_2 = wn.synsets(f2w[0])
  max_sim = 0.0
  for s1 in wn.synsets(f1w[0]):
    for s2 in wn.synsets(f2w[0]):      sim = s1.path_similarity(s2)
      if sim > max_sim:
        max_sim = sim

  # TODO: finish this?
  return None
