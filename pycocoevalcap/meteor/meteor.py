#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help 

# Last modified : Wed 22 May 2019 08:10:00 PM EDT
# By Sabarish Sivanath
# To support Python 3

import os
import sys
import subprocess
from nltk.translate import meteor_score
from nltk.tokenize import word_tokenize

# Example candidate and reference translation

# Compute METEOR score
# score = 

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
# print METEOR_JAR

class Meteor:

    def __init__(self):
        pass

    def compute_score(self, gts, res, verbose=0):
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        #import pdb;pdb.set_trace()
        for i in imgIds:
            assert(len(res[i]) == 1)
            # import pdb;pdb.set_trace()
            tmp_res = word_tokenize(res[i][0])
            tmp_gt = word_tokenize(gts[i][0])
            score = meteor_score.meteor_score([tmp_res],tmp_gt)
            scores.append(score)
        score = sum(scores) / len(scores)

        return score, scores

    def method(self):
        return "METEOR"

   