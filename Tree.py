# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 03:05:02 2017

@author: leejia
"""

class Tree:
    def __init__(self, data=None, left=None, right=None, nodenum=None, positivecount=None, negativecount=None):
        self.data = data
        self.left  = left
        self.right = right
        self.nodenum = nodenum
        self.positivecount = positivecount
        self.negativecount = negativecount

    def __str__(self):
        return str(self.data)