#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 16:02:37 2019

@author: tyler
"""

from cnn import build_cnn
is_training = True
batch_size = 7
height = 32
width = 256
channels = 1

o,i =build_cnn(is_training, batch_size, height, width, channels)