#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 23:35:37 2025

@author: eddie
"""

import matplotlib.pyplot as plt
import numpy as npy

import seaborn as sns
sns.set_theme(style = "whitegrid")

import os

os.system("python initialisation.py")

sns.catplot(data = df, x = "gender", y = "reading_score")




