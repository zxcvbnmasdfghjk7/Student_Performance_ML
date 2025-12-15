#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 02:23:15 2025

@author: eddie
"""
import numpy as npy
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split

import pandas as pd
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]

# Save Raw Dataset
path = project_root / "data" / "clean"
df = pd.read_csv(path / "student_performance_ml_clean.csv")

# Use train-test split.
df_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)

model_writing_train = smf.ols('writing_score ~ gender + student_ethnicity + parents_degree + lunch + test_prep', data = df_train)

print("OLS Summary for writing_score: ")
print(model_writing_train.fit().summary())

model_read_train = smf.ols('reading_score ~ gender + student_ethnicity + parents_degree + lunch + test_prep', data = df_train)

print("OLS Summary for reading_score:")
print(model_read_train.fit().summary())

model_math_train = smf.ols('math_score ~ gender + student_ethnicity + parents_degree + lunch + test_prep', data = df_train)

print("OLS Summary for math_score:")
print(model_math_train.fit().summary())

