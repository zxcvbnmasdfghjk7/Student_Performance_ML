#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 23:39:18 2025

@author: eddie
"""
import pandas as pd
from pathlib import Path
from ydata_profiling import ProfileReport

current_path = Path(__file__).resolve()

print(current_path)
project_root_path = current_path.parents[2]

raw_data_dir = project_root_path / "data" / "raw"

# Save Raw Dataset
df = pd.read_csv(project_root_path / "data" / "raw" / "student_performance_ml.csv")

profile = ProfileReport(df, title = "Exploratory Data Analysis Report")

report_directory_path = project_root_path / "reports"

# if reports/ don't exist, create
report_directory_path.mkdir(parents=True, exist_ok=True)

profile.to_file(report_directory_path/"eda_report.html")