# %%
# Import Dataset

import kagglehub 
from kagglehub import KaggleDatasetAdapter

from pathlib import Path

df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, 
                            "sadiajavedd/students-academic-performance-dataset", 
                            "StudentsPerformance.csv")

project_root = Path(__file__).resolve().parents[2]

# Save Raw Dataset
raw_path = project_root / "Student_Performance_ML" / "data" / "raw"
df.to_csv(raw_path / "student_performance_ml.csv", index=False)

print("First 10 Observations in dataset")
print(df.head(n = 10))

print("Display Info on all variables")
print(df.info())

# ------------------------------------------------------------------------
## Data Wrangling
# ------------------------------------------------------------------------

# Check for duplicate entries
df.duplicated()

# Drop duplicate observations for unique entries.
df.drop_duplicates(inplace = True)

# Fix formatting for variables
df.rename(columns = {"race/ethnicity": "student_ethnicity", 
                     "parental level of education":"parents_degree", 
                     "test preparation course": "test_prep", 
                     "math score": "math_score", 
                     "reading score": "reading_score", 
                     "writing score": "writing_score"}, inplace = True)

print(df.head(n = 10).T)
print("Display Info on all updated variables")
print(df.info())

# Convert objects to catagorical variables
df["gender"] = df["gender"].astype("category")
df["student_ethnicity"] = df["student_ethnicity"].astype("category")
df["parents_degree"] = df["parents_degree"].astype("category")
df["lunch"] = df["lunch"].astype("category")
df["test_prep"] = df["test_prep"].astype("category")


# Show all unique catagorical variables.
print("Unique Catagorical Variables are below, ")
for col in ['gender', 'student_ethnicity', 'parents_degree', 'lunch', 'test_prep']:
    print(df[col].unique())

# Rename categories into 'reasonable' entries.    
df["student_ethnicity"] = df["student_ethnicity"].cat.rename_categories({
    'group A': 'A', 
    'group B': 'B', 
    'group C': 'C', 
    'group D': 'D', 
    'group E': 'E'
    })

df['parents_degree'] = df['parents_degree'].cat.rename_categories({
    "bachelor's degree": 'bachelor_degree', 
    "some college": 'unknown_college_degree', 
    "master's degree": "master_degree", 
    "high school": "high_school_degree", 
    "associate's degree": "associate_degree", 
    "some high school": "unknown_high_school_degree"
    })

df['lunch'] = df['lunch'].cat.rename_categories({
    'free/reduced': "subsidised"})

df['test_prep'] = df['test_prep'].cat.rename_categories({
    "none": "no", 
    "completed": "yes"})

# save "clean" csv
clean_path = project_root / "Student_Performance_ML" / "data" / "clean"
df.to_csv(clean_path / "student_performance_ml_clean.csv", index=False)

