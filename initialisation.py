# %%
# Import Dataset

import kagglehub 
from kagglehub import KaggleDatasetAdapter

df = kagglehub.dataset_load(KaggleDatasetAdapter.PANDAS, 
                            "sadiajavedd/students-academic-performance-dataset", 
                            "StudentsPerformance.csv")

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
    "high school": "high_school_cerif", 
    "associate's degree": "associate_degree", 
    "some high school": "unknown_high_school_cerif"
    })

df['lunch'] = df['lunch'].cat.rename_categories({
    'free/reduced': "subsidised"})

def dataframe_object():
    return df