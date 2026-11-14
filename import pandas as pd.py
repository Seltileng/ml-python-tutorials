# Import pandas library for tabular data manipulation and analysis
import pandas as pd

# Load the student scores dataset from CSV format
# Note: Using pd.read_csv to properly parse comma-separated values
data = pd.read_csv('student_scores (3).csv')

# Display the first 5 records of the dataset
print(data.head())
