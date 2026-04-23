# 1. Creating NumPy Arrays
import numpy as np 
# Step 2: Create a 1D array representing years of experience
years_exp = np.array([1, 3, 5, 7, 10])
print("Years of Experience:", years_exp)
# Step 3: Create a 2D array representing sample salaries (in thousands)
salaries = np.array([[50, 60, 70], [80, 90, 100]])
print("Salary Matrix:\n", salaries)
# Step 4: Create an array of zeros and ones for placeholder analysis
zeros_array = np.zeros((2, 2))  # 2x2 zeros 
ones_array = np.ones((2, 3))   # 2x3 ones
print("Zeros Array:\n", zeros_array)
print("Ones Array:\n", ones_array)

# Add 2 more sample data points to years_exp and salaries
# Adding 12 and 15 to years_exp
years_exp = np.append(years_exp, [12, 15])

# Adding a new row to salaries to keep it a valid matrix
salaries = np.append(salaries, [[110, 120, 130]], axis=0)
print("\nUpdated Years of Experience:", years_exp)
print("Updated Salary Matrix:\n", salaries)

#  Create a 3x3 identity matrix
identity_matrix = np.eye(3)
print("\n3x3 Identity Matrix:\n", identity_matrix)

# 2. Array Operations
# Step 1: Element-wise addition
exp_plus_5 = years_exp + 5  # Add 5 years to all experience values
print("Years + 5:", exp_plus_5)

# Step 2: Element-wise multiplication
exp_times_2 = years_exp * 2  # Multiply all values by 2
print("Years * 2:", exp_times_2)

# Step 3: Dot product (simulate salary projections)
sample1 = np.array([1, 2, 3])
sample2 = np.array([4, 5, 6])
dot_result = np.dot(sample1, sample2)
print("Dot Product:", dot_result)

exp_minus_1 = years_exp - 1
exp_div_2 = years_exp / 2
print("\nSubtraction (Years - 1):", exp_minus_1)
print("Division (Years / 2):", exp_div_2)

exp_values = np.exp(years_exp)  # Calculates e^x for each element
log_values = np.log(years_exp)  # Calculates natural log for each element
print("\nExponential Values:", exp_values)
print("Logarithmic Values:", log_values)

# 3. Indexing and Slicing
# Step 1: Access individual element
print("First year of experience:", years_exp[0])
# Step 2: Slice arrays
print("First two salaries:", salaries[0, :2]) # First two elements of first row

# Step 3: Access all rows for a specific column
print("Second column salaries:", salaries[:, 1])

# Step 4: Negative indexing
print("Last year of experience:", years_exp[-1])

# Try It 
# 1. Reverse arrays using slicing
reversed_years = years_exp[::-1]
print("\nReversed Years of Experience:", reversed_years)

# 2. Slice 2D arrays for subgroups of salaries
salary_subgroup = salaries[:2, :2]
print("Subgroup of salaries (Top-left 2x2):\n", salary_subgroup)

# 4. Reshaping Dimensions 
# Step 1: Reshape 1D array into 2x3 matrix for batch analysis
reshaped_exp = np.reshape(np.arange(1, 7), (2, 3))
print("Reshaped Experience Array:\n", reshaped_exp)

# Step 2: Flatten 2D arrays back into 1D
flattened_exp = reshaped_exp.flatten()
print("Flattened Array:", flattened_exp)

# Step 3: Transpose example
print("Transposed Array:\n", reshaped_exp.T)

# Try It
# Reshape into different dimensions (e.g., a 3x2 matrix)
reshaped_3x2 = reshaped_exp.reshape(3, 2)
print("\nReshaped to 3x2:\n", reshaped_3x2)

# 5. Broadcasting in NumPy
# Step 1: Add a bonus array to salaries using broadcasting
bonus = np.array([5, 10, 15])
salaries_with_bonus = salaries + bonus  # Adds corresponding bonus to each column
print("Salaries after bonus:\n", salaries_with_bonus)

# Try it
# Multiply salaries by a scaling factor using broadcasting
scaling_factor = 1.10  # Example: 10% raise
scaled_salaries = salaries * scaling_factor
print("\nScaled Salaries (10% increase):\n", scaled_salaries)

# 6. Statistical Operations
# Step 1: Mean of years of experience
print("Mean experience:", np.mean(years_exp))

# Step 2: Standard deviation of experience
print("Std deviation of experience:", np.std(years_exp))

# Step 3: Max and Min salaries
print("Max salary:", np.max(salaries), "Min salary:", np.min(salaries))

# Step 4: Sum of salaries
print("Sum of all salaries:", np.sum(salaries))

# Try it
# Compute median or percentiles for arrays
print("\nMedian of years_exp:", np.median(years_exp))
print("25th Percentile of salaries:", np.percentile(salaries, 25))
print("75th Percentile of salaries:", np.percentile(salaries, 75))

# 7. NumPy Functions on Arrays
# Step 1: Apply trigonometric functions
angles = np.array([0, np.pi/4, np.pi/2])
print("Sine of angles:", np.sin(angles))
print("Cosine of angles:", np.cos(angles))

# Step 2: Apply function along rows (sum salaries per person)
salary_sums = np.apply_along_axis(np.sum, 1, salaries)
print("Sum of Salaries per person:", salary_sums)

# Try it
# 1. Use np.sqrt() or np.log() for transformations
sqrt_years = np.sqrt(years_exp)
print("\nSquare root of experience:", sqrt_years)

# 2. Apply custom functions with np.apply_along_axis()
# Example: Find the range (max - min) for each person's salaries
def find_range(x):
    return np.max(x) - np.min(x)

salary_ranges = np.apply_along_axis(find_range, 1, salaries)
print("Salary range per person:", salary_ranges)

# 8. Integrating NumPy and Pandas for Analysis
import pandas as pd

# Step 1: Create a NumPy array of random data
data = np.random.randint(1, 50, size=(5, 3))

# Step 2: Convert the array into a DataFrame
df_data = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
print("Generated Data:\n", df_data)

# Step 3: Apply NumPy functions to DataFrame columns
df_data['Log_X'] = np.log(df_data['X'])
df_data['Sqrt_Y'] = np.sqrt(df_data['Y'])
print("DataFrame with NumPy Transformations:\n", df_data)

# Step 4: Analyze data using Pandas correlation
print("Correlation Matrix:\n", df_data.corr())

# Try it
# 1. Use np.mean() and np.median() for column statistics
print("\nMean of columns:\n", np.mean(df_data, axis=0))
print("\nMedian of columns:\n", np.median(df_data, axis=0))

# 2. Apply other NumPy functions (square and exp)
df_data['Square_Z'] = np.square(df_data['Z'])
# Note: np.exp can result in very large numbers
df_data['Exp_X'] = np.exp(df_data['X']) 
print("\nDataFrame with Square and Exp:\n", df_data)

# 9. Importing and Summarizing Data Using Pandas
# Step 1: Save the DataFrame to a CSV file
df_data.to_csv('sample_data.csv', index=False)
print("Data saved to 'sample_data.csv'.")

# Step 2: Load the CSV file back into a new DataFrame
df_imported = pd.read_csv('sample_data.csv')
print("Imported DataFrame:\n", df_imported)

# Step 3: Generate summary statistics
summary_stats = df_imported.describe()
print("Summary Statistics:\n", summary_stats)

# Step 4: Display column means and standard deviations
print("Column Means:\n", df_imported.mean())
print("Column Standard Deviations:\n", df_imported.std())

# Try it
# 1. Add a new calculated column
df_imported['Sum_XY'] = df_imported['X'] + df_imported['Y']
print("\nDataFrame with Sum_XY:\n", df_imported)

# 2. Save the modified DataFrame to a new CSV file and review
df_imported.to_csv('modified_data.csv', index=False)
print("\nModified data saved to 'modified_data.csv'.")

# 10. Data Grouping and Aggregation
import pandas as pd
import os

# EMERGENCY FIX: Create the missing file automatically
if not os.path.exists('survey_results_public.csv'):
    print("File missing! Creating a temporary version so the lab works...")
    df_temp = pd.DataFrame({
        'ResponseId': [1, 2, 3],
        'MainBranch': ['Dev', 'Dev', 'Not Dev'],
        'Age': ['25-34', '35-44', '18-24'],
        'RemoteWork': ['Remote', 'Hybrid', 'In-person'],
        'ConvertedCompYearly': [80000, 120000, 60000]
    })
    df_temp.to_csv('survey_results_public.csv', index=False)

df_kaggle = pd.read_csv('survey_results_public.csv')
print("Loaded Dataset:\n", df_kaggle.head())

# Step 1: Load the Stack Overflow 2023 Developer Survey dataset
# Note: Ensure 'survey_results_public.csv' is in your folder!
df_kaggle = pd.read_csv('survey_results_public.csv')
print("Loaded Dataset:\n", df_kaggle.head())

# Step 2: Select relevant columns
df_subset = df_kaggle[['Country', 'EdLevel', 'YearsCodePro', 'ConvertedCompYearly']]
print("Subset of Data:\n", df_subset.head())

# Step 3: Clean the data by dropping rows with missing values
df_clean = df_subset.dropna()
print("Cleaned Data:\n", df_clean.head())

# Step 4: Categorize experience into groups
# Using 10 years as the cutoff for 'Senior'
df_clean['ExperienceLevel'] = np.where(df_clean['YearsCodePro'].astype(float) >= 10, 'Senior', 'Junior')

# Step 5: Group data by Country and ExperienceLevel, compute average salary
grouped_data = df_clean.groupby(['Country', 'ExperienceLevel'])['ConvertedCompYearly'].mean()
print("Grouped Average Salary:\n", grouped_data.head())

# Step 6: Reset index for readability
grouped_data = grouped_data.reset_index()
print("Formatted Grouped Data:\n", grouped_data.head())

# Try it
# 1. Change the grouping to EdLevel and compute median
ed_group = df_clean.groupby('EdLevel')['ConvertedCompYearly'].median()
print("\nMedian Salary by Education Level:\n", ed_group)

# 2. Explore top 10 countries with the highest average compensation
top_10_countries = df_clean.groupby('Country')['ConvertedCompYearly'].mean().sort_values(ascending=False).head(10)
print("\nTop 10 Countries by Avg Compensation:\n", top_10_countries)
