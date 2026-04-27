import numpy as np
import pandas as pd

# Global Student Info
STUDENT_NAME = "Cricylus Garrell Nery"
STUDENT_ID = 25900 

print(f"--- RUNNING LAB 6 ACTIVITIES FOR: {STUDENT_NAME} ({STUDENT_ID}) ---\n")

# --- Activity 1: Basic Statistics ---
np.random.seed(STUDENT_ID)
data = np.random.randint(1, STUDENT_ID % 100 + 50, size=10)
mean_val = np.mean(data)
std_val = np.std(data)
print(f"1. Array: {data}\n   Mean: {mean_val:.2f}, Std Dev: {std_val:.2f}\n")

# --- Activity 2: Matrix Sums ---
np.random.seed(STUDENT_ID)
matrix = np.random.randint(1, STUDENT_ID % 50 + 20, size=(3, 4))
row_sums = np.sum(matrix, axis=1)
col_sums = np.sum(matrix, axis=0)
print(f"2. Matrix:\n{matrix}\n   Row sums: {row_sums}\n   Col sums: {col_sums}\n")

# --- Activity 3: 3D Array Slicing ---
np.random.seed(STUDENT_ID)
arr3d = np.random.randint(0, STUDENT_ID % 20 + 10, size=(3, 3, 3))
slice_mid = arr3d[:, 1, :]
print(f"3. 3D Slice[:, 1, :]:\n{slice_mid}\n")

# --- Activity 4: Boolean Indexing ---
np.random.seed(STUDENT_ID)
x = np.random.randint(1, STUDENT_ID % 100 + 50, size=10)
filtered = x[(x > STUDENT_ID % 50) & (x < STUDENT_ID % 100)]
print(f"4. Original: {x}\n   Filtered: {filtered}\n")

# --- Activity 5: Matrix Algebra ---
np.random.seed(STUDENT_ID)
A = np.random.randint(1, STUDENT_ID % 50 + 20, size=(2, 2))
B = np.random.randint(1, STUDENT_ID % 50 + 20, size=(2, 2))
product = A @ B
det_A = np.linalg.det(A)
print(f"5. Matrix Product A@B:\n{product}\n   Det(A): {det_A:.2f}\n")

# --- Activity 6: Trig Functions ---
angles = np.linspace(0, 2 * np.pi, 8)
sine_vals = np.sin(angles)
cos_vals = np.cos(angles)
print(f"6. Angles: {angles}\n   Sine: {sine_vals}\n")

# --- Activity 7: Fancy Slicing ---
np.random.seed(STUDENT_ID)
arr_mod = np.random.randint(1, STUDENT_ID % 50 + 20, size=(4, 4))
arr_mod[::2, ::2] = 0
print(f"7. Modified 4x4 (Checkerboard Zeros):\n{arr_mod}\n")

# --- Activity 8: np.where ---
np.random.seed(STUDENT_ID)
scores = np.random.randint(0, STUDENT_ID % 100 + 50, 10)
grades = np.where(scores >= 70, 'Pass', 'Fail')
print(f"8. Scores: {scores}\n   Grades: {grades}\n")

# --- Activity 9: Flatten/Transpose ---
np.random.seed(STUDENT_ID)
arr_9 = np.random.randint(1, STUDENT_ID % 50 + 20, size=(3, 4))
print(f"9. Flattened: {arr_9.flatten()}\n   Transposed:\n{arr_9.T}\n")

# --- Activity 10: Masking ---
np.random.seed(STUDENT_ID)
arr_10 = np.random.randint(1, STUDENT_ID % 50 + 20, (3, 3))
arr_10[arr_10 % 2 == 0] = -1
print(f"10. Evens replaced with -1:\n{arr_10}\n")

# --- Activity 11: Pandas DataFrame ---
np.random.seed(STUDENT_ID)
df11 = pd.DataFrame({
    'Name': [STUDENT_NAME] * 5,
    'Score': np.random.randint(STUDENT_ID % 50, STUDENT_ID % 100 + 50, 5),
    'YearsCodePro': np.random.randint(0, STUDENT_ID % 20 + 10, 5)
})
print(f"11. DataFrame:\n{df11}\n")

# --- Activity 12: GroupBy ---
np.random.seed(STUDENT_ID)
df12 = pd.DataFrame({
    'EdLevel': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'ConvertedComp': np.random.randint(40000, 150000, 5)
})
avg_salary = df12.groupby('EdLevel')['ConvertedComp'].mean()
print(f"12. Avg Salary by EdLevel:\n{avg_salary}\n")

# --- Activity 13: Sorting ---
top5 = df12.sort_values(by='ConvertedComp', ascending=False)
print(f"13. Sorted Salaries:\n{top5}\n")

# --- Activity 14: DF Filtering ---
high_exp = df11[df11['YearsCodePro'] > STUDENT_ID % 10]
print(f"14. High Experience Filter:\n{high_exp}\n")

# --- Activity 15: Binning ---
bins = [0, 50000, 100000, 150000, 200000]
labels = ['0-50k', '50-100k', '100-150k', '>150k']
df12['Bracket'] = pd.cut(df12['ConvertedComp'], bins=bins, labels=labels)
print(f"15. Salary Brackets:\n{df12}\n")

# --- Activity 16: Log Transform ---
df12['LogComp'] = np.log(df12['ConvertedComp'])
print(f"16. Log Transformation:\n{df12[['ConvertedComp', 'LogComp']]}\n")

# --- Activity 17: Correlation ---
corr_matrix = df11[['Score', 'YearsCodePro']].corr()
print(f"17. Correlation Matrix:\n{corr_matrix}\n")

# --- Activity 18: Aggregation ---
summary = df12.groupby('EdLevel')['ConvertedComp'].agg(['mean', 'median', 'std']).reset_index()
print(f"18. Summary Stats:\n{summary}\n")

# --- Activity 19: Thresholding ---
df12['HighPay'] = np.where(df12['ConvertedComp'] > 110000, 'Yes', 'No')
print(f"19. Thresholding (>110k):\n{df12[['ConvertedComp', 'HighPay']]}\n")

# --- Activity 20: Country Aggregation ---
df20 = pd.DataFrame({
    'Country': ['Philippines', 'USA', 'UK', 'Canada', 'Germany'],
    'ConvertedComp': np.random.randint(40000, 150000, 5)
})
top_countries = df20.groupby('Country')['ConvertedComp'].agg(['mean', 'max'])
print(f"20. Country Aggregation:\n{top_countries}")