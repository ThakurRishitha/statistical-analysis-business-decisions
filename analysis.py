import pandas as pd

from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import norm

columns = [
    'CRIM','ZN','INDUS','CHAS','NOX',
    'RM','AGE','DIS','RAD','TAX',
    'PTRATIO','B','LSTAT','MEDV'
]

df = pd.read_csv(
    "4) house_price.csv",
    sep=r"\s+",
    header=None,
    names=columns
)

print("Dataset Loaded Successfully!")
print(df.shape)

# =====================
# T-TEST STARTS HERE
# =====================

river = df[df['CHAS'] == 1]['MEDV']

nonriver = df[df['CHAS'] == 0]['MEDV']

t_stat, p_value = ttest_ind(river, nonriver)

print("\nT-Test Results")
print("T Statistic:", t_stat)
print("P Value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis")
    print("House prices are significantly different.")
else:
    print("Fail to Reject Null Hypothesis")
    print("No significant difference in house prices.")
    # =====================
# CHI-SQUARE TEST
# =====================

df['High_Price'] = (df['MEDV'] > df['MEDV'].median()).astype(int)

contingency_table = pd.crosstab(
    df['CHAS'],
    df['High_Price']
)

print("\nContingency Table")
print(contingency_table)

chi2, p, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Test Results")
print("Chi-Square Statistic:", chi2)
print("P Value:", p)

if p < 0.05:
    print("Reject Null Hypothesis")
    print("River location and high house prices are associated.")
else:
    print("Fail to Reject Null Hypothesis")
    print("No significant association found.")
    # =====================
# PROBABILITY DISTRIBUTION
# RISK ANALYSIS
# =====================

mean_price = df['MEDV'].mean()
std_price = df['MEDV'].std()

probability = 1 - norm.cdf(
    30,
    mean_price,
    std_price
)

print("\nRisk Analysis")
print("Mean House Price:", mean_price)
print("Standard Deviation:", std_price)
print("Probability House Price > 30:", probability)
# =====================
# A/B TESTING
# =====================

groupA = df[df['CHAS'] == 0]['MEDV']
groupB = df[df['CHAS'] == 1]['MEDV']

t_stat, p_value = ttest_ind(groupA, groupB)

print("\nA/B Testing Results")
print("T Statistic:", t_stat)
print("P Value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis")
    print("River-side houses have significantly different prices.")
else:
    print("Fail to Reject Null Hypothesis")
    print("No significant difference in prices.")
    # =====================
# CONFIDENCE INTERVAL
# =====================

import numpy as np

mean = df['MEDV'].mean()

se = df['MEDV'].std() / np.sqrt(len(df))

margin_of_error = 1.96 * se

lower = mean - margin_of_error
upper = mean + margin_of_error

print("\nConfidence Interval")
print("Lower Bound:", lower)
print("Upper Bound:", upper)

print("\nMargin of Error:", margin_of_error)

import matplotlib.pyplot as plt
plt.figure(figsize=(8,5))
plt.hist(df['MEDV'], bins=20)
plt.title('Distribution of House Prices')
plt.xlabel('House Price')
plt.ylabel('Frequency')

plt.savefig("histogram.png")

plt.show()

