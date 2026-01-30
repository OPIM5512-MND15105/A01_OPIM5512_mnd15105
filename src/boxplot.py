from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Creat the boxplot
plt.figure(figsize=(6, 8))
df["MedHouseVal"].plot.box(vert=True)
plt.xlabel("California Housing Dataset")
plt.ylabel("Median House Value")
plt.title("Boxplot of Median House Value (California Housing)")

# Save the figure
plt.savefig("figures/california_housing_boxplot.png")
plt.close()

import matplotlib.pyplot as plt
# Show plot
plt.show()