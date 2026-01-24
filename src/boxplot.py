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
plt.boxplot(df["MedHouseVal"], vert=True)
plt.ylabel("Median House Value")
plt.title("Boxplot of Median House Value (California Housing)")

# Save the figure
plt.savefig("california_housing_boxplot.png", dpi=300, bbox_inches="tight")

# Show plot
plt.show()