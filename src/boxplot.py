from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use("TkAgg")

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

print(df.head())
print(df.shape)

plt.figure(figsize=(6, 8))
df["MedHouseVal"].plot.box(vert=True)

plt.xlabel("California Housing Dataset")
plt.ylabel("Median House Value")
plt.title("Boxplot of Median House Value (California Housing)")

os.makedirs("figs", exist_ok=True)
plt.savefig(os.path.join("figs", "california_housing_boxplot.png"))

plt.show()
