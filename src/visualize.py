# src/visualize.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("artifacts/predictions_all.csv")

sns.countplot(x="pred", data=df)
plt.title("Distribution of Predicted Classes")
plt.xlabel("Predicted Label")
plt.ylabel("Count")
plt.savefig("artifacts/plots/predicted_distribution.png")
plt.close()
print("Saved artifacts/plots/predicted_distribution.png")
