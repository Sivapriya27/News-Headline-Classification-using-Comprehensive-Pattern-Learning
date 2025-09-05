import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("artifacts/predictions_all.csv")

ct = pd.crosstab(df['label'], df['pred'])
sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
plt.title("True vs Predicted (Full Dataset)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("artifacts/plots/true_vs_pred.png")
plt.close()
print("Saved artifacts/plots/true_vs_pred.png")
