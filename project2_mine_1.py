import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

raw = load_iris()

flower_df = pd.DataFrame(raw.data, columns=["sepal_len", "sepal_wid", "petal_len", "petal_wid"])
flower_df["label"] = raw.target
flower_df["flower"] = flower_df["label"].map({0: "Setosa", 1: "Versicolor", 2: "Virginica"})

print("First few rows of my dataset:")
print(flower_df.head(8))
print("\nTotal rows and columns:", flower_df.shape)
print("Any missing values?", flower_df.isnull().sum().sum())
print("\nClass distribution:")
print(flower_df["flower"].value_counts())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Iris Flower Data - Visual Check", fontsize=13)

my_colors = {"Setosa": "tomato", "Versicolor": "mediumseagreen", "Virginica": "steelblue"}

for fname, grp in flower_df.groupby("flower"):
    ax1.scatter(grp["sepal_len"], grp["sepal_wid"], label=fname,
                color=my_colors[fname], s=65, alpha=0.75, edgecolors="white")
ax1.set_xlabel("Sepal Length")
ax1.set_ylabel("Sepal Width")
ax1.set_title("Sepal Comparison")
ax1.legend()
ax1.grid(alpha=0.3)

for fname, grp in flower_df.groupby("flower"):
    ax2.scatter(grp["petal_len"], grp["petal_wid"], label=fname,
                color=my_colors[fname], s=65, alpha=0.75, edgecolors="white")
ax2.set_xlabel("Petal Length")
ax2.set_ylabel("Petal Width")
ax2.set_title("Petal Comparison")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("flower_scatter.png", dpi=150)
plt.show()

features = flower_df[["sepal_len", "sepal_wid", "petal_len", "petal_wid"]].values
target   = flower_df["label"].values

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=7, shuffle=True)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

print("\nTraining size:", X_train.shape[0])
print("Testing size :", X_test.shape[0])

err = []
k_vals = range(1, 26)

for k in k_vals:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    pred_temp = clf.predict(X_test)
    err.append(1 - accuracy_score(y_test, pred_temp))

plt.figure(figsize=(10, 4))
plt.plot(k_vals, err, marker="o", color="navy", linewidth=2,
         markersize=6, markerfacecolor="orangered")
plt.title("Finding Best K Value", fontsize=12)
plt.xlabel("K")
plt.ylabel("Error Rate")
plt.xticks(k_vals)
plt.grid(alpha=0.4)

best_k = k_vals[err.index(min(err))]
plt.axvline(x=best_k, color="orangered", linestyle="--", label=f"Best K = {best_k}")
plt.legend()
plt.tight_layout()
plt.savefig("best_k_plot.png", dpi=150)
plt.show()

print("\nBest K I found:", best_k)

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

output = knn.predict(X_test)

acc = accuracy_score(y_test, output)
f1  = f1_score(y_test, output, average="weighted")

print("\nModel Results:")
print(f"  Accuracy : {acc * 100:.2f}%")
print(f"  F1 Score : {f1:.4f}")
print("\nFull Report:")
print(classification_report(y_test, output, target_names=raw.target_names))

mat = confusion_matrix(y_test, output)

plt.figure(figsize=(7, 5))
sns.heatmap(mat, annot=True, fmt="d", cmap="YlGnBu",
            xticklabels=raw.target_names, yticklabels=raw.target_names,
            linewidths=0.5, linecolor="white")
plt.title("Confusion Matrix", fontsize=12)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_mat.png", dpi=150)
plt.show()

sample = np.array([[5.9, 3.0, 5.1, 1.8]])
sample_scaled = sc.transform(sample)
guess = knn.predict(sample_scaled)
names = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
print("\nTesting with a new flower sample:")
print(f"  Input  : {sample[0]}")
print(f"  Output : {names[guess[0]]}")

print("\n--- Project 2 Done ---")
print(f"Algorithm : KNN  |  Best K : {best_k}  |  Accuracy : {acc*100:.2f}%  |  F1 : {f1:.4f}")
