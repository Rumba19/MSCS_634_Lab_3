# Wine Dataset Clustering Lab

## Overview
This lab applies clustering techniques on the **Wine Dataset** from `sklearn`. Two clustering methods—**K-Means** and **K-Medoids (PAM)**—are implemented to explore grouping patterns in chemical properties of wines. Evaluation metrics such as **Silhouette Score** and **Adjusted Rand Index (ARI)** are used to measure cluster quality and compare both algorithms.

---

## Step 1: Load and Prepare the Dataset

### Process:
1. Loaded Wine Dataset using `sklearn.datasets.load_wine()`.
2. Created a pandas DataFrame for easier exploration.
3. Displayed dataset shape, sample records, and feature names.
4. Analyzed class distribution:
   - 3 classes (Wine Cultivars)
   - Balanced distribution: 59, 71, and 48 samples respectively
5. Standardized data using **z-score normalization** via `StandardScaler`.

### Key Observations:
- 13 numeric features, including alcohol, ash, magnesium, and color intensity.
- No missing or null values.
- Data standardized with mean ≈ 0 and std ≈ 1.

---

## Step 2: K-Means Clustering

### Process:
- Applied **K-Means** with `k=3` (reflecting the 3 actual wine classes).
- Computed cluster labels for all samples.
- Evaluated performance with **Silhouette Score** and **ARI**.

### Results Example (your values may vary):
- Silhouette Score ≈ **0.28**
- Adjusted Rand Index ≈ **0.89**

### Interpretation:
- The moderate Silhouette Score shows decent but not perfect separation.
- High ARI indicates clusters align well with true wine classes.
- K-Means effectively grouped wines based on chemical composition similarities.

---

## Step 3: K-Medoids (PAM)

### Process:
- Implemented **Partitioning Around Medoids (PAM)** using pure NumPy.
- Each cluster center is an **actual data point (medoid)** rather than an average.
- Used Euclidean distances for similarity calculation.

### Results Example:
- Silhouette Score ≈ **0.25**
- Adjusted Rand Index ≈ **0.82**

### Interpretation:
- Slightly lower metrics compared to K-Means.
- More robust to outliers, as medoids are actual observations.
- Computationally heavier but ensures cluster centers are real points.

---

## Step 4: Visualization and Comparison

### Visualizations:
- PCA reduced dataset to 2D for visualization.
- Side-by-side scatter plots:
  - **K-Means:** Cluster centroids marked with black X.
  - **K-Medoids:** Medoids marked with white X.
  - **Ground Truth:** Actual wine classes for reference.

### Comparison Summary:
| Metric | K-Means | K-Medoids | Better |
|--------|----------|------------|---------|
| Silhouette | ~0.28 | ~0.25 | K-Means |
| ARI | ~0.89 | ~0.82 | K-Means |

### Insights:
- K-Means produced slightly better-defined clusters.
- K-Medoids was more stable when tested with outlier perturbations.
- Visual plots show clusters overlap but capture major class boundaries.

---

## Step 5: Discussion and Conclusions

### Algorithm Characteristics:
- **K-Means:**
  - Pros: Fast, effective for well-separated spherical clusters.
  - Cons: Sensitive to outliers; centroids can be non-existent points.
- **K-Medoids:**
  - Pros: Robust to outliers; medoids are real data points.
  - Cons: Slower, especially for large datasets.

### When to Use:
- Use **K-Means** for large, clean datasets where speed matters.
- Use **K-Medoids** when interpretability and robustness to noise are required.

### Final Conclusions:
- Both algorithms correctly identified the main wine groupings.
- K-Means achieved better clustering efficiency and accuracy for this dataset.
- K-Medoids provided stronger interpretability and stability.

---

## Challenges Encountered
1. **NumPy Compatibility** – Some environments (e.g., Colab) use NumPy 2.x; K-Medoids libraries compiled for 1.x caused version errors. Fixed by using a NumPy-only PAM implementation.
2. **Cluster Visualization** – High-dimensional data required PCA projection for clear plotting.
3. **Metric Variability** – Random initialization affected Silhouette and ARI slightly; controlled using `random_state=42`.

---

## Files Included
- `WineClustering.ipynb` – full notebook implementation
- `README.md` – summary and insights report
- PCA scatter plots for both clustering algorithms (optional visualization outputs)

---

## References
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- Kaufman & Rousseeuw (1990), *Finding Groups in Data: An Introduction to Cluster Analysis*.
- Tan, Steinbach, & Kumar (2018), *Introduction to Data Mining, 2nd Edition*.

