# Wine Dataset Clustering Analysis

## Overview
This lab applies two clustering algorithms—**K-Means** and **K-Medoids (PAM)**—to the **Wine Dataset** from scikit-learn. The goal is to discover natural groupings in wine chemical properties and compare the performance of both methods using quantitative evaluation metrics.

---

## Dataset Description

### Wine Dataset Characteristics
- **Source:** UCI Machine Learning Repository (via sklearn.datasets)
- **Samples:** 178 wine instances
- **Features:** 13 chemical properties including:
  - Alcohol content
  - Malic acid
  - Ash
  - Alkalinity of ash
  - Magnesium
  - Total phenols
  - Flavanoids
  - Nonflavanoid phenols
  - Proanthocyanins
  - Color intensity
  - Hue
  - OD280/OD315 of diluted wines
  - Proline

### Class Distribution
- **Class 0 (Cultivar 1):** 59 samples (33.1%)
- **Class 1 (Cultivar 2):** 71 samples (39.9%)
- **Class 2 (Cultivar 3):** 48 samples (27.0%)

### Data Preprocessing
- **Standardization:** Applied z-score normalization using StandardScaler
  - Mean ≈ 0
  - Standard deviation ≈ 1
- **Rationale:** Ensures all features contribute equally to distance calculations, preventing features with larger scales from dominating the clustering process

---

## Methodology

### Step 1: Data Loading and Exploration
- Loaded Wine dataset using `sklearn.datasets.load_wine()`
- Created pandas DataFrame for analysis
- Verified data quality (no missing values)
- Examined feature distributions and correlations
- Applied standardization for clustering algorithms

### Step 2: K-Means Clustering
**Algorithm Details:**
- Implementation: sklearn.cluster.KMeans
- Number of clusters: k=3 (matching true wine classes)
- Initialization: k-means++ (20 runs)
- Random state: 42 (reproducibility)

**How K-Means Works:**
1. Randomly initializes k cluster centers
2. Assigns each point to nearest center
3. Recalculates centers as mean of assigned points
4. Repeats steps 2-3 until convergence

**Complexity:** O(n × k × i × d) where n=samples, k=clusters, i=iterations, d=dimensions

### Step 3: K-Medoids (PAM) Clustering
**Algorithm Details:**
- Implementation: Custom NumPy implementation
- Initialization: k-means++ adapted for medoids
- Number of clusters: k=3
- Distance metric: Euclidean

**How K-Medoids Works:**
1. Selects k actual data points as initial medoids
2. Assigns each point to nearest medoid
3. Swaps medoids with non-medoids to minimize total distance
4. Repeats until no improving swap exists

**Complexity:** O(k × (n-k)² × i) - significantly slower than K-Means for large datasets

### Step 4: Evaluation and Comparison
**Metrics Used:**
1. **Silhouette Score** (internal metric)
   - Measures cluster cohesion and separation
   - Range: -1 to +1 (higher is better)
   - Formula: (b - a) / max(a, b)
     - a = mean intra-cluster distance
     - b = mean nearest-cluster distance

2. **Adjusted Rand Index (external metric)**
   - Measures agreement with ground truth labels
   - Range: -1 to +1 (1 = perfect match, 0 = random)
   - Accounts for chance agreement

**Visualization:**
- Applied PCA (2 components) for 2D projection
- Created side-by-side comparison plots:
  - K-Means clusters with centroids
  - K-Medoids clusters with medoids
  - Ground truth class distribution

---

## Results

### K-Means Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | 0.281 | Moderate cluster separation; some overlap between groups |
| **Adjusted Rand Index** | 0.897 | Strong agreement with true wine classes (89.7% match) |
| **Cluster Sizes** | Varies | Relatively balanced distribution across clusters |

**Strengths Observed:**
- Fast convergence (typically < 10 iterations)
- Clear cluster boundaries in PCA projection
- High agreement with true wine cultivars
- Computationally efficient

**Limitations Observed:**
- Centroids are synthetic points (not actual wines)
- Moderate Silhouette indicates some cluster overlap
- Sensitive to initialization (mitigated by multiple runs)

### K-Medoids (PAM) Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Silhouette Score** | 0.254 | Slightly lower than K-Means; acceptable separation |
| **Adjusted Rand Index** | 0.821 | Good agreement with true classes (82.1% match) |
| **Cluster Sizes** | Varies | Similar distribution to K-Means |

**Strengths Observed:**
- Medoids are actual wine samples (interpretable)
- More robust to outliers in testing
- Stable results across multiple runs
- Works with any distance metric

**Limitations Observed:**
- Slower computation (noticeable even with 178 samples)
- Slightly lower performance metrics
- More complex implementation

### Head-to-Head Comparison

| Aspect | K-Means | K-Medoids | Winner |
|--------|---------|-----------|--------|
| **Silhouette Score** | 0.281 | 0.254 | K-Means (+0.027) |
| **Adjusted Rand Index** | 0.897 | 0.821 | K-Means (+0.076) |
| **Computation Time** | ~0.05s | ~2.1s | K-Means (42× faster) |
| **Interpretability** | Moderate | High | K-Medoids |
| **Outlier Robustness** | Low | High | K-Medoids |
| **Center Type** | Synthetic | Real data | K-Medoids |

**Overall Winner:** K-Means for this dataset (better metrics, faster)

---

## Insights and Interpretation

### Why K-Means Performed Better
1. **Clean Dataset:** Wine dataset has minimal outliers or noise
2. **Spherical Clusters:** Chemical properties form relatively compact, spherical groups
3. **Euclidean Distance:** Works well for standardized chemical measurements
4. **Well-Separated Classes:** Three wine cultivars have distinct chemical profiles

### When Each Algorithm Excels

**Use K-Means when:**
- ✓ Dataset is large (>1000 samples)
- ✓ Clusters are roughly spherical and similar in size
- ✓ Data is clean with minimal outliers
- ✓ Speed is critical
- ✓ Using Euclidean distance metric

**Use K-Medoids when:**
- ✓ Dataset contains outliers or noise
- ✓ Need actual data points as cluster representatives
- ✓ Interpretability is important (e.g., "representative wines")
- ✓ Using non-Euclidean distance metrics
- ✓ Dataset is small-to-medium (<5000 samples)
- ✓ Robustness more important than speed

### Real-World Applications for Wine Dataset

**K-Means Results Could Be Used For:**
- Quality control: Identifying wines that deviate from their cultivar profile
- Pricing strategies: Grouping wines by similar chemical compositions
- Product recommendations: Suggesting wines with similar characteristics

**K-Medoids Results Could Be Used For:**
- Creating "reference wines" for each cluster (actual samples)
- Training wine tasters with representative examples
- Quality benchmarking using real wine samples as standards

---

## Cluster Characteristics Analysis

### Cluster Size Distribution

**K-Means Clusters:**
- Cluster 0: 62 samples (34.8%)
- Cluster 1: 51 samples (28.7%)
- Cluster 2: 65 samples (36.5%)

**K-Medoids Clusters:**
- Cluster 0: 59 samples (33.1%)
- Cluster 1: 48 samples (27.0%)
- Cluster 2: 71 samples (39.9%)

**Comparison with True Classes:**
- Both algorithms captured the 3-cluster structure
- K-Means achieved slightly better alignment with actual cultivars
- K-Medoids cluster sizes closely match true class distribution

### Visual Analysis (PCA Projection)

**Observations from 2D Plots:**
1. **Cluster Overlap:** Some overlap between classes, especially in compressed 2D space
2. **Boundary Clarity:** K-Means shows slightly tighter clusters
3. **Center Positioning:** K-Means centroids vs K-Medoids actual wine samples clearly visible
4. **Ground Truth Alignment:** Both methods capture major class boundaries

**Note:** PCA explains ~55% of variance, so some separation is lost in visualization

---

## Challenges and Solutions

### 1. Library Compatibility Issues
**Problem:** K-Medoids libraries (sklearn-extra, pyclustering) incompatible with NumPy 2.x in some environments

**Solution:** Implemented custom PAM algorithm using pure NumPy:
- K-means++ initialization adapted for medoids
- Efficient swap operation testing
- Early stopping when no improvement found

### 2. High-Dimensional Visualization
**Problem:** 13-dimensional data difficult to visualize directly

**Solution:** Applied PCA dimensionality reduction to 2D:
- Preserves major variance patterns
- Enables intuitive scatter plots
- Maintains relative cluster positions

### 3. Metric Interpretation
**Problem:** Understanding trade-offs between different evaluation metrics

**Solution:** Used complementary metrics:
- **Silhouette:** Internal quality (no labels needed)
- **ARI:** External validation (compares to ground truth)
- Both provide different perspectives on clustering quality

### 4. Random Initialization Variability
**Problem:** Different runs produced slightly different results

**Solution:** 
- Set `random_state=42` for reproducibility
- Used k-means++ for smarter initialization
- K-Means: 20 independent runs (best selected automatically)

---

## Conclusions

### Key Findings
1. **Both algorithms successfully identified wine groups** based on chemical properties
2. **K-Means achieved better quantitative metrics** (Silhouette: 0.281, ARI: 0.897)
3. **K-Medoids provided more interpretable results** (real wine samples as centers)
4. **Speed difference significant:** K-Means 42× faster than K-Medoids
5. **High ARI scores** (>0.8) indicate strong alignment with true cultivar labels

### Practical Recommendations

**For Wine Industry Applications:**
- Use **K-Means** for large-scale wine classification and quality control
- Use **K-Medoids** when identifying representative wine samples for tasting panels or quality benchmarks
- Consider ensemble approaches: K-Means for efficiency, K-Medoids for validation

**For Machine Learning Practice:**
- K-Means is excellent for initial exploratory analysis
- K-Medoids useful for understanding cluster centers in business context
- Always evaluate with multiple metrics (internal + external)

### Future Enhancements
1. **Optimal k selection:** Implement elbow method and silhouette analysis
2. **Feature importance:** Determine which chemical properties drive clustering
3. **Hierarchical clustering:** Compare with dendrogram-based approaches
4. **DBSCAN:** Test density-based clustering for noise handling
5. **Cross-validation:** Multiple train/test splits for robust evaluation

---

## Technical Implementation

### Environment and Dependencies
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
```

### Code Organization
1. **Step 1:** Data loading and preprocessing
2. **Step 2:** K-Means clustering implementation
3. **Step 3:** K-Medoids (PAM) implementation
4. **Step 4:** Visualization and comparison
5. **Step 5:** Results analysis and reporting

### Reproducibility
- Random seed: 42 (all stochastic operations)
- scikit-learn version: 1.3+
- NumPy version: 1.24+ (compatible with 2.x)
- Python version: 3.8+

---

 

## References

### Academic Sources
1. **MacQueen, J. (1967).** "Some methods for classification and analysis of multivariate observations." *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability*, 1(14), 281-297.

2. **Kaufman, L., & Rousseeuw, P. J. (1990).** *Finding Groups in Data: An Introduction to Cluster Analysis*. John Wiley & Sons.

3. **Rousseeuw, P. J. (1987).** "Silhouettes: A graphical aid to the interpretation and validation of cluster analysis." *Journal of Computational and Applied Mathematics*, 20, 53-65.

4. **Hubert, L., & Arabie, P. (1985).** "Comparing partitions." *Journal of Classification*, 2(1), 193-218.

### Technical Documentation
- **Scikit-learn:** https://scikit-learn.org/stable/modules/clustering.html
- **Wine Dataset:** https://archive.ics.uci.edu/ml/datasets/wine
- **NumPy Documentation:** https://numpy.org/doc/stable/

---

### Silhouette Score Calculation
For each sample i:
- a(i) = average distance to other points in same cluster
- b(i) = average distance to points in nearest other cluster
- s(i) = (b(i) - a(i)) / max(a(i), b(i))

Overall Silhouette = mean of s(i) across all samples

**Interpretation:**
- s(i) ≈ 1: Sample well-matched to its cluster
- s(i) ≈ 0: Sample on border between clusters
- s(i) ≈ -1: Sample likely in wrong cluster

### Adjusted Rand Index Calculation
- Measures similarity between predicted and true labels
- Adjusts for chance: ARI = (RI - Expected_RI) / (max_RI - Expected_RI)
- Ranges from -1 to 1, with 0 indicating random labeling

**Interpretation:**
- ARI = 1.0: Perfect match
- ARI = 0.0: Random labeling
- ARI < 0: Worse than random

---

