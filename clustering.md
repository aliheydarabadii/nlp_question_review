# Exam Review: Clustering Text

This guide provides a complete summary of the key concepts, algorithms, and applications related to text clustering.

## 1. Introduction to Text Clustering

### What is Clustering?
Clustering is an **unsupervised** machine learning task. This is a critical distinction from classification, which is supervised.
- **Goal:** To automatically group a set of objects (e.g., documents) into coherent subgroups, called **clusters**.
- **"Unsupervised" means:** We do **not** have pre-defined labels for the documents. The algorithm discovers the groupings on its own based on the data's inherent structure.

### Why Cluster Text?
The goal is to find documents that are similar based on some property. Common applications include:
- **Topic Modeling:** Grouping news articles by topic (e.g., sports, politics, technology). This is the most common use case.
- **Author Identification:** Grouping documents written by the same author.
- **Sentiment/Emotion Analysis:** Grouping customer reviews by emotion (e.g., angry, happy, disappointed).
- **Security:** Detecting phishing campaigns by clustering malicious emails that share similar features.
- **Search Result Organization:** Grouping search results to provide a summarized overview (like Google News).
- **De-duplication:** Finding and grouping identical or near-identical documents in a large collection (e.g., a web crawl).

## 2. Representing Documents & Measuring Similarity

To cluster documents, we first need a way to represent them numerically and then a way to measure how similar two representations are.

### Document Representation: TF-IDF Vectors
- Documents are traditionally represented as **tf-idf vectors**.
- This is a more sophisticated version of the **bag-of-words** model.
- Each dimension in the vector corresponds to a word in the vocabulary.
- The value in that dimension is the **tf-idf score** of the word, which reflects:
    - **TF (Term Frequency):** How often the word appears in the document.
    - **IDF (Inverse Document Frequency):** How rare the word is across the entire collection.
- **Key Property:** These vectors are very high-dimensional and **sparse** (most values are zero).

### Similarity Measure: Cosine Similarity
- **Problem:** Longer documents naturally have more words and higher TF values, but this doesn't mean they are "more similar". We need a measure that is independent of document length.
- **Solution:** **Cosine Similarity**.
- **Concept:** It measures the cosine of the angle between two vectors. It effectively measures the orientation of the vectors, not their magnitude.
- **Formula:**
  `sim(d1, d2) = (d1 ⋅ d2) / (||d1|| * ||d2||)`
- **In practice:** This is equivalent to taking the dot product of the two vectors after they have been normalized to have a length of 1. A similarity of 1 means the vectors point in the same direction (very similar content), while a similarity of 0 means they are orthogonal (no shared content).

## 3. Clustering Algorithms

These are different methods for grouping the document vectors.

### 3.1. K-Means Clustering

This is the most common and "go-to" clustering algorithm.

- **How it Works:**
    1.  **Initialize:** Randomly choose `k` initial cluster centers, called **centroids**. A centroid is the average of all points in a cluster.
    2.  **Assign:** Assign each document (data point) to its nearest centroid. **Euclidean distance** is typically used to measure "nearness".
    3.  **Recompute:** Recalculate the position of each centroid by taking the mean of all points assigned to it.
    4.  **Repeat:** Repeat steps 2 and 3 until the centroids no longer move significantly (the algorithm has converged).
- **Objective:** The algorithm's goal is to minimize the overall **variance** within clusters (the sum of squared distances from each point to its cluster's centroid).
- **Advantages:**
    - Simple, fast, and robust.
    - Scales well to large datasets because it doesn't compute all pairwise distances.
- **Disadvantages & Problems:**
    - **You must specify `k` (the number of clusters) in advance.** Choosing the right `k` is critical.
    - **Sensitive to initial centroid placement:** Can converge to a **local minimum**, resulting in a suboptimal clustering. (To mitigate this, run the algorithm multiple times with different random initializations).
    - **Assumes globular (spherical) clusters:** Struggles with clusters that are elongated or have complex shapes.
    - **Uses Euclidean distance:** This is not ideal for high-dimensional, sparse text data where cosine similarity is preferred.

### 3.2. K-Medoids Clustering

A variation of K-Means that addresses some of its weaknesses.

- **Key Difference:** Instead of a **centroid** (which is an abstract average point), K-Medoids uses a **medoid** to represent the cluster center.
- **Medoid:** A medoid is an **actual data point** in the cluster that is most central (has the smallest average distance to all other points in the cluster).
- **Advantages:**
    - **Works with any distance metric:** You can use cosine similarity, which is better for text.
    - The cluster center is a real document, making it more **interpretable**.
- **Disadvantages:**
    - **Much higher computational complexity:** It is significantly slower than K-Means because it needs to compute pairwise distances between points to find the new medoid at each step (O(n²)).

### 3.3. Agglomerative Hierarchical Clustering

This method builds a hierarchy of clusters from the bottom up.

- **How it Works:**
    1.  Start by assigning each document to its own cluster (N documents = N clusters).
    2.  Find the two **most similar** clusters and **merge** them into one.
    3.  Repeat step 2 until only one single cluster (containing all documents) remains.
- **Output:** The result is a tree-like structure called a **dendrogram**, which visualizes the hierarchy of merges.
- **Linkage Criteria:** To merge clusters, we need to define the distance between them. The choice of linkage criteria is crucial and affects the final cluster shapes.
    - **Complete-linkage:** Distance is the **maximum** distance between any two points in the two clusters. Tends to produce tight, globular clusters.
    - **Single-linkage:** Distance is the **minimum** distance between any two points in the two clusters. Can find long, thin, chain-like clusters.
    - **Average-linkage:** Distance is the **average** distance between all pairs of points. A good compromise.
- **Advantages:**
    - Does not require you to specify the number of clusters (`k`) in advance.
    - The dendrogram provides rich information about the data's structure.
- **Disadvantages:**
    - **High time complexity:** Unsuitable for large datasets.

### 3.4. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)

This algorithm groups points based on density.

- **How it Works:** DBSCAN defines clusters as continuous regions of high density. It classifies each point as one of three types:
    - **Core Point:** A point that has at least `minPoints` neighbors within a radius of `ε` (epsilon).
    - **Border Point:** A point that is within `ε` of a core point but doesn't have enough neighbors to be a core point itself.
    - **Noise Point:** A point that is neither a core nor a border point (an outlier).
- **Key Parameters:** You must specify `ε` and `minPoints`. These together define what "high density" means.
- **Advantages:**
    - **Does not require `k` to be specified.** The number of clusters is inferred from the data.
    - **Can find arbitrarily shaped clusters** (not just spheres).
    - **Robust to outliers**, which are identified as noise.
- **Disadvantages:**
    - Performance is **highly sensitive** to the choice of `ε` and `minPoints`.
    - Struggles with clusters that have **different densities**.

### Algorithm Comparison Table

| Feature                    | K-Means                                    | K-Medoids                                | Hierarchical                          | DBSCAN                                      |
| -------------------------- | ------------------------------------------ | ---------------------------------------- | ------------------------------------- | ------------------------------------------- |
| **Needs `k`?**             | **Yes**                                    | **Yes**                                  | No                                    | No                                          |
| **Cluster Shape**          | Globular (Spherical)                       | Globular                                 | Any (depends on linkage)              | **Arbitrary**                               |
| **Handles Outliers?**      | No (outliers pull on centroids)            | Better than K-Means, but still sensitive | No                                    | **Yes** (classifies them as noise)          |
| **Computational Cost**     | **Low / Scalable**                         | High (O(n²))                             | High (O(n²) to O(n³))                 | Medium (O(n log n) with index)              |
| **Metric Restriction**     | Euclidean (typically)                      | **Any distance metric**                  | **Any distance metric**               | **Any distance metric**                     |
| **Key Idea**               | Centroid-based (average point)             | Medoid-based (real point)                | Bottom-up merging (dendrogram)        | Density-based (core, border, noise)         |

## 4. Topic Modeling (e.g., LDA)

Topic modeling is an advanced form of clustering that provides a richer, more nuanced representation of documents.

### What is a Topic Model?
- It's a form of **soft clustering**: each document can belong to **multiple topics** to different degrees.
- It provides a **low-dimensional representation** of documents. Instead of a sparse vector with thousands of dimensions (one for each word), a document is represented as a dense vector of, say, 100 dimensions (one for each topic).
- **Topic:** A topic itself is defined as a **probability distribution over words**. For example, a "Sports" topic would have high probabilities for words like "ball", "game", "team", "score".

### How does it work? Matrix Decomposition
- Conceptually, topic modeling is a form of **matrix decomposition**.
- It takes the large, sparse **`Terms × Documents`** matrix and approximates it as the product of two smaller, dense matrices:
  **`Terms × Topics`** and **`Topics × Documents`**.
- This significantly reduces the number of parameters to estimate and reveals the latent topic structure.

### Latent Dirichlet Allocation (LDA)
- **LDA is the most famous topic modeling algorithm.**
- **Generative Story:** LDA assumes documents are created via a probabilistic process:
    1. For each document, choose a mixture of topics (e.g., 70% sports, 30% business).
    2. For each word in that document, first pick a topic from that mixture, then pick a word from that topic's word distribution.
- The algorithm's job is to work backward from the observed documents to infer the hidden topic structures.

### Why is Topic Modeling Useful?
- **Solves key problems of bag-of-words:**
    - **Synonymy:** Words like "cancer" and "oncology" will be grouped into the same medical topic, so the model learns they are related.
    - **Polysemy:** The model can learn that the word "bank" can belong to a "Finance" topic and a "Geography" topic, resolving its ambiguity based on context.
- **Better Document Representation:** Creates dense, meaningful vectors that are better for calculating similarity.
- **Visualization:** Excellent for visualizing the thematic structure of a large collection of documents.