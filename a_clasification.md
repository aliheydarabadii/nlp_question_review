# Summary: Classifying Text

This document summarizes a presentation on text classification, covering foundational machine learning concepts, feature extraction from text, a deep dive into linear classification models, and methods for evaluation.

## 1. Quick Revision: Supervised Machine Learning

### Definition of Machine Learning
Machine Learning (ML) encompasses techniques that enable machines to "act more intelligent" by generalizing from past data to predict future data.

> **Standard Definition (Tom M. Mitchell):** "A computer program is said to learn from **experience E** with respect to some class of **tasks T** and a **performance measure P**, if its performance at tasks in T, as measured by P, improves because of experience E.”

### The Classification Task
- **Goal:** To automatically categorize items into predefined classes.
- **Example (Fish Sorting):** A system is built to automatically sort fish on a conveyor belt into different types (e.g., bass, tuna).
    1.  **Sensors:** A camera captures images of the fish.
    2.  **Feature Extraction:** From each image, relevant **features** are computed (e.g., `X₁`: length, `X₂`: width, `X₃`: colour).
    3.  **Learn Rules:** A model learns rules from historical (labeled) data to classify new fish.
        - `if (X₁/X₂ < 1.4 and X₃=blue) => bass`

### Supervised Learning
In supervised learning:
- Each training instance (e.g., a fish) is represented as a **vector** in a **feature space**.
- Each training instance is **labeled** with a ground-truth class.
- The **task** is to **partition the feature space** to make accurate predictions for new, unlabeled instances.
- In practice, data often overlaps, and classes may not be linearly separable.

### Training a Model
- **Model:** A parameterized formula (e.g., `y = θᵀx`) used to predict labels for new instances. The parameters are denoted by `θ`.
- **Learning Algorithm:**
    - Takes **training instances** and their **ground truth labels** as input.
    - Searches for the optimal model **parameters** (`θ*`) that minimize a **prediction error (loss)** on the training labels.
    - The learning algorithm itself has settings called **hyper-parameters**.

### Preventing Overfitting
- **Overfitting:** A model learns the training data too well, including spurious patterns (noise), which causes it to perform poorly on new, unseen data (poor generalization).
- **Hyper-parameters** (e.g., model complexity, learning rate) can control the trade-off between fitting the training data and generalizing.
- **Problem:** We can't use the training set to tune hyper-parameters (as it would lead to overfitting) and we can't use the test set (as it must be reserved for final, unbiased evaluation).
- **Solution: Validation Set**
    - The original training data is split into a **training set** and a **validation set**.
    - Models are trained on the training set with different hyper-parameter settings.
    - Each trained model is evaluated on the validation set.
    - The hyper-parameter settings that yield the best performance on the validation set are chosen for the final model.



## 2. Text Classification

### What is Text Classification?
The process of training a model to classify documents into predefined categories. Examples include:
- Science Fiction vs. Romance
- Spam vs. Not Spam
- Positive vs. Negative Review

### Common Applications
Text classification is an extremely common task:
- **Spam/phishing detection**
- **Authorship identification** (e.g., who is Satoshi Nakamoto?)
- **Sentiment analysis**
- **Offensive content detection**
- **Personalized news feeds**
- **Routing customer emails** to the correct department.

### Types of Text Classification Problems
1.  **Binary Classification:** Output is one of two classes (e.g., `POSITIVE`/`NEGATIVE`).
2.  **Multi-class Classification:** Output is one of N classes (e.g., `REPAIRS_DEPT`, `SALES_DEPT`).
3.  **Ordinal Regression:** Output is an ordered category (e.g., star ratings `1_STAR`, `2_STAR`...).
4.  **Multi-label Classification:** Output is a set of categories from N possible classes (e.g., a news article can be about `POLITICS` and `SPORT`).

## 3. Extracting Features from Text

### The Need for Feature Extraction
- Text documents are of **arbitrary length**.
- Machine learning models require a **fixed-size vector** as input.
- Therefore, **features** must first be extracted from the text to create this vector.

### Bag-of-Words (BOW) Representation
This is the most common approach for representing text for classification.
- **Core Idea:** The vocabulary of a document provides the most useful signal for its category. The order of words is ignored.
- **Process:**
    1.  A vocabulary is defined (all unique words in the entire collection of documents).
    2.  Each document is represented as a vector where each dimension corresponds to a word in the vocabulary.
    3.  The value in each dimension is the **count** (or frequency) of that word in the document.
- **Characteristics:**
    - **Ignores word order:** "the cat sat on the mat" and "the mat sat on the cat" have the same BOW representation. (This can be partially mitigated by using n-grams like "on the").
    - **Very Sparse:** The resulting vectors are very long (size of the vocabulary) but contain mostly zeros, since any given document only uses a small subset of the total vocabulary.

### Aside: Statistical Laws of Text (Heap's & Zipf's)
These laws explain why BOW is feasible.
- **Heap's Law:** The vocabulary size `V` grows slowly with the length of the document collection `l`. Specifically, `V(l) ∝ l^β` where `β` is typically around 0.5. This means the number of features (vocabulary size) doesn't explode uncontrollably.
- **Zipf's Law:** A few words are extremely common, while most words are very rare. The frequency of a word is inversely proportional to its rank.

### Challenges of BOW
- The number of features (vocabulary size) is often **much larger than the number of documents (examples)**.
- This "curse of dimensionality" means that multiple parameter settings can explain the data, making overfitting a high risk.
- **Strong regularization** is needed to guide the learner.

## 4. Tokenizing & Preprocessing

### Tokenization
The process of breaking down a string of text into a sequence of tokens (words, punctuation, etc.).
- **Simple:** Use a regular expression like `\b\w\w+\b` to find sequences of alphanumeric characters.
- **Advanced (e.g., NLTK):** Use more complex regex to handle abbreviations (`U.S.A.`), currency (`$12.40`), and hyphens.

### Common Pre-processing Steps
1.  **Prior to Tokenization:**
    - **Remove mark-up:** Get rid of HTML tags, etc.
    - **Lowercase:** Reduces vocabulary size (e.g., 'The' and 'the' become the same).
    - **Remove punctuation.**
2.  **After Tokenization:**
    - **Remove stopwords:** Extremely common words ('a', 'the', 'is') that carry little semantic meaning.
    - **Remove very low-frequency words:** Words that appear too rarely to provide a reliable signal.
    - **Stemming/Lemmatization:** Reduce words to their root form ('running' -> 'run').
    - **Spelling Correction.**

### In-depth Example: Probabilistic Spelling Correction
This serves as a great introduction to the Naive Bayes model.
- **Task:** Given an observed, misspelled word (e.g., "acress"), find the most likely intended word (e.g., "actress").
- **Approach:** Use **Bayes' Rule**. We want to find the `correct` word that maximizes `P(correct | observed)`.

`P(correct | observed) ∝ P(observed | correct) * P(correct)`

- **`P(correct)` (The Prior):** How common is the candidate word in the language? Estimated from its frequency in a large corpus.
- **`P(observed | correct)` (The Likelihood):** How likely is this specific typo to occur? Estimated from a *confusion matrix* that counts historical error types (e.g., how often is 'e' substituted for 'o'?). This is also known as the **Noisy Channel Model**.
- **Context is Key:** The model might predict "acres" is more likely than "actress". By including context (e.g., the preceding word "versatile"), we can improve the prediction. This leads directly to the Naive Bayes classifier.
    - We replace `P(correct)` with `P(bigram)` (e.g., `P("versatile actress")`).

## 5. Linear Classification Models

For high-dimensional BOW data, **linear models** are often sufficient and highly effective.

### Decision Boundaries: Hyperplanes
Linear classifiers learn a **hyperplane** that separates the classes in the feature space.
- The equation of the hyperplane is `θ·x - b = 0`.
- `θ` is a vector orthogonal to the hyperplane, defining its orientation.
- `b` is an offset, indicating its distance from the origin.

### A. Multinomial Naïve Bayes (NB)
- **Core Assumption (The "Naïve" Part):** Features (word occurrences) are **statistically independent** of each other, given the class label.
    - `P(word1, word2 | spam) = P(word1 | spam) * P(word2 | spam)`
- **How it Works:** It uses Bayes' rule and the independence assumption to calculate the probability of a class given the words in a document. The class with the highest probability is chosen.
- **Smoothing:** To avoid zero probabilities for words not seen in the training data for a given class, a small pseudo-count `α` is added (e.g., Laplace/add-one smoothing where `α=1`).
- **Pros:**
    - Very **fast** to train (one pass over data).
    - Works well even with little data.
- **Cons:**
    - The independence assumption is almost always false.
    - Predicted probabilities are often **overconfident** and not well-calibrated.

### B. Logistic Regression (LR)
- **Core Idea:** Models the probability of a class directly.
- **How it Works:**
    1.  Calculates the signed distance from the decision boundary (hyperplane): `s(x) = θ·x - b`.
    2.  Uses the **logistic (sigmoid) function** to map this distance (from -∞ to +∞) to a probability (from 0 to 1).
        - `sigmoid(s) = 1 / (1 + e⁻ˢ)`
- **Pros:**
    - Produces **well-calibrated probability estimates**.
    - Trains efficiently and scales well.
- **Cons:**
    - Assumes feature values are linearly related to the *log-odds* of the class.

### C. Support Vector Machines (SVM)
- **Core Idea:** Find the hyperplane that **maximizes the margin** between the two classes. The margin is the "street" between the closest points of each class.
- **Support Vectors:** The data points that lie exactly on the margin. These are the only points that define the position of the hyperplane.
- **Loss Function: Hinge Loss**
    - `L(x,y) = max(0, 1 - y(w·x))`
    - This is the key difference from LR. Hinge loss only penalizes points that are on the wrong side of the margin or inside it. Correctly classified points outside the margin have **zero loss**.
- **Regularization:** The objective function balances maximizing the margin with minimizing classification errors (for non-separable data). A hyper-parameter `C` controls this trade-off.
- **Summary:** While both LR and SVM are linear classifiers, they use different loss functions.
    - **LR** penalizes all points based on their probability of being misclassified.
    - **SVM** focuses only on the points near the boundary (the support vectors).

## 6. Evaluating Text Classifiers

### The Confusion Matrix & Key Metrics (Binary)
After training, the model is evaluated on a held-out **test set**.
- **True Positive (TP):** Correctly predicted positive.
- **False Positive (FP):** Incorrectly predicted positive.
- **True Negative (TN):** Correctly predicted negative.
- **False Negative (FN):** Incorrectly predicted negative.

From these, we derive:
- **Accuracy:** `(TP + TN) / Total` (Correct predictions / All predictions)
- **Precision:** `TP / (TP + FP)` (Of all positive predictions, how many were correct?)
- **Recall:** `TP / (TP + FN)` (Of all actual positives, how many did we find?)
- **F1-Measure:** `2 * (Precision * Recall) / (Precision + Recall)` (Harmonic mean of precision and recall, useful when they are in tension).
- **AuC (Area under ROC Curve):** A single metric that evaluates the classifier across all possible confidence thresholds.

### Evaluating Multi-class Classifiers
- The confusion matrix becomes `n x n` for `n` classes.
- Precision and recall are calculated for each class in a one-vs-all manner.
- These per-class scores are combined into a single measure:
    - **Macro-average:** Average the metric (e.g., precision) across all classes, giving each class **equal weight**.
    - **Micro-average:** Aggregate the counts (TP, FP, etc.) across all classes and then compute the metric once. This gives more weight to **more populous classes**.

## 7. Conclusions
- The traditional and highly effective approach to text classification involves:
    1.  Representing text documents using a **bag-of-words (BOW)** model.
    2.  Applying a **linear classifier** to the resulting high-dimensional, sparse vectors.
- **Support Vector Machines (SVMs)** and **regularized Logistic Regression** are excellent choices for this task and often serve as strong baselines.
