# Exam Review: Searching Text (Information Retrieval)

This guide provides a complete summary of the key concepts, algorithms, and methodologies for searching text, also known as Information Retrieval (IR).

## 1. Core Concepts of Information Retrieval

### What is Information Retrieval (IR)?
IR is the task of finding content (documents, images, etc.) that is **relevant** to a user's **information need**.

- **Information Need:** A high-level goal or question a user has.
  - *Example:* "I want to make the delicious banana and apple pancakes I saw a picture of."
- **Query:** The user translates their need into a set of **keywords** to enter into a search engine.
  - *Example:* `"banana apple pancakes recipe"`
- **The Core IR Problem:** How to rank a massive collection of documents to find the few that best satisfy the user's query.

### The Challenge of Search
Simple keyword matching isn't enough because:
1.  **Too many documents** might contain a common query term (e.g., "apple").
2.  **No single document** might contain all the query terms (e.g., a recipe might be called "Banana Oat Pancakes" but still be relevant).
3.  Some terms are more important (**discriminative**) than others.

Therefore, we need a system that can **score and rank** documents based on how well they match the query.

### Is IR just Text Classification?
No. While you could frame it as classifying a (query, document) pair as "relevant" or "not relevant," this is computationally infeasible for a large-scale search engine.
- A classifier would need to iterate over **every document** in the collection for **every query**, which is far too slow.
- The number of features for all possible query-document word interactions would be enormous.
- The core of IR relies on **fast indexing** and efficient scoring, not direct classification.

## 2. Lexical Retrieval & Scoring Models

This is the "classic" approach to search, based on matching words. The goal is to assign a score to each document for a given query.

### a) Inverse Document Frequency (IDF)

- **Intuition:** Words that are **rare** across the entire collection are more informative and should be given more weight. A document matching the term "photosynthesis" is much more specific than one matching "the".
- **Concept:** IDF is based on the probability of a random document containing a term. The less probable it is, the higher its information content (and score).
- **Document Frequency (df_t):** The number of documents in the collection that contain term `t`.
- **Formula:**
  `idf_t = log(N / df_t)`
  - `N` = Total number of documents in the collection.
  - `df_t` = Document frequency of term `t`.

### b) Term Frequency (TF)

- **Intuition:** If a query term appears **multiple times** in a single document, that document is more likely to be about that term and thus more relevant.
- **Problem with Linear TF:** Simply using the raw count (`tf_t,d`) is problematic. The difference in relevance between 1 and 2 occurrences is significant, but the difference between 20 and 21 occurrences is not. The score should not increase linearly forever.
- **Solution (Log Frequency Weighting):** We "dampen" the term frequency by taking its logarithm.
  - Common variant: `log(1 + tf_t,d)`

### c) TF-IDF Weighting

TF-IDF combines both concepts to create a powerful term weighting scheme. The weight of a term `t` in a document `d` is high if the term appears often in that document (high TF) AND is rare in the overall collection (high IDF).

- **Full TF-IDF Score (for a document `d` and query `q`):**
  `score(q, d) = Σ [ log(1 + tf_t,d) * log(N / df_t) ]`  (summed over all terms `t` in the query `q`)

### d) Length Normalization & The Vector Space Model

- **Problem:** Longer documents are more likely to contain query terms by chance and will have higher raw TF scores. We need to **normalize for document length**.
- **Vector Space Model:** We treat documents and queries as vectors in a high-dimensional space where each dimension is a word in the vocabulary. The value in each dimension is the TF-IDF weight of that word.
- **Solution: Cosine Similarity:**
  - Instead of just summing scores, we calculate the **cosine of the angle** between the query vector and each document vector.
  - This measures the **orientation** (topical similarity) of the vectors, not their magnitude (length).
  - A score of 1 means they are perfectly aligned; 0 means they are unrelated.
  - This is achieved by normalizing the vectors by their **L2 norm** (Euclidean length).

### e) Okapi BM25: The Go-To Formula

BM25 (Best Match 25) is a state-of-the-art ranking function that improves upon standard TF-IDF. It is the default, highly effective method for term-based retrieval.

- **Key Features:**
    1.  **Smarter TF Saturation:** It includes a term frequency saturation component. The score from TF increases quickly at first and then **asymptotes**, preventing documents with an extreme number of term occurrences ("keyword stuffing") from dominating the results.
    2.  **Smarter Length Normalization:** It uses a more sophisticated normalization that compares a document's length to the average document length in the collection.
- **Parameters:** It has tunable parameters, `k1` (controls TF saturation) and `b` (controls length normalization), which can be optimized on a validation set.

## 3. Indexing & Crawling

For a search engine to be fast, it cannot scan every document for every query. It relies on pre-built data structures called indices.

### Inverted Indices
- This is the **core data structure** of a search engine.
- It is a mapping from **terms (TermIDs) to a list of documents (DocIDs)** that contain them. This list is called a **Posting List**.
- To answer a query like "banana apple", the system retrieves the posting list for "banana" and the posting list for "apple" and finds their intersection (the documents that are in both lists).
- **Optimization:**
    - **Early Termination:** Posting lists are often sorted by a static quality score (like PageRank) or by term frequency, allowing the system to stop searching after finding enough high-quality results.
    - **Index Pruning:** Techniques to remove low-quality documents from indices to save space and time.

### Positional Indices
- An enhancement to the inverted index where, for each document in the posting list, we also store the **positions** of the term (e.g., term "apple" appears at positions 7, 54, and 92 in doc 123).
- **Purpose:** This allows for **proximity search** (are the query words close together?) and **phrase search** (do the words appear in a specific sequence?).

### Web Crawling
- **Purpose:** To discover and download web pages to be included in the index.
- **Process:** A crawler starts with a set of seed URLs, downloads the pages, extracts any hyperlinks from them, and adds those new links to a queue of pages to visit.
- **Challenges:**
    - **Prioritizing URLs:** Deciding which pages to crawl first.
    - **Duplicate Detection:** Many URLs can point to the same or similar content.
    - **Robots.txt:** A file on a website that tells crawlers which parts of the site they are not allowed to access. Crawlers must respect this.

## 4. Learning to Rank (LTR)

Modern search engines go beyond a single scoring formula like BM25. They combine **hundreds of signals (features)** using machine learning. This process is called **Learning to Rank**.

### Why Rerank?
A simple lexical score isn't enough. We need to incorporate many other features, such as:
- **Multiple Retrieval Scores:** BM25 score, embedding-based similarity, etc.
- **Document Features:** PageRank (link authority), technical quality (load speed), title content, anchor text of incoming links.
- **User Features:** Personalized results based on click history, location, language.

LTR provides a principled way to **combine these diverse signals** into a single, optimized score.

### The Two-Stage Reranking Process
This is the standard architecture for modern search:
1.  **Stage 1: Retrieval (Fast)**
    - For a live query, use a fast initial ranker (like BM25) to retrieve a set of potentially relevant candidates (e.g., the top 100-1000 documents).
2.  **Stage 2: Reranking (Slow & Powerful)**
    - For this small set of candidates, compute the full, rich set of features.
    - Use a complex, pre-trained machine learning model (`f(x)`) to calculate a final, more accurate score for each candidate.
    - Sort the candidates by this new score to produce the final ranked list.

### Training a Reranking Model
The key is creating the training dataset.
1.  Take a large set of past queries.
2.  For each query, run the fast Stage 1 retrieval to get a candidate set.
3.  For each `(query, document)` pair in the candidate set, compute all the features (BM25 score, PageRank, etc.). This creates the **feature vector**.
4.  **Manual Labeling:** Employ human raters to assign a **relevance label** to each pair (e.g., 0=Irrelevant, 1=Fair, 2=Good, 3=Excellent).
5.  The final training set consists of `(feature_vector, relevance_label)` instances.

### LTR Loss Functions
The way the ML model is trained depends on its loss function.
- **Pointwise:** Treats each document independently. It tries to predict the exact relevance label (e.g., '2'). This is a simple regression problem, but it's **suboptimal** because it doesn't directly optimize the *order* of the list.
- **Pairwise:** Compares pairs of documents. The model learns to predict which one is more relevant. The loss is based on the number of incorrectly ordered pairs. This is better than pointwise.
- **Listwise:** The best approach. The model looks at the entire list of candidates at once and directly optimizes a list-based IR evaluation metric like **NDCG**.

### LambdaMART
- The most famous and successful LTR algorithm.
- It is a **listwise** rank learner that uses **boosted regression trees**.
- It is the default, high-performance baseline for most LTR applications.

## 5. Evaluating Search Results

### Gathering Relevance Judgments
- The ground truth labels for evaluation are crucial. They can be gathered by:
    - **Employing human raters** (e.g., Google's Quality Raters). This is expensive but high quality.
    - **Collecting judgments from users** (e.g., Wikipedia's surveys).
    - **Using click data is risky** as it creates a feedback loop (users click on what's ranked high, which then gets ranked higher).
    - **Using an LLM** to judge relevance is a modern, emerging approach.

### Key Evaluation Metrics
- **Precision at k (P@k):** What percentage of the top `k` results are relevant? Measures the quality of the first page of results.
- **Recall at k (R@k):** What percentage of *all possible relevant documents* were found in the top `k`?
- **Mean Average Precision (MAP):** The average of the P@k values calculated at the position of each relevant document. Rewards systems that rank relevant documents higher up.
- **Normalized Discounted Cumulative Gain (NDCG@k):**
    - The most important and widely used metric for modern search.
    - **Handles graded relevance** (e.g., a "good" result is better than a "fair" one).
    - **Discounts** the value of relevant documents that appear lower in the ranking, better reflecting user experience.