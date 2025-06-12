# Exam Review: Language Models and Word Embeddings

This guide provides a complete summary of Language Models (LMs) and Word Embeddings, covering the core concepts, classic models, neural network approaches, and evaluation methods.

## 1. Language Modeling

### What is a Language Model?
A statistical language model is a **probability distribution over sequences of words**.
- **Core Task:** To answer the question, "What is the probability of this sequence of words occurring?"
- **Primary Use:** To predict the **next word** in a sequence, given the previous words.
- Because they can predict the next word, LMs are fundamentally **general-purpose text generators**.

### N-Gram Language Models (Markov Models)

This is the classic, count-based approach to language modeling.

- **The Markov Assumption:** To simplify the problem of predicting the next word, we assume its probability only depends on a **fixed number of previous words**. This fixed window is the "n" in "n-gram".
- **N-Gram:** A sequence of `n` words.
  - **Unigram:** 1 word (e.g., "cheese")
  - **Bigram:** 2 words (e.g., "cheddar cheese")
  - **Trigram:** 3 words (e.g., "aged cheddar cheese")
- **How it Works:** We calculate probabilities by counting n-grams in a massive corpus of text.
  - `P(word_n | word_n-1) = count(word_n-1, word_n) / count(word_n-1)`
  - *Example:* `P(croquet | to play) = count("to play croquet") / count("to play")`
- **Quality:** As `n` increases (longer context), the quality and believability of the generated text improve dramatically.

### Problems with N-Gram Models

1.  **Sparsity / The Curse of Dimensionality:** As `n` gets larger, the number of possible n-grams grows exponentially. It becomes almost impossible to find enough examples of any specific long n-gram in the training corpus. There is **never enough training data**.
2.  **Scalability:** The memory required to store all the n-gram counts scales exponentially, making models with large `n` impractical.
3.  **Inability to Generalize:** The model has no concept of similarity. "The cat sat on the..." and "The feline sat on the..." are treated as completely unrelated contexts.

### Solutions for Sparsity in N-Gram Models

Because most n-grams will be unseen, we need techniques to avoid assigning zero probability.

- **Smoothing (e.g., Additive/Laplace Smoothing):** Add a small constant (a *pseudocount*, often 1) to all n-gram counts. This ensures no probability is ever zero.
- **Backoff:** If a high-order n-gram (e.g., a trigram) has a zero count, "back off" to a lower-order model (e.g., a bigram). If that is also zero, back off to a unigram.
- **Interpolation:** Calculate the final probability as a weighted average of the unigram, bigram, and trigram probabilities. `P_final = λ₁*P_uni + λ₂*P_bi + λ₃*P_tri`.

### Generating Text from a Language Model

Once we have a way to predict the next word, there are several strategies for generating a full sequence:

- **Greedy Search:** Always choose the single most probable next word. It's deterministic (always produces the same text) and often results in boring, repetitive output.
- **Random Sampling:** Sample the next word from the entire probability distribution. This produces more varied but often incoherent text.
- **Top-k Sampling:** A compromise. Limit the sampling pool to the `k` most likely words.
- **Temperature Sampling:** Adjusts the "sharpness" of the probability distribution. High temperature makes the distribution more uniform (more random), while low temperature makes it more greedy.
- **Beam Search:** A search algorithm that keeps track of the `k` most probable sequences (beams) at each step. It is computationally more expensive but often produces the highest-quality text among these methods.

### Evaluating Language Models

- **Extrinsic Evaluation:** Evaluate the LM's performance on a downstream task (e.g., does it improve a spelling corrector or a machine translation system?). This is the "true" test but is slow and expensive.
- **Intrinsic Evaluation:** Measure the quality of the model itself by seeing how well it fits a held-out test dataset. The primary intrinsic metric is **Perplexity**.

#### Perplexity (PP)
- **Concept:** Measures the **surprise** or **confusion** of the model when presented with new text. It is a measure of how **unlikely** the observed test data is under the model.
- **Rule:** **Lower perplexity is better.** A lower perplexity means the model assigned a higher probability to the test data, so it was less "surprised."
- **Calculation:** It is the inverse of the geometric mean of the probability of the sequence.
  - `PP(W) = P(w₁, w₂, ..., wₙ)^(-1/n)`
- **Relationship to Other Metrics:** Perplexity is directly related to **Cross-Entropy (CE)** and **Negative Log-Likelihood (nLL)**. Minimizing perplexity is equivalent to minimizing cross-entropy.

## 2. Neural Networks (A Quick Revision)

Neural networks are the foundation for modern word embeddings and language models.

- **Neuron:** A simple processing unit that takes weighted inputs, sums them, and applies a non-linear **activation function** (like Sigmoid, Tanh, or ReLU) to produce an output.
- **Network:** Neurons are arranged in **layers**. Layers between the input and output are called **hidden layers**.
- **Power:** A neural network with a single, sufficiently wide hidden layer can approximate **any function**. Hidden layers allow the model to learn **non-linear decision boundaries**.
- **Training:**
    - Parameters (**weights** and **biases**) are learned via **backpropagation**, which is a **gradient descent** routine.
    - It uses the **chain rule** to calculate how much each weight contributed to the final error and updates the weights accordingly.

## 3. Word Embeddings

Word embeddings are the modern solution to the generalization problem that plagued n-gram models.

### What are Word Embeddings?
- They are **dense vectors** that represent words in a high-dimensional space (typically 100-1000 dimensions).
- This is a dramatic shift from **one-hot encodings**, which are extremely high-dimensional (size of the vocabulary) and sparse (all zeros except for one '1').
- **Key Idea:** Words with similar meanings will have similar vectors. Embeddings are learned from a word's **surrounding context**.

### The Supervised Learning Problem for Embeddings
- The core task is to **predict a missing word based on its context**.
- **Problem with classic methods:** If you use a bag-of-words representation for the context and a standard linear classifier, the number of parameters required becomes quadratic in the size of the vocabulary (billions of parameters), which is intractable.
- **Solution (Word2Vec):** A shallow neural network is used to solve this problem efficiently.

### Word2Vec

- **Architecture:** A simple neural network with a single **linear** hidden layer. The "embedding" for a word is its corresponding weight vector in this hidden layer.
- **Two Main Versions:**
    1.  **Continuous Bag-of-Words (CBOW):** The task is to predict a target word given its surrounding context words (a "many-to-one" prediction). The context is represented as a bag-of-words.
    2.  **Skip-gram:** The task is to predict the surrounding context words given a single target word (a "one-to-many" prediction). This generally performs better.
- **Training Trick: Negative Sampling**
    - The original formulation required predicting the correct word out of the entire vocabulary (a huge softmax calculation).
    - **Negative Sampling** reformulates the problem as a simple **binary classification task**: given a word pair `(word, context)`, predict whether this pair was actually observed (`positive example`) or was randomly generated (`negative example`). This is vastly more efficient to train.

### GloVe (Global Vectors)
- An alternative embedding model that operates on a different principle.
- **Core Idea:** Instead of just looking at local context windows, GloVe is trained to learn vectors such that their dot product equals the logarithm of their global co-occurrence probability.
- It directly factorizes the global **word co-occurrence matrix**.

### Properties & Applications of Word Embeddings

- **Semantic Clustering:** Words that are semantically related (e.g., "frog", "toad", "lizard") appear close to each other in the embedding space.
- **Meaningful Analogies (Additive Semantics):** The vector space has meaningful directions. The most famous example is:
  `vector('king') - vector('man') + vector('woman') ≈ vector('queen')`
- **Generalization:** Because similar words have similar vectors, a model can **generalize** from seen examples to unseen but semantically related examples. This solves the primary weakness of n-gram models.
- **"Sriracha Sauce of NLP":** Adding pre-trained word embeddings as the input layer to a neural network improves performance on almost every NLP task.

### Extensions: FastText & Sub-word Embeddings

- **Problem:** Standard Word2Vec/GloVe have a fixed vocabulary and cannot handle **new or unknown words** (Out-of-Vocabulary or OOV words).
- **Solution: FastText**
    - Represents each word as the sum of embeddings for its **character n-grams**.
    - *Example:* The word "where" is represented by n-grams like `wh`, `whe`, `her`, `ere`, `re`, etc.
    - **Advantage:** It can construct a vector for any OOV word by summing the vectors of its character n-grams. It also handles morphological variations well (e.g., "believe" and "believing" will have similar vectors).