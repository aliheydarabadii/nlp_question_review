# Exam Review: Applications of Transformers

This guide provides a complete summary of how Transformer models like BERT and GPT are adapted and applied to a wide variety of NLP tasks.

## 1. Reminder: The Three Transformer Architectures

Understanding which architecture to use is key to understanding its applications.

1.  **Original Transformer (Encoder-Decoder):**
    -   Designed for **translation**.
    -   Consists of two stacks of Transformer blocks: an encoder to process the source text and a decoder to generate the target text.
    -   Models like **T5** and **BART** use this full architecture.

2.  **BERT (Encoder-Only):**
    -   Stands for **B**idirectional **E**ncoder **R**epresentations from **T**ransformers.
    -   It's a "noisy autoencoder": pretrained by masking random words and learning to recover them.
    -   Because it sees the whole sentence at once, it is **bidirectional** and builds a deep contextual understanding.
    -   **Its primary strength is representing text for understanding tasks** (e.g., classification, entity recognition).

3.  **GPT (Decoder-Only):**
    -   Stands for **G**enerative **P**retrained **T**ransformer.
    -   It's **autoregressive**: pretrained to predict the very next word given the previous words.
    -   It is **unidirectional** (left-to-right).
    -   **Its primary strength is generating fluent and coherent text**.

## 2. Fine-Tuning Transformers for Specific Tasks

**Fine-tuning** is the process of taking a pre-trained Transformer and continuing to train it (updating its weights) on a small, labeled dataset for a specific downstream task.

### a) Fine-Tuning BERT (The Flexible Workhorse)

BERT's architecture is extremely flexible and can be adapted for many NLU tasks.

-   **Text Classification:**
    -   **How:** Add a special `[CLS]` token to the start of the input. The final hidden state of this token is used as a representation of the entire sentence, which is fed into a simple linear classifier. We fine-tune the model to predict the class label.

-   **Sequence Labeling (e.g., Named Entity Recognition):**
    -   **How:** Instead of one output for the `[CLS]` token, we add a classifier on top of *every* output token's final hidden state. The model is fine-tuned to predict a label (e.g., B-PER, I-PER, O) for each token in the sequence.

-   **Text Pair Classification:**
    -   **How:** Concatenate two pieces of text, separated by a special `[SEP]` token (e.g., `[CLS] sentence A [SEP] sentence B [SEP]`). Then use the `[CLS]` token's output to classify the relationship between the two texts.
    -   **Applications:** Determining if two sentences are paraphrases, if a hypothesis is supported by a premise (Natural Language Inference), or if two documents discuss the same topic.

-   **Question Answering:**
    -   BERT can be fine-tuned for extractive QA (where the answer is a span of text in a given passage). GPT is generally more suited for generative QA.

### b) Fine-Tuning GPT-2 (The Text Generator)

While GPT can be used for classification, its strength lies in generation.

-   **How:** Fine-tuning GPT involves teaching it to respond to specific **prompts** or **special tokens**. The dataset is formatted to show the model the desired input-output behavior.
-   **Applications:**
    -   **Translation:** The training data consists of examples like: `"I am a student <to-fr> je suis étudiant"`. The `<to-fr>` token prompts the model to translate.
    -   **Summarization:** The training data looks like: `"[Article Text] <summarize> [Article Summary]"`. The `<summarize>` token prompts for a summary.
    -   **Dialog:** Can be fine-tuned on conversation logs to act as a chatbot.

## 3. Learning Without Fine-Tuning: In-Context Learning

For very large language models (like GPT-3 and beyond), a new paradigm emerged that **does not require gradient updates**. The model "learns" the task simply from the examples provided in the prompt.

-   **Fine-Tuning vs. In-Context Learning:**
    -   **Fine-Tuning:** A large corpus of examples is used to *update the model's weights* via gradient descent.
    -   **In-Context Learning:** A few examples are placed directly into the *prompt*. No weights are updated.

-   **Types of In-Context Learning:**
    -   **Zero-shot:** The model is given only a natural language description of the task. (e.g., `"Translate cheese to French:"`)
    -   **One-shot:** The model is given one complete example of the task. (e.g., `"sea otter => loutre de mer, cheese =>"`)
    -   **Few-shot:** The model is given a few (2+) complete examples. This generally yields the best performance.

-   **How is this possible?** The model has seen countless examples of similar formats (Q&A, lists, translations) during its massive pre-training on web data. It learns to recognize the pattern in the prompt and complete it.

-   **Flexibility:** This allows LMs to be used as "universal learners" for many tasks on the fly, like question answering, reading comprehension, and summarization (e.g., by adding `"tl;dr:"` to the end of a long text).

## 4. Document Embeddings & Semantic Search

A major application of Transformers is creating high-quality, contextual embeddings for sentences and documents, enabling semantic search.

### The Problem: Computational Cost of Pairwise BERT
-   Using a fine-tuned BERT model for text-pair classification (e.g., `(query, document) -> relevance_score`) is very powerful but **extremely slow**.
-   You would need to run the full BERT model for **every document in your collection for every query**. This is computationally infeasible for any large-scale system.

### Solution 1: Lexical Search + Reranking (The Two-Stage Approach)
This is the standard industry practice for balancing speed and quality.
1.  **Retrieve:** Use a fast, traditional "lexical search" engine (like BM25 or TF-IDF) to quickly find a set of a few hundred potentially relevant "candidate" documents.
2.  **Rerank:** Run the powerful but slow pairwise BERT classifier **only on this small set of candidates** to produce a final, high-quality ranking.

### Solution 2: Pre-computing Document Embeddings
The ideal solution is to do as much computation as possible offline, before the user ever submits a query.

-   **Goal:** Create a single vector (embedding) for every document in the collection *once*. Then, at query time, create a vector for the query and perform a very fast nearest neighbor search.
-   **Challenge:** The standard BERT `[CLS]` token is not ideal for this out-of-the-box, as it wasn't trained to produce good sentence-level similarity embeddings.

-   **Sentence-BERT (SBERT): A Better Approach**
    -   **Method:** SBERT fine-tunes a BERT-like model specifically to produce useful document embeddings.
    -   **Training:** It uses **contrastive learning**. The model is shown pairs of documents and trained to:
        -   Produce **similar** embeddings (high dot-product) for documents that are semantically similar.
        -   Produce **dissimilar** embeddings (low dot-product) for documents that are not similar.

### Vector Databases & Approximate Nearest Neighbor (ANN) Search
-   **Vector Databases:** Specialized databases designed to store and efficiently search through millions or billions of high-dimensional embedding vectors. (e.g., FAISS, Milvus, Pinecone).
-   **The Search Problem:** Finding the exact nearest neighbor in high-dimensional space is very slow (the "curse of dimensionality").
-   **Solution: Approximate Nearest Neighbor (ANN) Search.** These algorithms trade perfect accuracy for immense speed.
    -   **HNSW (Hierarchical Navigable Small Worlds):** A state-of-the-art ANN algorithm that builds a multi-layered graph of the vectors to allow for very fast traversal and search.

## 5. Advanced & Multi-Domain Applications

### Multi-task Learning
-   **Concept:** Instead of fine-tuning a model for one specific task, train it on **many different tasks at the same time**.
-   **Benefit:** Models trained this way often **outperform models trained on a single task**. The model learns more robust and generalizable underlying representations by seeing how they apply across different contexts.

### Multi-modal Learning
-   **Concept:** Extending text-only models to understand and process multiple modalities (e.g., **text and images**). The Transformer architecture is very flexible for this.
-   **CLIP (Contrastive Language-Image Pre-training):**
    -   **Goal:** To create a **shared, aligned embedding space** for both images and text.
    -   **How:** It is trained on a massive dataset of `<image, text caption>` pairs. It learns to map a given image and its correct caption to nearby points in the embedding space.
    -   **Killer Application:** Enables powerful **semantic image search using a text query**. You can search for "a soccer fan elephant riding a bicycle," and CLIP will find images that match that concept, even if the words never appeared in the image's metadata.