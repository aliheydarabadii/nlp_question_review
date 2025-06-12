# NLP Exam 2 - Comprehensive Review Guide

This guide synthesizes the key concepts from the 2023-24 NLP exam analysis.

#### **1. Foundational Concepts & Terminology**

*   **Statistical Laws of Text:**
    *   **Heap's Law:** Describes vocabulary growth. The size of the vocabulary grows roughly in proportion to the **square root** of the length of the document/collection. `V ≈ K * N^β` (where `β` is ~0.5). (Q4)
    *   **Zipf's Law:** Describes word frequency distribution. A few words are extremely common (stop words), while most words are very rare.

*   **Ambiguity in Language:**
    *   **Syntactic Ambiguity:** A sentence can be parsed in multiple ways. *Example: "I made her duck."* (Did I cook a duck for her, or cause her to lower her head?). (Q13)
    *   **Lexical Ambiguity (Homographs):** Words with the same spelling but different meanings and pronunciations. *Example: "bass" (fish vs. instrument)*. This is a key challenge for Text-to-Speech (TTS) systems. (Q55)
    *   **Solving Ambiguity:** The primary tool for disambiguation is **surrounding context**.

*   **Discourse & Speech Acts:**
    *   **Speech Acts (Bach & Harnish):**
        *   **Directives:** The speaker tries to get the hearer to do something (e.g., advise, ask, order, request). (Q14)
        *   **Commissives:** The speaker commits themselves to a future action (e.g., promise).
        *   **Constatives:** The speaker commits to the truth of a statement (e.g., claim).
    *   **Grounding:** Repeating back what a speaker said is a mechanism to **acknowledge understanding and establish a common ground** for the conversation. (Q38)

*   **Information Theory:**
    *   **Self-Information (Surprisal):** The amount of information learned from an event is `log(1 / P(event))` or `-log(P(event))`. Low-probability (surprising) events have high information content. (Q64)

#### **2. Text Preprocessing & Representation**

*   **Common Preprocessing Steps:** (Q85)
    *   **Tokenization:** Splitting text into tokens (words, sub-words).
    *   **Case-Folding:** Converting text to lowercase.
    *   **Removing Markup:** Deleting HTML tags, etc.
    *   **Stop Word Removal:** Removing high-frequency, low-information words (e.g., "the", "is"). Stop words have the **longest posting lists** and are the **least discriminative**. (Q1)
    *   **Stemming & Lemmatization:** (Q10, Q44)
        *   **Stemming:** A fast, crude, rule-based process that chops off suffixes (e.g., `studies` -> `studi`). The result may not be a real word.
        *   **Lemmatization:** A slower, more sophisticated, dictionary-based process that finds the root form (lemma) (e.g., `studies` -> `study`).
        *   **Purpose:** To **reduce the feature set** and consolidate examples, improving generalization.

*   **Text Representation Methods:**
    *   **Bag-of-Words (BoW):** (Q19, Q63)
        *   Represents a document as a sparse vector of word counts.
        *   **Ignores word order**.
        *   Can be conceptualized as the **summation of the one-hot vectors** for each word in the document.
        *   Weakness: Does not handle synonymy (e.g., "car" and "auto" are different features).
    *   **One-Hot Vectors:** A sparse vector with a single '1' at the index corresponding to a word, and '0's everywhere else. (Q19)
    *   **Word Embeddings (Dense Vectors):** (Q43, Q81)
        *   Represent words as dense, low-dimensional, floating-point vectors.
        *   **Key Property:** Semantically similar words have similar vectors (e.g., `vec(king) ≈ vec(queen)`).
        *   Learned by models like **Word2Vec**, **GloVe**, and **FastText**.
    *   **Sub-word Embeddings:** (Q3, Q25)
        *   **Method:** Break words into smaller units (e.g., using **Byte-Pair Encoding (BPE)**) and learn embeddings for these units.
        *   **Primary Advantage:** Can represent **Out-of-Vocabulary (OOV)** words.
        *   **FastText:** A specific type that represents a word as a bag of its character n-grams. (Q67)

#### **3. Language Modeling & Generation**

*   **Language Model (LM):** A model that computes a **probability distribution over sequences of words**. (Q22)
*   **N-gram Models:**
    *   Estimate the probability of a word given the previous `n-1` words.
    *   **Limitation:** Suffer from data sparsity. The probability of seeing a specific n-gram decreases **exponentially** with `n`. (Q51)
    *   **Smoothing/Backoff/Interpolation:** Techniques used to handle unseen n-grams and improve probability estimates. (Q76)
*   **Evaluation - Perplexity:**
    *   A measure of how well a model predicts a test set. **Lower perplexity is better**. (Q68)
    *   Formula: `PP(W) = P(W)^(-1/N)`, where `N` is the number of tokens. (Q12)
*   **Text Generation Techniques:**
    *   **Greedy Sampling:** Always pick the single most probable next token.
    *   **Random Sampling:** Sample from the full probability distribution.
    *   **Top-k Sampling:** Sample from the `k` most probable tokens. Setting `k` to the vocabulary size is equivalent to **random sampling**. (Q6, Q21)
    *   **Beam Search:** Keeps track of the `b` most likely sequences at each step. It is the most **computationally expensive** of these methods. (Q26)

#### **4. Machine Learning Models**

*   **Classic Classifiers:**
    *   **Naive Bayes:** Assumes features are conditionally independent. The **prior probability** of a class is its frequency in the training data. (Q32)
    *   **Logistic Regression (LR):** A linear classifier that produces **well-calibrated probability estimates**. It uses **Log Loss**, not Hinge Loss. (Q18)
    *   **Support Vector Machines (SVMs):** A linear classifier that uses **Hinge Loss**.
*   **Clustering:**
    *   **K-Means vs. K-Medoids:** K-Medoids is more robust to outliers but is much more **computationally expensive** because it must test every point in a cluster as a potential new medoid. (Q77)
    *   **Latent Dirichlet Allocation (LDA):** An unsupervised model used to **discover latent topics** in a document collection. (Q11)

#### **5. Modern Deep Learning & Transformers**

*   **Evolution of Architectures:** (Q29, Q71, Q66)
    1.  **RNNs/LSTMs:** Process sequences token-by-token. Suffer from vanishing gradients but LSTMs mitigate this with gates (forget, input, output). Can be stacked.
    2.  **Sequence-to-Sequence (Seq2Seq):** An encoder-decoder architecture, originally using RNNs, for tasks where input/output lengths differ (e.g., translation). (Q31, Q74)
    3.  **Transformers:** The current SOTA. Replaced recurrence with **self-attention**, allowing for massive parallelization and faster training.

*   **The Transformer Architecture:**
    *   **Attention Mechanism:** Allows the model to weigh the importance of different input tokens when producing an output token. The formula `softmax( (Q * K^T) / sqrt(d_k) ) * V` uses **Query, Key, and Value** vectors. (Q9, Q30)
    *   **Positional Encoding:** Since the architecture has no inherent sense of order, these are added to give the model information about token position.

*   **Key Transformer Models:**
    *   **BERT (Bidirectional Encoder):** (Q5)
        *   Uses an encoder-only architecture.
        *   **Bidirectional:** Learns from both left and right context simultaneously using a Masked Language Model (MLM) pre-training objective.
        *   Excellent for NLU tasks like classification.
        *   Uses `[CLS]` and `[SEP]` tokens for classification and pairwise tasks. (Q79)
    *   **GPT (Generative Pre-trained Transformer):** (Q34, Q35)
        *   Uses a decoder-only architecture.
        *   **Unidirectional (Autoregressive):** Reads left-to-right, making it ideal for text generation.
    *   **T5 (Text-to-Text Transfer Transformer):** (Q40)
        *   Uses a full **encoder-decoder** architecture.
        *   Frames all NLP tasks as a text-to-text problem.
    *   **CLIP (Contrastive Language-Image Pre-training):** (Q75)
        *   A multi-modal model that learns a **shared embedding space for images and text**, enabling text-to-image retrieval and zero-shot classification.

*   **Fine-Tuning & Adaptation:**
    *   **In-Context Learning:** Prompting a model with examples without updating its weights. (Q54, Q58, Q73)
        *   **Zero-shot:** No examples given.
        *   **One-shot:** One example given.
        *   **Few-shot:** A few examples given.
    *   **LoRA (Low-Rank Adaptation):** A Parameter-Efficient Fine-Tuning (PEFT) method that freezes pre-trained weights and injects small, trainable low-rank matrices to adapt a model. (Q2)
    *   **RLHF (Reinforcement Learning from Human Feedback):** The process of fine-tuning a base LLM to become a helpful and harmless instruction-following assistant (chatbot) by using human-ranked responses to train a reward model. (Q42)
    *   **Multi-task Learning:** Training a model on many tasks at once to improve generalization and create a single, versatile model. (Q16)

*   **LLM Concepts & Limitations:**
    *   **Hallucination:** Generating factually incorrect or nonsensical statements. This happens because LMs are probabilistic sequence generators, not fact-checkers. (Q59)
    *   **Anthropomorphism:** The act of attributing human emotions and intentions to a chatbot. (Q48)
    *   **Quantization:** A model compression technique that reduces the precision of the model's weights (e.g., from 32-bit to 4-bit) to save memory and speed up inference. (Q46)
    *   **RAG (Retrieval-Augmented Generation):** A technique where an LLM's knowledge is augmented by first **retrieving** relevant documents from a knowledge base (like a vector DB) and adding them to the prompt before **generating** an answer. (Q8)

#### **6. NLP Tasks & Applications**

*   **Named Entity Recognition (NER):** Identifying and classifying named entities (Person, Organization, Location). Often uses **BIO (Begin-Inside-Outside) tagging** for training data. (Q27, Q57)
*   **Co-reference Resolution:** Determining which expressions (e.g., pronouns) refer to the same entity. (Q52)
*   **Sentence Segmentation:** Splitting text into sentences. Complicated by the ambiguous use of punctuation like `.` and `?`. (Q72)
*   **Spoken Language Systems:**
    *   **ASR (Speech-to-Text):** Converts audio to text.
    *   **TTS (Text-to-Speech):** Converts text to audio. Requires **text normalization** (e.g., "123" -> "one hundred and twenty-three"). (Q50)
    *   **Spectrograms:** Visual representations of audio signals. A **Mel Spectrogram** has a linear time axis and logarithmic frequency/amplitude axes. Created using STFT, windowing (Hamming), and pre-emphasis. **K-Means is NOT used**. (Q24, Q36)
    *   **Challenges vs. Chatbots:** Latency, barge-in detection, prosody generation, and robustness to ASR errors make voice agents much harder to build. (Q45)

#### **7. Information Retrieval (IR)**

*   **Lexical (Term-based) Search:**
    *   **Inverted Index:** A data structure mapping terms to **posting lists**. A posting list for a term contains the IDs of all documents that contain it. (Q62)
    *   **Ranking Functions:**
        *   **TF-IDF:** Basic term weighting scheme.
        *   **BM25:** An advanced function that includes **term frequency saturation**, making it more robust to keyword stuffing than TF-IDF. (Q33)
*   **Semantic (Vector) Search:**
    *   Represents documents and queries as embeddings.
    *   **Vector Databases:** Specialized systems designed for efficient **fast nearest neighbor search** in high-dimensional embedding spaces. (Q60)
*   **Hybrid Approach (Re-ranking):** Use a fast lexical search (e.g., BM25) to get initial candidates, then use a powerful but slow neural model (e.g., BERT) to **re-rank** the top results. (Q41)
*   **Evaluation Metrics:**
    *   **P@k (Precision at k), MAP, NDCG** are common IR evaluation metrics. (Q70, Q78)
    *   **Silhouette Index** is for clustering, NOT search evaluation. (Q78)

#### **8. Key Calculations to Know**

*   **Prior Probability:** `P(class) = Count(class) / Total_Count` (Q32)
*   **Precision:** `Precision = TP / (TP + FP)` (Q69)
*   **Perplexity:** `PP = P(sequence)^(-1/N)` (Q12)
*   **N-gram Probability:** `P(w1,w2,w3) = P(w1) * P(w2|w1) * P(w3|w1,w2)` (Q23)
*   **Model Memory Size:** `Size = (Num_Params) * (Bytes_per_Param)` (Q46, Q80)
    *   Remember: 8 bits = 1 byte. 64-bit float = 8 bytes. 4-bit = 0.5 bytes.
