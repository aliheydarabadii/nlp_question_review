# Exam Review: Sequence-to-Sequence Models & Transformers

This guide provides a complete summary of the evolution from Recurrent Neural Network (RNN) based sequence-to-sequence models to the modern Transformer architecture, covering attention, self-attention, and the key models like BERT and GPT.

## 1. Sequence-to-Sequence (Seq2Seq) Models

Seq2Seq models are designed for tasks where the input is a sequence and the output is also a sequence, and their lengths can be different.

-   **Canonical Tasks:** Machine Translation, Text Summarization, and Dialog Systems.
-   **Core Idea:** You can frame almost any NLP task as a Seq2Seq problem (e.g., Question Answering: input a question, output an answer).

### The Classic RNN/LSTM Architecture
The original Seq2Seq models consist of two main components, both typically built using RNNs or LSTMs:

1.  **The Encoder:**
    -   Reads the input sequence one word at a time.
    -   Its job is to compress the entire meaning of the input sequence into a single fixed-size vector, often called the **context vector** or "thought vector".

2.  **The Decoder:**
    -   Takes the context vector from the encoder as its initial state.
    -   Generates the output sequence one word at a time, using its previous output as input for the next step.

### The Bottleneck Problem
The classic Seq2Seq architecture has a major flaw:
-   The **single context vector** is a bottleneck. It's incredibly difficult to compress all the information from a long input sentence (e.g., a politician's rambling speech) into one vector without losing crucial details.

## 2. The Attention Mechanism

The **Attention Mechanism** was introduced to solve the bottleneck problem and revolutionized Seq2Seq models.

-   **Core Idea:** Instead of forcing the encoder to create a single context vector, let the decoder have **direct access to the hidden states of every single input word** from the encoder.
-   **How it Works:** At each step of generating an output word, the decoder:
    1.  Looks at all the encoder's output states.
    2.  Calculates a set of **attention scores** or **weights**, which represent how "important" each input word is for generating the *current* output word.
    3.  Creates a new, dynamic context vector for that specific timestep, which is a **weighted average** of all the encoder output states, using the attention weights.
    4.  This context vector is then used to help predict the next output word.

-   **Why it's powerful:**
    -   It provides a **direct route** for information to flow from input to output, bypassing the bottleneck.
    -   It allows the model to handle **word reordering** between languages (e.g., by learning a diagonal attention pattern).
    -   It allows the model to handle **long-range dependencies**, like using a noun far ahead in the input sentence to determine the gender of an article it is generating now.

### Calculating Attention Scores
-   **Multiplicative (Dot-Product) Attention:** The most common method. The score is calculated by taking the **dot product** between the decoder's current hidden state (`query`) and each of the encoder's hidden states (`key`). The result is often scaled by the square root of the embedding dimension to stabilize training.
-   **Additive Attention:** An alternative where the decoder and encoder states are concatenated and fed through a small feed-forward network.

### Generalized Notation: Query, Key, Value
The attention mechanism can be generalized with this terminology:
-   **Query (Q):** What I am currently looking for. In classic attention, this is the decoder's hidden state.
-   **Key (K):** An "index" or label for a piece of information. This is an encoder hidden state.
-   **Value (V):** The actual information content. In classic attention, this is also the encoder hidden state (K and V are the same).

The process is: compare your **Query** to every **Key** to get weights, then use those weights to create a weighted sum of the **Values**.

## 3. The Rise of Deep Learning & The Transformer

### What is Deep Learning?
-   **Definition:** Simply Neural Networks with **many layers** (deep networks).
-   **Why it's effective:** Stacking many layers allows the network to learn a **hierarchy of features** automatically, from simple patterns in lower layers to complex concepts in higher layers.
-   **Downsides:** Requires massive amounts of data, huge computing resources (GPUs), and can be difficult to train.

### The Problem with RNNs
-   Even with attention, RNNs have a fundamental training problem: they are **inherently sequential**.
-   The calculation for step `t` depends on the output of step `t-1`. This means the computation **cannot be parallelized** across the sequence.
-   This makes training deep stacks of RNNs on very long sequences extremely **slow**.

## 4. The Transformer: "Attention Is All You Need"

The Transformer architecture, introduced in 2017, was a breakthrough that solved the RNN training problem.

-   **The "Crazy Idea":** What if we **remove the recurrent links** entirely and rely *only* on attention?
-   This leads to the core mechanism of the Transformer: **Self-Attention**.

### Self-Attention
-   **Purpose:** A mechanism for a model to update the embedding of each word in a sequence by looking at **all other words in the same sequence**.
-   **How it works:** For each word's embedding, `e_i`:
    -   It creates its own **Query (Q)** vector.
    -   It compares its Query to the **Key (K)** vector of every other word in the sequence (including itself) to get attention weights.
    -   It then calculates a new, updated embedding for itself, which is a weighted average of all the other words' **Value (V)** vectors.
-   **Result:** The new embedding for a word is contextualized by all other words in the sentence. For example, the embedding for "bank" in "river bank" will be different from the one in "investment bank" because it will be influenced by the presence of "river" or "investment".

### The Transformer Block
The Transformer is built by stacking a basic module, the **Transformer block**, many times. Each block contains two main sub-layers:
1.  **A Self-Attention Layer:** This is where the contextualization happens. It often has **multiple attention heads** working in parallel, each focusing on different types of relationships.
2.  **A Feed-Forward Neural Network (FFNN):** A simple, position-wise MLP that processes the output of the self-attention layer. This layer acts like a key-value memory, helping to "look up" facts and add more information to the representation.

### Key Architectural Details
-   **Positional Encoding:** Since self-attention is order-agnostic, we must explicitly add information about the position of each word. This is done by adding a **positional encoding vector** (often generated using sine and cosine functions) to the initial word embedding.
-   **Parallelization:** The entire self-attention calculation can be performed with highly optimized **matrix multiplications**. This means all token positions are updated in **parallel**, making Transformers dramatically **faster to train** than RNNs.

## 5. BERT vs. GPT: A Tale of Two Architectures

The original Transformer had both an **encoder** and a **decoder**. Subsequent research found that using just one of these components was extremely powerful for different purposes.

### BERT: Bidirectional Encoder Representations from Transformers
-   **Architecture:** Uses only the **Encoder** stack of the Transformer.
-   **Training Objective: Masked Language Modeling (MLM).**
    -   It takes a sentence, randomly **masks** out some of the words (e.g., replaces them with a `[MASK]` token), and its job is to predict what the original masked words were.
-   **Key Property: Bidirectional.** Because it sees the entire sentence (with holes) at once, it learns from both the left and right context of a word.
-   **Best For:** **Representing text** and natural language understanding (NLU) tasks like **text classification**, where you need a deep understanding of the entire sentence.

### GPT: Generative Pretrained Transformer
-   **Architecture:** Uses only the **Decoder** stack of the Transformer.
-   **Training Objective: Causal Language Modeling.**
    -   Its job is to **predict the next word** in a sequence, given only the words that came before it. This is done by masking all *future* words.
-   **Key Property: Autoregressive.** It generates text one token at a time, from left to right.
-   **Best For:** **Generating text**.

### Pre-training
-   Both BERT and GPT are **pre-trained** on massive amounts of text data (e.g., Wikipedia, books, web text). This is where they learn general knowledge about language.
-   They can then be **fine-tuned** on a smaller, task-specific dataset.

## 6. Using Transformers in Practice

### Text Classification with BERT
-   **Advantage:** BERT **removes the need for manual feature engineering**. You no longer need to decide on stemming, n-grams, etc. The model learns the best features from raw text.
-   **How it Works (Fine-tuning):**
    1.  Add a special `[CLS]` token to the beginning of the input text.
    2.  Feed the text through the pre-trained BERT model.
    3.  Take the final hidden state vector corresponding to the `[CLS]` token (which now represents the entire sequence's meaning).
    4.  Add a simple linear classifier on top of this `[CLS]` vector and train it to predict the class label.

-   **Practical Considerations:**
    -   **Cons:** Slower, requires more memory and powerful hardware (GPUs), and is less interpretable than simple linear models.
    -   **Hard Text Limit:** Has a fixed context size (e.g., 512 or 1024 tokens).

### Transfer Learning & Multi-linguality
-   Transformers enable powerful **transfer learning**. A model pre-trained on a massive dataset can be fine-tuned with a very small amount of labeled data for a specific task and still achieve high performance.
-   **Multi-linguality:** Models like mBERT are pre-trained on over 100 languages, allowing you to, for example, train a classifier on English data and have it work surprisingly well on Italian data, a feat previously very difficult to achieve.