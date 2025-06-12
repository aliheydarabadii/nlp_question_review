# Exam Review: Sequence Classification & Labeling

This guide provides a complete summary of models that process sequences of text, covering the importance of word order, key model architectures (HMMs, CRFs, RNNs), and their primary applications in NLP.

## 1. Why Word Order is Crucial

Simple models like **Bag-of-Words (BoW)** treat text as an unordered collection of words. This fails for many tasks because **word order defines meaning**.

-   **Semantic Meaning:**
    -   `"There’s a white rat in the house"` (A rodent is present).
    -   `"There’s a rat in the White House"` (A person is in a specific government building).
    - These have identical words but completely different meanings. BoW cannot distinguish them.

-   **Negation & Sentiment:**
    -   `"I am not happy about going to school."` (Negative sentiment)
    -   `"I am happy about not going to school."` (Positive sentiment)
    - BoW would see these as nearly identical. Models that understand sequences are required to correctly interpret them.

## 2. Sequence Classification vs. Sequence Labeling

These are the two main tasks for models that process ordered text.

-   **Sequence Classification:**
    -   **Input:** An ordered sequence of tokens (a sentence or document).
    -   **Output:** A **single label** for the *entire* sequence.
    -   **Example:** Sentiment analysis (classifying a whole review as "positive" or "negative").

-   **Sequence Labeling (or Tagging):**
    -   **Input:** An ordered sequence of tokens.
    -   **Output:** A **sequence of labels**, one for each token in the input.
    -   **Example:** Part-of-Speech (POS) tagging (labeling each word as a noun, verb, etc.).

## 3. Models for Sequence Labeling

### a) Traditional Models

These were state-of-the-art before the deep learning revolution.

#### Hidden Markov Models (HMMs)
-   **Analogy:** An HMM is to **Naive Bayes** what a sequence model is to a simple classifier. It's a generative model.
-   **Components:**
    1.  **Hidden States:** The labels we want to predict (e.g., POS tags like `NOUN`, `VERB`). They are "hidden" because we can't see them directly.
    2.  **Observed Words:** The words in the text.
    3.  **Transition Probabilities `P(state₂ | state₁)`:** The probability of moving from one hidden state to another (e.g., a `DETERMINER` is often followed by a `NOUN`).
    4.  **Emission Probabilities `P(word | state)`:** The probability of a state "emitting" (producing) a certain word (e.g., the `NOUN` state is likely to emit words like "cat", "house").

#### Conditional Random Fields (CRFs)
-   **Analogy:** A CRF is to **Logistic Regression** what a sequence model is to a simple classifier. It's a discriminative model.
-   **Key Difference from HMMs:** CRFs model the probability of the entire label sequence *given* the word sequence, `P(labels | words)`. They are generally more powerful than HMMs because they can handle a much richer set of overlapping features without the strict independence assumptions of HMMs.

### b) Modern Models: Recurrent Neural Networks (RNNs)

RNNs are neural networks specifically designed to process sequences.

-   **Core Idea:** An RNN processes a sequence one token at a time. At each step, it maintains a **hidden state** vector, which acts as its "memory" of everything it has seen so far.
-   **Process at each step:**
    `new_hidden_state = function(current_word_embedding, previous_hidden_state)`
-   This recurrent nature allows RNNs to **accumulate information** and let the prediction for a word be influenced by any word that came before it.

#### Long Short-Term Memory (LSTM) Networks
LSTMs are an advanced and highly effective type of RNN that solved a major problem with simple RNNs (the vanishing gradient problem, which made it hard to learn long-range dependencies).

-   **Key Innovation: The Gating Mechanism**
    -   An LSTM cell has internal "gates" that learn to control the flow of information. It can learn **what to remember, what to forget, and what to output** at each timestep.
    -   **Forget Gate:** Decides what information to discard from the previous state.
    -   **Input Gate:** Decides what new information to add to the state.
    -   **Output Gate:** Decides what information to use from the state to make a prediction.
-   **Why it's powerful:** LSTMs can capture **long-range dependencies** and handle complex nested contexts (e.g., tracking a subject across multiple sentences or understanding the scope of a negation).

## 4. Applications of Sequence Models

Sequence labeling and classification are fundamental to many core NLP tasks.

### a) Part-of-Speech (POS) Tagging
-   **Task:** Assigning a grammatical category (e.g., Noun, Verb, Adjective, Adverb) to each word in a sentence.
-   **Word Classes:**
    -   **Open Class:** Content words that new words can be added to (Nouns, Verbs, Adjectives, Adverbs).
    -   **Closed Class:** Function words that are fixed (Determiners like "the", Prepositions like "in", Pronouns).
-   **Why it's useful:** It's a crucial first step for many downstream tasks. It helps reduce ambiguity (e.g., distinguishing "book" the noun from "book" the verb) and is needed for parsing.
-   **Difficulty:** While most words are unambiguous, ambiguous words are very common. The word "back" can be a Noun, Adjective, Adverb, Verb, or Particle.

### b) Named Entity Recognition (NER)
-   **Task:** To find and classify named entities (proper names) in text.
-   **Common Entity Types:**
    -   `PER` (Person): *Marie Curie*
    -   `LOC` (Location): *Lake Michigan*
    -   `ORG` (Organization): *Stanford University*
    -   `GPE` (Geo-Political Entity): *Boulder, Colorado*
-   **Challenges:**
    1.  **Segmentation:** Entities are often multi-word phrases.
    2.  **Type Ambiguity:** The same phrase can be different entity types depending on context (e.g., "*Paris Hilton* [PER] was photographed leaving the *Paris Hilton* [ORG/LOC].").

#### BIO Tagging Scheme
-   **Purpose:** To turn NER (a phrase-finding task) into a standard per-token sequence labeling task.
-   **Tags:**
    -   `B-TYPE`: The **B**eginning of an entity of a certain type (e.g., `B-PER` for "Jane").
    -   `I-TYPE`: A token **I**nside an entity (e.g., `I-PER` for "Villanueva").
    -   `O`: A token **O**utside any entity.

### c) Entity Linking
-   **Task:** The next step after NER. It involves linking a mentioned entity in the text to a unique, real-world entity in a **Knowledge Base** (like Wikipedia or DBpedia).
-   **Challenge: Ambiguity.**
    -   *Example:* Does "Paris" refer to the capital of France, Paris, Texas, or Paris Hilton?
    -   Context (other entities in the text) is used to disambiguate.

### d) Relation Extraction
-   **Task:** The next step after Entity Linking. It involves identifying the **semantic relationship** that exists between two linked entities.
-   **Example:** From the sentence `"Paris is the capital of France"`, we can extract the relation `is_capital_of(Paris, France)`. This is used to populate knowledge graphs.

### e) Syntactic Parsing
-   **Task:** To analyze the grammatical structure of a sentence.
-   **Output:** A **Parse Tree** (either a constituency or dependency parse).
-   **Purpose:** The parse tree shows **how words in a sentence relate to one another**, which is crucial for determining the precise meaning.
-   **Example:** In `"The chef who ran to the store was out of food"`, the parse tree would show that "was out of food" modifies "The chef", not "the store".

### f) Co-reference Resolution
-   **Task:** To identify all expressions in a text that refer to the same entity.
-   **Example:** In `"John went to the dealership. He looked at the Acura"`, co-reference resolution links `"He"` back to `"John"`.
-   **Why it's useful:** Essential for understanding the full context of what is being said about an entity, which is needed for tasks like information extraction and building coherent chatbots.

## 5. Ontologies & Knowledge Bases

-   **Taxonomy:** A simple hierarchy of concepts, usually with `is-a` relationships (e.g., a poodle *is a* dog, a dog *is an* animal).
-   **Ontology:** A more formal and rich specification of concepts and their relationships. Ontologies form a **knowledge graph**.
    -   **Components:**
        -   **Classes:** Types of objects (e.g., `Wine`).
        -   **Individuals:** Specific instances of a class (e.g., `Champagne`).
        -   **Attributes:** Properties with data types (e.g., `price` has type `integer`).
        -   **Relationships:** Links between individuals/classes (e.g., `winery` *produces* `wine`).
        -   **Logical Rules:** Inference rules (e.g., `hasParent(x,y) ^ hasBrother(y,z) -> hasUncle(x,z)`).

-   **Open World vs. Closed World Semantics:**
    -   **Closed World (Databases/SQL):** If a fact is not in the database, it is assumed to be **false**.
    -   **Open World (Knowledge Bases/OWL):** If a fact is not known to be true, its truth value is **unknown**. This is a crucial distinction for reasoning over incomplete knowledge.