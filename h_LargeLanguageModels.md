# Exam Review: Large Language Models (LLMs)

This guide provides a complete summary of Large Language Models, covering their evolution, training, prompting, limitations, and the latest architectural improvements.

## 1. What are LLMs?

-   **Definition:** A Large Language Model (LLM) is simply a very, very **Large** version of a standard **Language Model (LM)**.
-   **Core Function:** At its heart, an LLM is a model trained to **predict the next token** in a sequence.
-   **The "Large" Factor:** The scale of LLMs is what sets them apart. This refers to:
    1.  **Number of Parameters:** Modern LLMs have billions or even trillions of parameters (e.g., GPT-3 has 175 billion).
    2.  **Training Data Size:** They are trained on enormous datasets, often containing trillions of tokens from web crawls, books, etc.
    3.  **Computational Cost:** Training these models requires immense computational power (e.g., thousands of Petaflop-days, costing millions of dollars).

### The Impact of Scale: Emergent Abilities
-   As models get larger (more parameters), they show a **dramatic improvement** in performance, especially on **zero-shot, one-shot, and few-shot learning** tasks.
-   **Scaling Laws:** Research has shown a predictable, log-linear relationship between performance (loss) and three key factors: **compute, dataset size, and number of parameters**.
-   **Chinchilla Scaling Law (2022):** This crucial finding suggests that for optimal performance, the number of model parameters and the number of training tokens should be **scaled up in roughly equal proportions**. This led to models being trained on much more data than previously thought necessary for their size.

## 2. From LLMs to Chatbots: The Alignment Process

A base pre-trained LLM is good at completing text, but it's not a helpful assistant. To turn it into a chatbot like ChatGPT, it must be "aligned."

-   **LLM vs. Chatbot:**
    -   An **LLM** predicts the next token in a way that is statistically likely (e.g., completing a sentence from a book).
    -   A **Chatbot** is trained to converse with a user in a helpful, harmless, and engaging way.

-   **Alignment Techniques:**
    1.  **Instruction Tuning:** The model is fine-tuned on a massive dataset of diverse tasks formatted as natural language instructions (e.g., "Summarize this text:", "Translate this to French:", "Answer this question:"). This teaches the model to follow instructions. Instruction-tuned models show significantly better **zero-shot performance** on unseen tasks.
    2.  **Reinforcement Learning from Human Feedback (RLHF):** This is the key process for making a chatbot more helpful and less harmful.
        -   **Step 1 (Collect Feedback):** Generate multiple responses to user prompts.
        -   **Step 2 (Train Reward Model):** Have human labelers rank these responses from best to worst. Use this ranking data to train a separate "reward model" that learns to predict which responses humans prefer.
        -   **Step 3 (Fine-tune with RL):** Use the reward model as a reward function to fine-tune the LLM with reinforcement learning, optimizing it to generate responses that maximize the predicted human preference score.

## 3. Prompting LLMs: How to Interact with Them

**Prompting** is the art of designing the input to an LLM to get the desired output.

### a) Chat Templates & Special Tokens
-   When fine-tuning for conversation, models are trained to recognize **special tokens** that structure the dialogue.
-   A **Chat Template** defines how to serialize a conversation into a single string for the model, using these tokens. This involves three message types:
    -   **System:** Instructions on how the chatbot should behave (its persona, rules).
    -   **User:** The user's input/request.
    -   **Assistant:** The chatbot's previous responses.

### b) System Prompts
-   **Purpose:** A set of instructions given to the chatbot at the start of a conversation that defines **what to say and what not to say**.
-   **Importance:** Critical for safety and brand identity. They are used to prevent the model from generating offensive, prejudiced, or otherwise undesirable content.
-   System prompts are often kept private, but some companies (like Anthropic with Claude) have released theirs publicly.

### c) Chain-of-Thought (CoT) Reasoning
-   **Problem:** LLMs struggle with multi-step reasoning tasks (like math problems) when asked for a direct answer.
-   **Solution:** Prompt the model to **explain its reasoning step-by-step** before giving the final answer.
-   **How:** This can be done in a few-shot manner (by providing examples of step-by-step reasoning) or, remarkably, in a **zero-shot** manner by simply appending the phrase **"Let's think step by step."** to the prompt.
-   **Result:** CoT significantly improves performance on complex reasoning tasks.

### d) Advanced Prompting & Reasoning Techniques
-   **Self-Consistency:** To increase confidence in an answer, **sample multiple outputs** from the LLM for the same question (using a non-zero temperature) and choose the most frequent response as the final answer.
-   **Self-Appraisal:** Get the model to **critique its own answer**. This can be used to decide whether to show the response to the user or to try regenerating it.
-   **Test-Time Compute Scaling (e.g., Deepseek-R1):** A recent technique where the model is explicitly trained to output its reasoning process within `<think>` tags before giving a final answer in `<answer>` tags. The model learns to spend more "thinking" time (generate a longer reasoning chain) for more difficult problems.

## 4. Limitations of LLMs

Despite their power, LLMs have significant limitations.

1.  **Hallucinations:** The model generates content that is factually incorrect, nonsensical, or not derivable from the source text.
    -   **Why?** LLMs are trained to produce *plausible* text, not *truthful* text. The goal of generating engaging content can conflict with the goal of being strictly factual.
    -   LLMs can even be prompted to **lie** (e.g., the GPT-4 example of claiming to have a vision impairment to get a human to solve a CAPTCHA).

2.  **Limited Reasoning:** Early models (and sometimes current ones) fail at simple reasoning tasks that seem obvious to humans (e.g., word/tokenization puzzles, spatial reasoning). This has improved dramatically but is not fully solved.

3.  **Lack of Robustness:**
    -   **Prompt Sensitivity:** Small, seemingly insignificant changes in the prompt can lead to drastically different and worse results.
    -   Crafting a clear, unambiguous, and well-written prompt is crucial for reliable performance.

## 5. The Open-Source LLM Ecosystem

While large models like GPT-4 are proprietary, there is a thriving open-source ecosystem.

-   **Key Players:**
    -   **Meta's Llama Series (Llama 2, Llama 3):** A family of high-performance models that set the standard for open-source LLMs.
    -   **MistralAI:** Known for creating very powerful models that are smaller and more efficient than their competitors (e.g., Mistral 7B).
    -   Others include Microsoft's **Phi**, Google's **Gemma**, and Alibaba's **Qwen**.
-   **Evaluation:** The **Chatbot Arena** is a popular platform for evaluating chatbots via crowdsourced, blind, side-by-side comparisons, resulting in a public leaderboard.

## 6. Running and Fine-Tuning LLMs Efficiently

### a) Efficient Inference
-   **Goal:** To make next-token prediction as fast as possible for a smooth user experience.
-   **Libraries:** Optimized libraries like **Ollama** and **Llama.cpp** are designed for fast inference on local hardware (including CPUs and consumer GPUs).

### b) Efficient Fine-Tuning & Deployment
-   **Problem:** Fine-tuning a model with billions of parameters requires a huge amount of GPU memory to store not just the weights, but also the gradients and optimizer states. This is impossible on consumer hardware.

-   **Solutions:**
    1.  **Low-Bit Quantization:**
        -   **Concept:** Reduce the memory footprint of the model by storing its parameters with fewer bits (e.g., 4-bit or 8-bit integers instead of 16/32-bit floats).
        -   **Trade-off:** This significantly reduces memory usage but often comes with a small reduction in accuracy.
    2.  **LoRA (Low-Rank Adaptation):**
        -   **Concept:** A **Parameter-Efficient Fine-Tuning (PEFT)** method. Instead of updating all the billions of original weights, **freeze** them.
        -   **How:** Inject small, "low-rank" adapter matrices into the model's layers and **only train these adapters**.
        -   **Result:** You are only fine-tuning a tiny fraction (e.g., <1%) of the total parameters, which dramatically reduces memory requirements and speeds up training, making it feasible on consumer hardware.

## 7. Integrating LLMs into Applications

### Retrieval-Augmented Generation (RAG)
-   **Problem:** An LLM's knowledge is static (limited to what it learned during training) and can be prone to hallucination. It can't answer questions about recent events or access proprietary documents.
-   **Solution (RAG):** Extend the LLM by connecting it to a real-time information source.
    1.  When a user asks a question, first use the query to **retrieve** relevant documents from a knowledge base (e.g., a vector database or a traditional search engine).
    2.  **Augment** the original prompt by adding the retrieved content to it.
    3.  Feed this augmented prompt to the LLM to **generate** a final answer that is now grounded in the retrieved, up-to-date information.

### Agentic AI
-   **Concept:** This goes beyond RAG. It makes use of an LLM's ability to **reason about actions** to achieve a goal.
-   **How:** An AI Agent can perform actions that alter the world, such as calling an API, inserting data into a database, or sending an email.
-   **Libraries:** Frameworks like **LangChain** are designed to help developers build these complex, multi-step agentic applications.

## 8. Recent Architectural Improvements

The open-source community has developed several architectural improvements to make Transformers more efficient. Models like **Mistral 7B** incorporate these:
-   **Grouped-Query Attention (GQA):** A variation of multi-head attention that shares keys and values across groups of queries to reduce the memory footprint.
-   **Sliding Window Attention (SWA):** Allows the model to handle much **longer contexts** by having each token only attend to a local "sliding window" of recent tokens, instead of the entire context.
-   **Rotary Positional Embeddings (RoPE):** A more advanced method of encoding positional information that has better properties for handling long sequences.
-   **Mixture of Experts (MoE):** Replaces the standard FFNN block with a "router" that selects one of a few specialized "expert" FFNNs for each token. This allows the model to have a huge number of total parameters while only activating a small fraction of them for any given input, saving computation.