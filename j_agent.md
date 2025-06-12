# Exam Review: Spoken Conversation, Dialog & Agentic AI

This guide provides a complete summary of conversational AI, from the foundational principles of human dialog to classic chatbot architectures and the modern paradigm of Agentic AI.

## 1. Fundamentals of Human Conversation

Understanding the properties of human dialog is key to building systems that can mimic it.

-   **Turn-Taking:** Conversations consist of participants taking turns to speak. A core challenge for a spoken dialog system is knowing when a user has finished talking (**end-point detection**) and when to handle interruptions (**barge-in**).
-   **Speech Acts:** Every utterance is a kind of **action**. The main types are:
    -   **Constatives:** Stating a fact or belief (e.g., *"I need to travel in May."*).
    -   **Directives:** Trying to get the hearer to do something (e.g., *"Turn up the music!"* or the question *"What day do you want to travel?"* which directs the user to answer).
    -   **Commissives:** Committing the speaker to a future action (e.g., *"I promise to be there."*).
    -   **Acknowledgments:** Expressing an attitude about the hearer's action (e.g., *"Thanks."*).
-   **Common Ground:** For a conversation to succeed, participants must establish a shared understanding of the world state. This is achieved by **grounding** each other's utterances, which means acknowledging that information has been received and understood (e.g., by saying "Okay" or repeating key information).
-   **Conversational Initiative:** Who is driving the conversation?
    -   **User Initiative:** The user asks questions, and the system responds.
    -   **System Initiative:** The system asks a series of questions to fill a form, and the user responds.
    -   **Mixed Initiative:** The most natural (and complex) form, where control shifts back and forth.
-   **Inference:** Humans rarely speak literally. We expect our conversational partners to **infer** meaning from context (e.g., replying to "What day do you want to travel?" with "I have a meeting from the 12th to the 15th" implies the travel dates).

## 2. Types of Conversational Agents (Chatbots)

There are two broad categories of conversational agents:

1.  **Open-Domain Chatbots:**
    -   **Goal:** To mimic unstructured, general human conversation.
    -   **Purpose:** Primarily for entertainment, companionship, or therapy. They are not designed to complete a specific task.

2.  **Task-Oriented Dialogue Systems:**
    -   **Goal:** To help a user complete a specific task.
    -   **Purpose:** Goal-based agents for booking flights, setting timers, controlling smart home devices, etc.

## 3. Open-Domain Chatbots

### a) Classic Rule-Based Chatbots
These early systems relied on handcrafted rules and pattern matching.
-   **ELIZA (1966):** Simulated a Rogerian psychologist. Its "trick" was to use simple pattern-matching rules to transform and reflect the user's statements back as questions (e.g., "I am feeling sad." -> "I AM SORRY TO HEAR YOU ARE FEELING SAD."). ELIZA has no real understanding.
-   **PARRY (1971):** A more complex rule-based system that simulated a person with paranoid schizophrenia. It added a **model of a mental state** (variables for Anger, Fear, Mistrust) that influenced its responses. PARRY was the first chatbot to pass a version of the Turing Test.

### b) Modern Corpus-Based Chatbots
These systems learn from massive datasets of human conversations.
-   **Data Sources:** Movie dialogues, social media posts, telephone conversation transcripts.
-   **Two Main Approaches:**
    1.  **Retrieval-Based:** Given a user's query, the system searches a large corpus of past conversations to find the most similar turn and returns the response that followed it.
    2.  **Generation-Based:** Uses a generative language model (like GPT) to create a new response from scratch, conditioned on the conversation history.

-   **Challenges of Generative Models:** They have a tendency to produce repetitive or dull, "safe" responses (e.g., "I don't know," "I'm OK."). This can be mitigated with techniques like diversity-enhanced beam search or training with diversity-focused objectives.
-   **Retrieval-Augmented Generation (RAG):** Modern chatbots often use a hybrid approach where they retrieve relevant factual information (e.g., from Wikipedia) and incorporate it into the generated response to be more informative.

## 4. Task-Oriented Dialogue Agents

### a) Classic Frame-Based Architecture (e.g., GUS, 1977)
This is a foundational, rule-based architecture for task-oriented systems.
-   **Core Components:**
    -   **Frame:** An action or function the agent can perform (e.g., `buy_book`).
    -   **Slots:** The variables or arguments for that frame (e.g., `author`, `title`, `credit-card`).
    -   **Values:** The information provided by the user to fill the slots.
-   **Process:** The system's goal is to **fill all the slots in a frame** by asking the user a series of questions. Once the frame is full, the agent performs the action (e.g., queries a database, makes a purchase).
-   **NLU Component:** This involves three steps:
    1.  **Domain Classification:** Is the user talking about flights or alarm clocks?
    2.  **Intent Determination:** What does the user want to do? (e.g., `SHOW-FLIGHTS`).
    3.  **Slot Filling:** Extracting values from the user's utterance (e.g., "Boston", "Tuesday").

### b) Modern Dialogue-State Architecture
This is a more sophisticated, modular architecture that forms the basis for most modern task-oriented systems.
-   **Four Main Components:**
    1.  **Natural Language Understanding (NLU):** Extracts the user's **dialogue act** and any **slot-fillers** from their utterance.
    2.  **Dialogue State Tracker (DST):** Maintains the current state of the conversation, including user constraints and what has been said so far. It also handles **corrections** if the user indicates the system has misunderstood.
    3.  **Dialogue Policy (DP):** The "brain" of the system. Based on the current dialogue state, it decides what the agent should do or say next (e.g., ask a clarifying question, query a database, confirm information).
    4.  **Natural Language Generation (NLG):** Takes the formal action from the Dialogue Policy and converts it into a natural language sentence to say to the user.

-   **Delexicalization:** To train the NLG component, slot values in the training data (e.g., "Au Midi") are replaced with generic placeholder tokens (e.g., "restaurant_name"). The model learns to generate a delexicalized sentence, which is then "relexicalized" with the real values.

## 5. Agentic AI: The Modern Paradigm

Agentic AI refers to autonomous systems that can make decisions and perform tasks. LLMs are now being used as the "reasoning engine" for these agents.

### Agentic Design Patterns
These are common architectures for building LLM-powered workflows.
1.  **Reflection:** The LLM checks its own output for quality, correctness, or adherence to rules (e.g., checking for offensive content).
2.  **Tool Use:** The LLM is given access to external tools (e.g., a calculator, a search engine API) and can decide when and how to use them.
3.  **ReAct (Reasoning and Acting):** A specific pattern for tool use. The LLM cycles through **Thought** (what should I do next?), **Action** (call a specific tool), and **Observation** (process the tool's output) to achieve a goal.
4.  **Planning:** The LLM generates a multi-step plan to solve a complex problem and then executes it, replanning if a step fails.
5.  **Multi-agent:** Multiple LLMs with different specialized capabilities interact with each other to solve a problem.

### How LLMs Use Tools
-   **Instruction Tuning:** LLMs are trained to understand and generate a special syntax for tool calls.
-   **Process:**
    1.  The agent is prompted with a list of available tools and their descriptions.
    2.  The LLM generates text containing a specific "Action" (e.g., `Action: Wikipedia`) and "Action Input" (e.g., `Action Input: number of moons of Mars`).
    3.  A backend parser (like in the **LangChain** toolkit) intercepts this text, calls the actual tool with the input, and gets the result.
    4.  The result is fed back to the LLM as an "Observation," and the reasoning cycle continues.

### Retrieval-Augmented Generation (RAG) as an Agentic System
-   **RAG is the most common application of an Agentic AI pattern.**
-   It uses a **retrieval tool** to overcome the LLM's limitations (static knowledge, hallucinations).
-   **Problem:** LLMs can't answer questions about recent events or proprietary data.
-   **RAG Solution:**
    1.  The agent **retrieves** real-time content from a knowledge base.
    2.  It **augments** the user's prompt with this retrieved content.
    3.  It **generates** an answer that is grounded in the provided facts.

## 6. Evaluating Dialogue Systems

-   **Open-Domain Chatbots:** Evaluation is difficult and subjective. Automated metrics like BLEU correlate poorly with human judgment. The primary method is **human evaluation**, assessing dimensions like:
    -   **Engagingness:** Would you want to talk to it for a long time?
    -   **Interestingness:** Is the conversation boring or interesting?
    -   **Humanness:** Does it sound like a human?
    -   **LLM-as-a-Judge:** A recent trend is to use a powerful LLM (like GPT-4) to evaluate and score the output of other chatbots.

-   **Task-Based Agents:** Evaluation is more objective.
    -   **Task Success:** Did the agent successfully complete the task (e.g., was the correct meeting added to the calendar)?
    -   **Slot Error Rate:** How many slots were filled incorrectly or missed?
    -   **Efficiency:** How long did the dialogue take (time, number of turns)?

## 7. Ethical Implications of Chatbots

-   **Anthropomorphism:** Users tend to treat chatbots as if they are human, leading to emotional involvement and potential risks to mental health.
-   **Safety:** A chatbot must not give harmful advice (e.g., in mental health or in-vehicle contexts).
-   **Representational Harm & Bias:** Chatbots trained on large internet corpora can learn and amplify harmful stereotypes, racism, and misogyny (e.g., the Microsoft Tay chatbot disaster).
-   **Privacy:**
    -   **Accidental Leakage:** Models trained on user data may inadvertently leak sensitive information like passwords or personal details.
    -   **Intentional Leakage:** Systems may be designed to send user data to developers or advertisers. Privacy-preserving designs are crucial.