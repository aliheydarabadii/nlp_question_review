# Exam Review: Speech Detection & Generation

This guide provides a complete summary of the key concepts, pipelines, and models for both Speech-to-Text (ASR) and Text-to-Speech (TTS) systems.

## 1. Fundamentals of Human Speech

### The Building Blocks of Speech
-   **Phonemes:** The basic, distinct units of sound in a language that distinguish one word from another (e.g., the /p/ and /b/ sounds in "pat" vs. "bat").
-   **Vowels:** Sounds produced *without* restricting the vocal tract. They are periodic signals (have a repeating waveform).
-   **Consonants:** Sounds produced by *partially or completely closing* the vocal tract. They are often aperiodic (noise-like).
    -   *Fricatives:* Forcing air through a narrow channel (e.g., the 'ch' in "watch").
    -   *Bursts/Stops:* A rapid transition from a complete closure (e.g., the 'd' in "dime").

### Source-Filter Model of Phonation
This is a simplified model of how humans produce speech sounds.
-   **Source:** The **larynx/glottis** produces a series of periodic pulses of air. This contains the fundamental frequency (pitch) but carries little information for identifying the specific sound.
-   **Filter:** The **vocal tract** (mouth, nose, tongue position) acts as a filter, shaping the sound pulses from the source to create different phonemes. The filter carries most of the important information for speech recognition.

## 2. From Audio Signal to Spectrogram

Speech is a sound wave, which is a 1D time-series of air pressure values. To analyze it, we must convert it into a 2D representation that shows how frequencies change over time.

### The Spectrogram
-   **Definition:** A 2D visualization of a sound signal, showing **time** on the x-axis, **frequency** on the y-axis, and the **amplitude** (intensity/loudness) of each frequency at each time, represented by color or brightness.

### How to Create a Spectrogram: Short-Time Fourier Transform (STFT)
The STFT is the core process for creating a spectrogram.
1.  **Chunking:** The continuous audio signal is divided into small, **overlapping chunks** (frames), typically around 25ms long.
2.  **Windowing:** Each chunk is multiplied by a **windowing function** (e.g., a **Hamming window**) to reduce "spectral leakage"—artifacts that appear at the borders of each chunk.
3.  **Fourier Transform:** A **Fast Fourier Transform (FFT)** is applied to each windowed chunk to calculate its frequency components (i.e., the amplitude of all the different sine waves that make up that chunk of sound).
4.  **Combine:** The resulting frequency vectors from all chunks are stacked together to form the final 2D spectrogram.

### Pre-Emphasis Filter
-   **What it is:** A simple filter applied to the raw audio signal *before* the STFT.
-   **Purpose:** It **amplifies the high frequencies** in the signal. This is useful for balancing the spectrum (since high frequencies usually have lower amplitude) and can improve the signal-to-noise ratio.

### Mel Spectrogram: The Human-Centric View
A standard spectrogram uses linear scales for frequency and amplitude, but human hearing is **logarithmic**. A **Mel Spectrogram** modifies the standard spectrogram to better match human perception.

1.  **Limit Frequency Range:** Human hearing is most sensitive below 8kHz, so we often cap the frequency range.
2.  **Logarithmic Frequency (The Mel Scale):** The y-axis is converted to the **Mel scale**, which is logarithmic. This gives more resolution to lower frequencies, where humans are better at distinguishing pitch differences.
3.  **Logarithmic Amplitude (Decibels):** The amplitude is represented in **decibels (dB)**, which is a logarithmic scale that better matches our perception of loudness.

**The Mel Spectrogram is the standard input representation for most modern speech processing systems.**

## 3. Speech-to-Text (STT) / Automatic Speech Recognition (ASR)

**Task:** To convert an audio signal (waveform) into a text transcript.

### The ASR Pipeline: Classic vs. Modern
-   **Classic ASR (Pre-Deep Learning):**
    1.  Extract features from the audio, typically **Mel-Frequency Cepstrum Coefficients (MFCCs)**, which are derived from the Mel Spectrogram.
    2.  Use a **Hidden Markov Model (HMM)** to model the sequence of phonemes and a **Gaussian Mixture Model (GMM)** to model the probability of seeing certain MFCC features for each phoneme.

-   **Modern ASR (Deep Learning):** End-to-end models have replaced the HMM/GMM pipeline. The key challenge they solve is the **alignment problem**: different parts of the audio signal have different lengths but map to the same phoneme (e.g., the 'i' in "diner" vs. the 'i' in "dinner").
    -   **Solution:** Use a **Sequence-to-Sequence (Seq2Seq) model**. The encoder processes the audio input (which has one length), and the decoder generates the text output (which has a different length).

### State-of-the-Art ASR Models
Modern ASR systems are based on the **Transformer** architecture.

-   **Wav2vec (2020):**
    -   A powerful Transformer-based model that works directly on **raw audio**.
    -   It uses a Convolutional Neural Network (CNN) front-end to create initial embeddings from the waveform, which are then processed by a Transformer.

-   **Whisper (2022):**
    -   The current state-of-the-art model from OpenAI.
    -   It uses a **Mel Spectrogram** as its input representation.
    -   It is a large-scale Transformer model trained in a **weakly supervised** manner on a massive, multi-lingual dataset from the web. This allows it to be extremely robust.
    -   It is a **multi-task** model, capable of transcription, translation, and language identification simultaneously.

### Evaluating ASR Systems
-   **Word Error Rate (WER):** The primary metric for ASR. It's based on the edit distance between the system's output and the correct transcript.
    -   `WER = (Substitutions + Deletions + Insertions) / Total_Words_in_Correct_Transcript`
    -   **Lower WER is better.**
-   **Sentence Error Rate (SER):** The percentage of sentences that have at least one error.

### Advanced ASR Topics
-   **Speaker Dependence:** ASR systems work better when personalized to a specific speaker. This can be done by:
    -   **Vocal Tract Length Normalization:** Warping the frequency axis of the spectrogram to account for the physical size of a speaker's vocal tract.
    -   **Modifying the Acoustic Model:** Fine-tuning a pre-trained model on a small amount of data from a new speaker.
-   **Handling Non-words:** Dealing with coughs, "um"s, "uh"s, and background noise by creating special tokens/phones for them and including these in the training data transcripts.

## 4. Text-to-Speech (TTS) / Speech Synthesis

**Task:** To convert a string of text into a natural-sounding audio waveform.

### The Modern TTS Pipeline
Modern TTS is typically a 3-stage process:
1.  **Text to Phoneme:** The input text string is converted into a sequence of phonemes. This is the **Text Analysis** step.
2.  **Phoneme to Mel Spectrogram:** A model (usually a Seq2Seq model) takes the phoneme sequence and generates a Mel Spectrogram.
3.  **Mel Spectrogram to Audio:** A **vocoder** takes the Mel Spectrogram and synthesizes the final audio waveform.

### Stage 1: Text Analysis & Normalization
Before phoneme conversion, the text must be normalized. This includes:
-   **Expanding Abbreviations:** "Dr." -> "Doctor".
-   **Verbalizing Numbers and Symbols:** "224" -> "two twenty-four"; "$1750" -> "one thousand seven hundred and fifty dollars". This process is context-dependent.
-   **Homograph Disambiguation:** Determining the correct pronunciation of words with the same spelling but different sounds based on context (e.g., "I like to play the **bass** /beɪs/ guitar" vs. "I'm cooking sea **bass** /bæs/").

### Stage 2: Prosody & Phoneme-to-Spectrogram
-   **Prosody:** The rhythm, stress, and intonation of speech. A good TTS system must model prosody to sound natural. This includes:
    -   **Prominence (Stress):** Predicting which words should be emphasized ("pitch accent").
    -   **Intonation:** Predicting the rise and fall of pitch (F0 contour), e.g., to make a sentence sound like a question.
    -   **Duration:** Predicting the length of each phoneme, which changes based on context (e.g., vowels are longer before a pause).
-   **Tacotron2 (2018):** A famous and highly successful LSTM-based Seq2Seq model for generating Mel Spectrograms from text/phonemes.

### Stage 3: The Vocoder
-   **WaveNet:** A powerful vocoder model used in Tacotron2. It is an **autoregressive, dilated convolution-based** model that generates the audio waveform one sample at a time, resulting in very high-fidelity, natural-sounding speech.

### Evaluating TTS Systems
TTS evaluation is subjective and requires human testers.
-   **Intelligibility:** Can listeners correctly understand what was said?
-   **Quality:** How natural, fluent, and clear is the speech?
    -   **Mean Opinion Score (MOS):** Raters score the quality of an utterance on a scale of 1 to 5.
    -   **AB Test:** Raters listen to the same utterance generated by two different systems (A and B) and choose which one sounds better.