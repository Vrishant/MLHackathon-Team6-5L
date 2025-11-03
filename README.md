# ML Hackathon Team 6 - Hangman AI Agent

## Team Members
- Vrishant Bhalla: PES12UG23CS706
- Vishnu Teja: PES12UG23CS698
- Vishwambhara R Hebbalalu: PES12UG23CS700
- Vivek Bhat: PES12UG23CS703

This repository contains a Jupyter Notebook (`MLHackathon_Team6_5L.ipynb`) that implements a hybrid AI agent for playing the game of Hangman. The agent combines Hidden Markov Model (HMM) probabilities for letter positioning with Q-Learning reinforcement learning to make intelligent guesses.

## Overview

The notebook trains and evaluates an AI agent that plays Hangman by:
1. Building an HMM model from a corpus of words to estimate letter probabilities at each position.
2. Training a Q-Learning agent that uses HMM probabilities as a heuristic to guide exploration.
3. Evaluating the trained agent on test words and corpus samples to compute a final score.

The final score is calculated as: `(successful_games * 2000) - (wrong_guesses * 5) - (repeated_guesses * 2)`.

## Prerequisites

- Python 3.x
- Required libraries: `re`, `pickle`, `os`, `collections`, `random`, `numpy`
- Data files: `corpus.txt` (training words), `test.txt` (evaluation words)

## Notebook Structure

The notebook is divided into three main cells:

### Cell 1: HMM Model Training

**Purpose:** Builds a probabilistic model of letter positions in words using Hidden Markov Model principles.

**Steps:**
1. Loads words from `corpus.txt` and preprocesses them (lowercase, alphanumeric only, unique).
2. Counts word lengths and letter frequencies at each position for each length.
3. Computes position probabilities: For each word length and position, calculates the probability of each letter appearing.
4. Saves the HMM model (`models/hmm_model.pkl`) containing:
   - `position_probs`: Nested dict of length → position → letter → probability
   - `length_counts`: Counter of word lengths

**Output:** Prints "hmm saved" upon completion.

### Cell 2: Q-Learning Agent Training

**Purpose:** Trains a reinforcement learning agent to play Hangman using Q-Learning, guided by HMM probabilities.

**Key Components:**

- **HangmanEnv Class:**
  - Initializes with word list and max lives (default 8).
  - `reset()`: Selects random word, initializes game state.
  - `step(letter)`: Processes a guess, returns new pattern, reward, done flag.
  - Reward system:
    - Correct guess: +12 + 4 * number of new letters revealed
    - Wrong guess: -10
    - Repeated guess: -4
    - Win: +150
    - Loss: -100

- **QLearningAgent Class:**
  - Uses Q-table for state-action values.
  - State representation: (current pattern, sorted guessed letters string)
  - Action selection: Epsilon-greedy with HMM probability boost (weight 10).
  - HMM integration: Sums position probabilities for available letters in blank positions.

**Training Process:**
- Trains for 40,000 episodes (increased from 12,000 for better convergence).
- Epsilon decays from 1.0 to 0.01.
- Prints progress every 500 episodes: episode, win rate, average reward, epsilon.
- Saves trained agent (`models/rl_agent.pkl`).

**Output:** Prints "RL agent trained and saved." upon completion.

### Cell 3: Agent Evaluation

**Purpose:** Evaluates the trained hybrid agent on test and corpus words, computes final score.

**Key Components:**
- Reloads HMM and RL models.
- Redefines classes for evaluation (simplified step method).
- Evaluates on:
  - All test words from `test.txt`.
  - Sample of 1000 corpus words.
- Agent runs in pure exploitation mode (epsilon=0).
- Tracks: successes, wrong guesses, repeated guesses.

**Scoring:**
- Success: +2000 points
- Wrong guess: -5 points
- Repeated guess: -2 points

**Output:** Detailed evaluation results including success rates, guess counts, and final score.

## File Structure

```
.
├── MLHackathon_Team6_5L.ipynb  # Main notebook
├── corpus.txt                  # Training word corpus
├── test.txt                    # Test words for evaluation
├── models/                     # Directory for saved models
│   ├── hmm_model.pkl          # Trained HMM model
│   └── rl_agent.pkl           # Trained Q-Learning agent
├── instructions.pdf           # Competition instructions
└── README.md                  # This file
```

## How to Run

1. Ensure `corpus.txt` and `test.txt` are in the working directory.
2. Run Cell 1 to train the HMM model.
3. Run Cell 2 to train the RL agent (may take several minutes).
4. Run Cell 3 to evaluate the agent and get the final score.

## Key Insights

- **HMM Component:** Provides position-aware letter probabilities, helping the agent prioritize likely letters in specific word positions.
- **RL Component:** Learns optimal guessing strategies through trial and error, balancing exploration and exploitation.
- **Hybrid Approach:** Combines statistical modeling (HMM) with adaptive learning (RL) for robust performance.
- **Evaluation:** Tests on unseen words to ensure generalization beyond training corpus.

## Performance Notes

- Training time: ~10-15 minutes for 40,000 episodes on typical hardware.
- Memory usage: Q-table grows with state space; may become large for complex games.
- Hyperparameters: Tuned for balance between HMM guidance and RL learning (HMM boost factor = 10 during training, 8 during evaluation).

## Potential Improvements

- Increase training episodes for better convergence.
- Implement more sophisticated state representations.
- Add word frequency weighting to HMM probabilities.
- Experiment with different reward structures or RL algorithms (e.g., SARSA).
