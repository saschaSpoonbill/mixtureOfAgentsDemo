# Mixture of Agents (MoA) Implementation Demo

This repository contains two Python scripts demonstrating the implementation of the Mixture of Agents (MoA) approach using the Together API.

## Programs

1. `run.py`: Implements a basic 2-layer MoA system with 4 reference models and 1 aggregator model.
2. `runAdvanced.py`: Implements an advanced 3+ layer MoA system, allowing for deeper refinement and synthesis of responses.

Both programs use multiple language models to answer a user query, then synthesize the responses into a final, comprehensive answer.

## Requirements

- Python 3.7+
- `together` Python package
- `python-dotenv` package

## Setup

1. Install the required packages:
   ```
   pip install together python-dotenv
   ```

2. Create a `.env` file in the root directory of the project and add your Together API key:
   ```
   TOGETHER_API_KEY=your_api_key_here
   ```

   **Note:** You will likely need a paid plan from Together.ai to access the required API functionality.
