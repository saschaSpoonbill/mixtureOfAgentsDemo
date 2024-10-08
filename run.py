# Mixture-of-Agents
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

import asyncio
import os
from together import AsyncTogether, Together

# Initialize Together API clients. Paid plan required.
client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

# Define the user prompt
user_prompt = "What are the most important things to know about the city of Karlsruhe?"

# List of reference models to be used
reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]

# Model used for aggregating responses
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

# System prompt for the aggregator model
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""

async def run_llm(model):
    """Run a single LLM call with a reference model."""
    # Make an async API call to the model
    response = await async_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.7,
        max_tokens=512,
    )
    print(model)
    return response.choices[0].message.content

async def main():
    # Run all reference models concurrently and gather results
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])

    # Create a streaming response from the aggregator model
    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {"role": "system", "content": aggreagator_system_prompt},
            {"role": "user", "content": ",".join(str(element) for element in results)},
        ],
        stream=True,
    )

    # Print the aggregated response as it streams
    for chunk in finalStream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

# Run the main async function
asyncio.run(main())