# Advanced Mixture-of-Agents – 3 layers
import asyncio
import os
from dotenv import load_dotenv
import together
from together import AsyncTogether, Together

# Load environment variables from .env file
load_dotenv()

# Initialize Together API clients. Paid plan required.
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
async_client = AsyncTogether(api_key=os.getenv("TOGETHER_API_KEY"))

user_prompt = "What are the 3 most important things to know about SAP?"
reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.

Responses from models:"""
layers = 3

def getFinalSystemPrompt(system_prompt, results):
    """Construct a system prompt for layers 2+ that includes the previous responses to synthesize."""
    return (
        system_prompt
        + "\n"
        + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(results)])
    )

async def run_llm(model, prev_response=None):
    """Run a single LLM call with a model while accounting for previous responses and rate limits."""
    for sleep_time in [1, 2, 4]:
        try:
            # Create message list based on previous responses
            messages = (
                [
                    {
                        "role": "system",
                        "content": getFinalSystemPrompt(
                            aggreagator_system_prompt, prev_response
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ]
                if prev_response
                else [{"role": "user", "content": user_prompt}]
            )
            # Send request to the model
            response = await async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=512,
            )
            print("Model: ", model)
            break
        except together.error.RateLimitError as e:
            # If rate limited, wait and retry
            print(e)
            await asyncio.sleep(sleep_time)
    return response.choices[0].message.content

async def main():
    """Run the main loop of the MOA process."""
    # First layer: Get responses from all reference models
    results = await asyncio.gather(*[run_llm(model) for model in reference_models])

    # Process additional layers (if any)
    for _ in range(1, layers - 1):
        results = await asyncio.gather(
            *[run_llm(model, prev_response=results) for model in reference_models]
        )

    # Final layer: Aggregate results with the aggregator model
    finalStream = client.chat.completions.create(
        model=aggregator_model,
        messages=[
            {
                "role": "system",
                "content": getFinalSystemPrompt(aggreagator_system_prompt, results),
            },
            {"role": "user", "content": user_prompt},
        ],
        stream=True,
    )
    # Stream the final response
    for chunk in finalStream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)

# Run the main program
asyncio.run(main())