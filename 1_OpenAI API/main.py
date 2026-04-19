import argparse
import os
from openai import OpenAI

# 1. Parse command-line arguments
parser = argparse.ArgumentParser(description="OpenAI CLI tool")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o-mini)")
parser.add_argument("--user_content", type=str, required=True, help="User message content")
parser.add_argument("--system_content", type=str, help="System message content")
parser.add_argument("--max_completion_tokens", type=int, default=100, help="Maximum number of tokens in the response")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--cost", action="store_true", help="Enable cost estimation")
args = parser.parse_args()

# 2. Load API key securely (environment variable) using setx OPENAI_API_KEY "sk-xxxxxxxxxxxxxxxx" in your environment variables
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
else:
    client = OpenAI(api_key=api_key) 

# 3. Create request message
response = client.chat.completions.create(
    model=args.model,
    max_completion_tokens=args.max_completion_tokens,
    temperature=args.temperature,
    messages=[
        {"role": "system", "content": args.system_content} if args.system_content else None,
        {"role": "user", "content": args.user_content}]
    )

# 4. Print response
print(response.choices[0].message.content)

# 5. Estimate cost
if args.cost:
    # Note: Actual costs may vary based on the model and usage. Always refer to OpenAI's pricing page for the most accurate information.
    input_token_price = 0.15 / 1_000_000
    output_token_price = 0.6 / 1_000_000

    # Extract input and output token usage from response
    input_tokens = response.usage.prompt_tokens
    output_tokens = args.max_completion_tokens

    # Calculate cost
    cost = (input_tokens * input_token_price + output_tokens * output_token_price)
    print(f"\nEstimated cost: ${cost}")

"""python3 main.py --cost --max_completion_tokens 100 --temperature 0.7 --model gpt-4o-mini
 --user_content "I want to learn to speak Dutch. Create a study plan for me."
 --system_content "You are a helpful assistant that creates study plans for learning Dutch."
 """