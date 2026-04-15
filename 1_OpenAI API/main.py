import argparse
import os
from openai import OpenAI

# 1. Parse command-line arguments
parser = argparse.ArgumentParser(description="OpenAI CLI tool")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt-4o-mini)")
parser.add_argument("--role", type=str, required=True, help="Role (user/system/assistant)")
parser.add_argument("--content", type=str, required=True, help="Message content")
args = parser.parse_args()

# 2. Load API key securely (environment variable) using setx OPENAI_API_KEY "sk-xxxxxxxxxxxxxxxx"
api_key = os.getenv("OPENAI_API_KEY")
if not api_key: raise ValueError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=api_key) 

# 3. Create request dynamically
response = client.chat.completions.create(
    model=args.model,
    messages=[{"role": args.role, "content": args.content}])

# 4. Print response
print(response.choices[0].message.content)

# python3 main.py --model gpt-4o-mini --role user --content "Write a short paragraph about me"