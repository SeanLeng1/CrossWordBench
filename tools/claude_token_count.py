import json
import os

import anthropic
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# reference https://docs.anthropic.com/en/docs/build-with-claude/token-counting
client = anthropic.Anthropic()

folder_path = '../eval_results/claude-3-7-sonnet-20250219/english/7x7/text_cot'
reasoning_tokens = 0
count = 0
for filename in tqdm(os.listdir(folder_path), total=len(os.listdir(folder_path))):
    if filename.endswith('.json'):
        with open(os.path.join(folder_path, filename), 'r') as f:
            data = json.load(f)
            for item in data:
                    completion = data['metadata']['choices'][0]['message']['reasoning_content']
                    response = client.messages.count_tokens(
                        model="claude-3-7-sonnet-20250219",
                        system="You are a scientist",
                        messages=[
                            {
                                "role": "assistant",
                                "content": completion
                            }
                        ],
                    )
                    reasoning_tokens += json.loads(response.to_json())['input_tokens']
                    count += 1

reasoning_tokens = reasoning_tokens / count
print(f"Average reasoning tokens: {reasoning_tokens}")