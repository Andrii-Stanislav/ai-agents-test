import os
from dotenv import load_dotenv

from huggingface_hub import InferenceClient
from transformers import AutoTokenizer

from _system_prompt import SYSTEM_PROMPT


load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

client = InferenceClient(model="meta-llama/Llama-3.2-3B-Instruct")

# Since we are running the "text_generation", we need to add the right special tokens.
prompt=f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}
<|eot_id|><|start_header_id|>user<|end_header_id|>
What's the weather in London ?
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

messages=[
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London ?"},
  ]

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt_2 = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, use_auth_token=True)

print('prompt_2: ', prompt_2)

output_2 = client.text_generation(
    prompt=prompt_2,
    max_new_tokens=100,
)

print('output_2: ', output_2)