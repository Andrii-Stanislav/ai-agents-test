import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


client = InferenceClient(model="meta-llama/Llama-3.2-3B-Instruct")

# If we now add the special tokens related to Llama3.2 model, the behaviour changes and is now the expected one.
prompt="""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

The capital of france is<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

#  1 way
output = client.text_generation(
    prompt,
    max_new_tokens=100,
)

print(output)

#  2 way
output_2 = client.chat.completions.create(
    messages=[
        {"role": "user", "content": "The capital of france is"},
    ],
    stream=False,
    max_tokens=1024,
)

print(output_2.choices[0].message.content)