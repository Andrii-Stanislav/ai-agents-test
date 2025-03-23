import os
from dotenv import load_dotenv

from huggingface_hub import InferenceClient

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

# Dummy Tool function
def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures! \n"

get_weather('London')

output = client.text_generation(
    prompt,
    max_new_tokens=100,
    stop=["Observation:"] # Let's stop before any actual function is called
)

new_prompt = prompt + output + get_weather('London')

print('output: ', output)

final_output = client.text_generation(
    prompt=new_prompt,
    max_new_tokens=200,
)

print('final_output: ', final_output)