import feedparser
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1. Fetch the RSS Feed (Same as before)
url = "https://www.nrk.no/nyheter/siste.rss"
feed = feedparser.parse(url)

news_items = []
for entry in feed.entries[:10]:
    title = entry.title
    summary = entry.get('description', 'No description available.')
    news_items.append(f"Title: {title}\nSummary: {summary}")

news_context = "\n\n".join(news_items)

# 2. Set Up Qwen 3.5 9B in 4-bit
model_id = "Qwen/Qwen3.5-9B" 

print(f"Loading {model_id} in 4-bit. This will fit on your RTX 3080...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# This is the magic that mirrors what LM Studio does!
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=quantization_config
)
# ---------------------------------------------------------
# 3. Format the Prompt (Updated System Prompt)
# ---------------------------------------------------------
messages = [
    {
        "role": "system",
        "content": "You are a highly efficient news assistant. Output ONLY the final summary in concise, well-structured bullet points. Do not include any thinking process, internal monologue, or analysis steps."
    },
    {
        "role": "user", 
        "content": f"Please read the following Norwegian news articles from NRK and provide a summary of the main events in a few bullet points:\n\n{news_context}"
    }
]

prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# ---------------------------------------------------------
# 4. Generate the Summary (Increased Token Limit)
# ---------------------------------------------------------
print("Reading the news and generating summary...")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=1024, # Increased from 400 to give it plenty of room
    temperature=0.3,
    do_sample=True
)

input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_length:]
summary_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n" + "="*50)
print("NRK LATEST NEWS SUMMARY (Powered by Qwen 3.5 9B)")
print("="*50)
print(summary_output)