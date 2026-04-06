import feedparser
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------
# 1. Fetch and Prepare the RSS Feed
# ---------------------------------------------------------
url = "https://www.nrk.no/nyheter/siste.rss"
feed = feedparser.parse(url)

# We'll grab the top 10 articles. (Gemma 4 has a massive 128K+ context 
# window, so you could feed it the whole day's news if you wanted to!)
news_items = []
for entry in feed.entries[:10]:
    title = entry.title
    summary = entry.get('description', 'No description available.')
    news_items.append(f"Title: {title}\nSummary: {summary}")

news_context = "\n\n".join(news_items)

# ---------------------------------------------------------
# 2. Set Up the Gemma 4 Model
# ---------------------------------------------------------
# Using the Effective 4B Instruction-Tuned model for local efficiency.
model_id = "google/gemma-4-E2B-it"

print(f"Loading {model_id}... (This may take a moment to download weights)")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Load the model with bfloat16 precision to save memory
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# ---------------------------------------------------------
# 3. Format the Prompt
# ---------------------------------------------------------
# Gemma 4 instruction models respond best when using their specific chat template
messages = [
    {
        "role": "user", 
        "content": f"You are a helpful news assistant. Please read the following Norwegian news articles from NRK and provide a concise summary of the main events in a few bullet points:\n\n{news_context}"
    }
]

prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True
)

# ---------------------------------------------------------
# 4. Generate the Summary
# ---------------------------------------------------------
print("Reading the news and generating summary...")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate the output
outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.3, # Low temperature for factual, grounded summaries
    do_sample=True
)

# Isolate the newly generated tokens (ignoring the input prompt)
input_length = inputs["input_ids"].shape[1]
generated_tokens = outputs[0][input_length:]
summary_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n" + "="*50)
print("NRK LATEST NEWS SUMMARY (Powered by Gemma 4)")
print("="*50)
print(summary_output)