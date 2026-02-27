import json
import os
import re
from tqdm import tqdm
import csv

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# CONFIG
SAMPLE_CSV_PATH = "notebooks/csv/sample_for_llm_shortReadme.csv"
OUTPUT_CSV = "notebooks/data/sample_agent_repos_llm_filtered_withShortReadme_Qwen.csv"
FEW_SHOT_JSON = "notebooks/data/few_shot_examples_shortReadme.json"

MODEL =  "Qwen/Qwen3-14B" #"Qwen/Qwen3-30B-A3B-Thinking-2507" #"Qwen/Qwen3-8B"   #"Qwen/Qwen3-4B"                 
CACHE_DIR = os.path.expanduser("~/hf_cache")
MAX_NEW_TOKENS = 100  # short enough for JSON output

# Device
if torch.cuda.is_available():
    device_map = "auto"
    torch_dtype = torch.float16
elif torch.backends.mps.is_available():
    device_map = {"": "mps"}
    torch_dtype = torch.float16
else:
    device_map = {"": "cpu"}
    torch_dtype = torch.float32

# Categories
CATEGORIES = [
    "collection of llm agent projects",
    "collection of datasets",
    "agentic framework",
    "agentic benchmark",
    "foundation model",
    "llm-based agentic system",
    "documents about llm agents",
    "other"
]
category_list = "\n".join(f"- {c}" for c in CATEGORIES)

# -------------------------------
# LOAD MODEL
# -------------------------------
print("Loading tokenizer and model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    cache_dir=CACHE_DIR,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    cache_dir=CACHE_DIR,
    device_map=device_map,
    torch_dtype=torch_dtype,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
print("Model loaded successfully.")

# -------------------------------
# FEW-SHOT EXAMPLES
# -------------------------------
def build_few_shot(path):
    with open(path, "r") as f:
        examples = json.load(f)

    messages = []

    for i, ex in enumerate(examples, 1):
        readme = ex.get("readme_content", "") or ""

        example_text = f"""
### Example {i}
Input:
name: "{ex['full_name']}"
description: "{ex['description']}"
topics: {ex['topics']}
readme:
\"\"\"
{readme}
\"\"\"

Output:
{{"category":"{ex['category']}"}}"""

        messages.append({"role": "user", "content": example_text})

    return messages

FEW_SHOT_MESSAGES = build_few_shot(FEW_SHOT_JSON)

# -------------------------------
# CLASSIFICATION
# -------------------------------
def classify_row(row, retries=2):
    name = row.get("full_name", "")
    desc = row.get("description", "")
    topics = row.get("topics", "")
    readme = row.get("readme_snippet", "")

    main_prompt = f"""
You are an expert classifier of GitHub repositories related to LLMs and AI agents.
IMPORTANT: Assign EXACTLY ONE high-level category that best describes the repository in a single JSON object in this exact format:
{{"category": "<category-name>"}}
Preferred categories:
{category_list}

Rules:
• The list above is a GUIDELINE, not a hard limit  
• Prefer using an existing category unless it is clearly incorrect 
• ALL new categories MUST start with the prefix "new:"
• New categories must be:
    - 1–5 words
    - lowercase
    - noun phrase
• Avoid creating unnecessary new categories
• Do NOT write any explanations, reasoning, or extra text.
• If uncertain between two categories, choose the more concrete one

Classification guidance:
• If the repository is a curated list, awesome list, or resource list → it is a collection  
• Choose the MOST SPECIFIC collection category if possible   
• Use ALL available information:
    - name
    - description
    - topics
    - readme 
• Prioritize repository purpose over implementation details

### Now classify this repository:
name: "{name}"
description: "{desc}"
topics: {topics}
readme:
\"\"\"
{readme}
\"\"\"

Output format:
{{"category": "<category-name>"}}
"""

    # Combine few-shot examples + current repo
    messages = FEW_SHOT_MESSAGES + [{"role": "user", "content": main_prompt}]
    
    """
    print ("Constructed messages for LLM:")
    for msg in messages:
        print(f"msg: {msg['role']} - {msg['content']}...")
    """
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # disables reasoning block
        )

        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        #print("Decoded LLM output:", decoded)

    except Exception as e:
        print("Generation error:", e)
        if retries > 0:
            return classify_row(row, retries - 1)
        return "other"

    # Parse JSON safely
    try:
        match = re.search(r"\{.*\}", decoded, re.DOTALL)
        if not match:
            print("No JSON found in LLM output.")
            return "other"

        parsed = json.loads(match.group())
        cat = parsed.get("category", "").strip().lower()

        if not cat:
            return "other"

        if cat in CATEGORIES or cat.startswith("new:"):
            return cat

        return cat

    except Exception as e:
        print("Error parsing JSON:", e)
        return "other"


if __name__ == "__main__":
    df = pd.read_csv(SAMPLE_CSV_PATH)
    print(f"Classifying {len(df)} repositories...")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Prepare CSV writing
    fieldnames = list(df.columns) + ["category"]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # write header once

        for _, row in tqdm(df.iterrows(), total=len(df)):
            row_dict = row.to_dict()

            category = classify_row(row_dict)
            row_dict["category"] = category

            writer.writerow(row_dict)
            csvfile.flush()   # VERY important for long runs / HPC safety

    print("Saved streamed results to:", OUTPUT_CSV)