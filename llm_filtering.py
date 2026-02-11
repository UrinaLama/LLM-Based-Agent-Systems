import json
import os
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SAMPLE_CSV_PATH = "notebooks/csv/sample_for_llm.csv"
OUTPUT_JSON = "notebooks/data/sample_agent_repos_llm_filtered.json"
FEW_SHOT_JSON = "notebooks/data/few_shot_examples.json"

# MODEL CONFIG
MODEL = "Qwen/Qwen3-8B"
MAX_CONTEXT_TOKENS = 32768
MAX_NEW_TOKENS = 100

FEWSHOT_README_TOKENS = 3000      # per example
TARGET_README_TOKENS = 3500       # for the repo being classified
SAFE_CONTEXT_LIMIT = 32000      # hard guard

CACHE_DIR=os.path.expanduser("~/hf_cache")

if torch.cuda.is_available():
    device_map = "auto"
    torch_dtype = torch.float16
elif torch.backends.mps.is_available():
    device_map = {"": "mps"}
    torch_dtype = torch.float16
else:
    device_map = {"": "cpu"}
    torch_dtype = torch.float32

# CATEGORIES
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


# LOAD MODEL
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

model.eval()
print("Model loaded successfully.")

# TOKEN

def truncate_to_tokens(text, max_tokens, tokenizer):
    if not text:
        return ""

    tokens = tokenizer.encode(text, add_special_tokens=False)

    if len(tokens) <= max_tokens:
        return text

    tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# FEW-SHOT EXAMPLES
def build_few_shot(path, tokenizer, max_readme_tokens):
    with open(path, "r") as f:
        examples = json.load(f)

    blocks = []

    for i, ex in enumerate(examples, 1):
        readme = ex.get("readme_content", "") or ""

        readme = truncate_to_tokens(
            readme,
            max_tokens=max_readme_tokens,
            tokenizer=tokenizer
        )

        blocks.append(f"""
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
{{"category":"{ex['category']}"}}
""")

    return "\n".join(blocks)


FEW_SHOT = build_few_shot(
    FEW_SHOT_JSON,
    tokenizer=tokenizer,
    max_readme_tokens=FEWSHOT_README_TOKENS
)


# PROMPT BUILDER
def build_prompt(row):
    name = row.get("name", "")
    desc = row.get("description", "")
    topics = row.get("topics", "")
    readme = row.get("readme_snippet", "")

    readme = truncate_to_tokens(
        readme,
        max_tokens=TARGET_README_TOKENS,
        tokenizer=tokenizer
    )

    return f"""
You are an expert classifier of GitHub repositories related to LLMs and AI agents.

Your task:
Assign EXACTLY ONE high-level category that best describes the repository.

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
• Do NOT explain your answer 
• Output ONLY valid JSON
• If uncertain between two categories, choose the more concrete one

Classification guidance:
• If the repository is a curated list, awesome list, or resource list → it is a collection  
• Choose the MOST SPECIFIC collection category if possible   
• Use ALL available information:
    - name
    - description
    - topics
    - README content
• Prioritize repository purpose over implementation details

### Examples:
{FEW_SHOT}

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


# CLASSIFICATION
def classify_row(row, retries=2):
    prompt = build_prompt(row)

    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )

        # ---- HARD SAFE GUARD ----
        if inputs["input_ids"].shape[1] > SAFE_CONTEXT_LIMIT:
            prompt = truncate_to_tokens(
                prompt,
                SAFE_CONTEXT_LIMIT,
                tokenizer
            )
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False
            )

        inputs = inputs.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                do_sample=False
            )

        decoded = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

    except Exception:
        if retries > 0:
            return classify_row(row, retries - 1)
        return "other"

    try:
        start = decoded.find("{")
        end = decoded.rfind("}")
        parsed = json.loads(decoded[start:end + 1])

        cat = parsed.get("category", "").strip().lower()

        if cat in CATEGORIES:
            return cat

        return cat if cat else "other"

    except Exception:
        return "other"


# MAIN
if __name__ == "__main__":
    df = pd.read_csv(SAMPLE_CSV_PATH)

    print(f"Classifying {len(df)} repositories...")

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    # Open file in append mode
    with open(OUTPUT_JSON, "a") as f:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            row_dict = row.to_dict()

            category = classify_row(row_dict)
            row_dict["category"] = category

            # Write immediately (JSON Lines format)
            f.write(json.dumps(row_dict) + "\n")
            f.flush()   # VERY important on HPC

    print("Saved streamed output to:", OUTPUT_JSON)

