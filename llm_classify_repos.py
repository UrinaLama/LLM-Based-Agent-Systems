import json
import time
import pandas as pd
import ollama
from tqdm import tqdm
tqdm.pandas()

CSV_PATH = "./Mining/csv/agent_repos_filtered_with_stars.csv"
OUTPUT_CSV = "./Mining/csv/agent_repos_llm_classified.csv"
MODEL = "qwen3:4b"

# Final categories 
CATEGORIES = [
    "collection of projects",
    "collection of papers",
    "collection of llm apps",
    "collection of datasets",
    "collection of tutorials",
    "collection",
    "framework",
    "benchmark",
    "foundation model",
    "llm agent",
    "autonomous agent"
    "handbook",
    "other"
]

# ---------- FEW-SHOT EXAMPLES  ---------- #

FEW_SHOT = """
### Example 1
Input:
name: "Awesome LLM Apps"
description: "A curated list of 300+ LLM-powered applications."

Output:
{"category":"collection of llm apps"}

### Example 2
Input:
name: "Awesome AI Papers"
description: "A large collection of recent AI and ML research papers."

Output:
{"category":"collection of papers"}

### Example 3
Input:
name: "DeepSeek V2"
description: "A powerful 100B+ open-source foundation model."

Output:
{"category":"foundation model"}

### Example 4
Input:
name: "AgentBench"
description: "A benchmark for evaluating LLM-based agents."

Output:
{"category":"benchmark"}

### Example 5
Input:
name: "LangChain"
description: "A framework for building LLM-powered applications."

Output:
{"category":"framework"}

### Example 6
Input:
name: "Awesome AI Projects"
description: "A curated list of the best open-source AI-related projects."

Output:
{"category":"collection of projects"}

### Example 7
Input:
name: "Generative AI Handbook"
description: "A practical handbook for working with LLMs."

Output:
{"category":"handbook"}

### Example 8
Input:
name: "Resource Collection"
description: "A list of useful links, datasets, models, and tutorials."

Output:
{"category":"collection"}
"""

# ---------- PROMPT BUILDER ---------- #

def build_prompt(row):
    name = row.get("name", "")
    desc = row.get("description", "")

    return f"""
You are an expert classifier. Assign this repository to EXACTLY ONE category from this list:

{CATEGORIES}

Your job:
• Detect if the repo is a collection (curated list, awesome list, list of X).  
• If it IS a collection, choose the MOST SPECIFIC category (e.g., "collection of papers", "collection of projects").  
• If it is a collection but unclear, use: "collection".  
• If NOT a collection, classify it normally (framework, benchmark, handbook, etc).  
• Output ONLY valid JSON.

{FEW_SHOT}

### Now classify this new input:
name: "{name}"
description: "{desc}"

Output format:
{{"category": "<one-valid-category>"}}
"""

# ---------- CLASSIFICATION ---------- #

def classify_row(row, retries=2):
    prompt = build_prompt(row)

    try:
        resp = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0}
        )
        content = resp["message"]["content"].strip()

    except Exception:
        if retries > 0:
            return classify_row(row, retries - 1)
        return "other"

    # extract JSON
    try:
        start = content.find("{")
        end = content.rfind("}")
        parsed = json.loads(content[start:end+1])
        cat = parsed.get("category", "").lower().strip()

        # match category exactly
        for c in CATEGORIES:
            if c.lower() == cat:
                return c

        if "agent" in cat or "autonomous" in cat:
            return "llm / autonomous agent"

        return cat

    except Exception:
        return "other"

# ---------- MAIN ---------- #

def main():
    df = pd.read_csv(CSV_PATH)
 
    print("Classifying", len(df))
    df["category"] = df.progress_apply(lambda row: classify_row(row.to_dict()), axis=1)

    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved:", OUTPUT_CSV)

if __name__ == "__main__":
    main()
