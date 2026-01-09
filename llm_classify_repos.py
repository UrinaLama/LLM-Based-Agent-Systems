import json
import pandas as pd
import ollama
from tqdm import tqdm

tqdm.pandas()

CSV_PATH = "Mining_final/csv/github_agent_repos_python_20251225.csv"
OUTPUT_CSV = "Mining_final/csv/github_agent_repos_llm_categorized.csv"
CHECKPOINT_CSV = "Mining_final/csv/github_agent_repos_llm_checkpoint_10.csv"

MODEL = "qwen3:4b"


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
    "autonomous agent",
    "handbook"
]

CATEGORY_BULLETS = "\n".join(f"- {c}" for c in CATEGORIES)

# FEW-SHOT EXAMPLES
FEW_SHOT = """
### Example 1
Input:
name: "kaushikb11/awesome-llm-agents"
description: "A curated list of awesome LLM agents frameworks."
topics: ["agents", "langchain", "llm"]
readme: "Awesome LLM Agent Frameworks. A curated list of awesome LLM frameworks and agent development tools."

Output:
{"category":"collection of frameworks"}

### Example 2
Input:
name: "THUDM/AgentBench"
description: "A Comprehensive Benchmark to Evaluate LLMs as Agents (ICLR'24)"
topics: ["chatgpt", "gpt-4", "llm", "llm-agent"]
readme: "AgentBench is a benchmark designed to evaluate LLMs as agents across multiple tasks."

Output:
{"category":"benchmark"}

### Example 3
Input:
name: "langchain-ai/langchain"
description: "The platform for reliable agents."
topics: ["framework", "agents", "llm", "rag", "multiagent"]
readme: "LangChain is a framework for building agents and LLM-powered applications."

Output:
{"category":"framework"}

### Example 4
Input:
name: "zai-org/GLM-4.5"
description: "Agentic, Reasoning, and Coding Foundation Models"
topics: ["llm", "foundation-model", "reasoning"]
readme: "GLM-4.5 is a family of foundation models designed for reasoning, coding, and agentic behavior."

Output:
{"category":"foundation model"}

### Example 5
Input:
name: "oxbshw/LLM-Agents-Ecosystem-Handbook"
description: "One-stop handbook for building, deploying, and understanding LLM agents."
topics: ["ai-agent", "llm", "rag", "voice-agent"]
readme: "A curated handbook covering LLM agents, tutorials, ecosystem guides, and evaluation tools."

Output:
{"category":"handbook"}
"""

# PROMPT
def build_prompt(row):
    name = row.get("name", "")
    desc = row.get("description", "")
    topics = row.get("topics", "")
    readme = row.get("readme_snippet", "")

    category_list = "\n".join(f"- {c}" for c in CATEGORIES)

    return f"""
    You are an expert classifier of GitHub repositories related to LLMs and AI agents.

    Your task:
    Assign EXACTLY ONE high-level category that best describes the repository.

    Preferred category list (use these when applicable):
    {category_list}

    Rules:
    • The list above is a GUIDELINE, not a hard limit  
    • If NONE of the listed categories fit well, you MAY create a NEW short category  
    • New categories must be:
        - 2–5 words
        - lowercase
        - noun phrase
        • Avoid creating unnecessary new categories
        • Be consistent across similar repositories
        • Do NOT output explanations
        • Output ONLY valid JSON

    Classification guidance:
    • If the repository is a curated list, awesome list, or resource list → it is a collection  
    • Choose the MOST SPECIFIC collection category if possible  
    • If collection type is unclear → use "collection"  
    • Frameworks, benchmarks, handbooks, and models should NOT be labeled as collections  

    Use ALL available information:
    - name
    - description
    - topics
    - README snippet

    ### Examples:
    {FEW_SHOT}

    ### Now classify this repository:
    name: "{name}"
    description: "{desc}"
    topics: {topics}
    readme: "{readme}"

    Output format:
    {{"category": "<category-name>"}}
    """


# CLASSIFIER
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

    try:
        start = content.find("{")
        end = content.rfind("}")
        parsed = json.loads(content[start:end + 1])
        cat = parsed.get("category", "").strip().lower()

        # exact match
        if cat in CATEGORIES:
            return cat

        # new category
        if cat.startswith("new:"):
            return cat.replace("new:", "").strip()

        # fallback
        return cat if cat else "other"

    except Exception:
        return "other"


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    print("Classifying repositories:", len(df))

    df["category"] = ""

    checkpoint_written = False

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        cat = classify_row(row.to_dict())
        df.at[idx, "category"] = cat

        # ONE-TIME checkpoint after first 10 rows
        if idx == 9 and not checkpoint_written:
            df.iloc[:10].to_csv(CHECKPOINT_CSV, index=False)
            tqdm.write(f"Checkpoint saved after 10 rows → {CHECKPOINT_CSV}")
            checkpoint_written = True

    # Final save
    df.to_csv(OUTPUT_CSV, index=False)
    print("Saved final output:", OUTPUT_CSV)