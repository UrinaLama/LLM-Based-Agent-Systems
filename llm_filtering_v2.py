import json
import pandas as pd
import ollama
from tqdm import tqdm
import os

tqdm.pandas()

SAMPLE_CSV_PATH = "notebooks/csv/sample_for_llm.csv"
OUTPUT_CSV = "notebooks/csv/sample_agent_repos_llm_filtered.csv"
FEW_SHOT_JSON = "notebooks/data/few_shot_examples.json"

MODEL = "qwen3:8b"

CATEGORIES = [
    "collection of llm agent projects", 
    "collection of datasets",
    "agentic framework", 
    "agentic benchmark", 
    "foundation model", 
    "llm based agentic system",
    "documents about llm agents" # (books, chapters, papers ...)
    "other"
]

category_list = "\n".join(f"- {c}" for c in CATEGORIES)


# FEW-SHOT EXAMPLES
def build_few_shot(path="notebooks/data/few_shot_examples.json", max_readme_words=3000):
    with open(path, "r") as f:
        examples = json.load(f)

    blocks = []

    for i, ex in enumerate(examples, 1):
        readme = ex['readme_content'] 

        if not isinstance(readme, str):
            readme = ""

        words = readme.split()
        if len(words) > max_readme_words:
            readme = " ".join(words[:max_readme_words]) 

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

FEW_SHOT = build_few_shot(FEW_SHOT_JSON, max_readme_words=3000)

# PROMPT
def build_prompt(row):
    name = row.get("name", "")
    desc = row.get("description", "")
    topics = row.get("topics", "")
    readme = row.get("readme_snippet", "")

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

        # fallback
        return cat if cat else "other"

    except Exception:
        return "other"


if __name__ == "__main__":
    df = pd.read_csv(SAMPLE_CSV_PATH)

    print("Classifying repositories:", len(df))

    df["category"] = ""

    # reset output file
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    write_header = True

    for idx in tqdm(range(len(df))):
        row = df.iloc[idx].to_dict()
        cat = classify_row(row)
        df.at[idx, "category"] = cat

        row["category"] = cat

        # append row immediately
        pd.DataFrame([row]).to_csv(
            OUTPUT_CSV,
            mode="a",
            header=write_header,
            index=False
        )

        write_header = False

    print("Saved final output:", OUTPUT_CSV)