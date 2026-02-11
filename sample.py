import pandas as pd

# Paths
CSV_PATH = "notebooks/csv/github_agent_repos_python_final.csv"
SAMPLE_CSV = "notebooks/csv/sample_for_llm.csv"
SAMPLE_SIZE = 200

# Load the full CSV
df = pd.read_csv(CSV_PATH)

# Randomly sample 200 rows
sample_df = df.sample(n=SAMPLE_SIZE, random_state=42)  # random_state ensures reproducibility

# Reset index (optional)
sample_df = sample_df.reset_index(drop=True)

# Save to new CSV
sample_df.to_csv(SAMPLE_CSV, index=False)

print(f"Saved {SAMPLE_SIZE} random rows to {SAMPLE_CSV}")