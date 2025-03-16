import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob

sns.set(style="whitegrid")

def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found.")
        return []

result_files = glob.glob('results/speed_test*.jsonl')

all_data = []

for file_path in result_files:
    data = load_jsonl(file_path)
    if data:
        all_data.extend(data)

df = pd.DataFrame(all_data)

grouped = df.groupby('input_size').agg({
    'micrograd_time': 'mean',
    'no_torch_time': 'mean',
    'pytorch_time': 'mean'
}).reset_index()

plt.figure(figsize=(12, 8))
plt.title('Framework Speed Comparison, 4 trial average', fontsize=16)
plt.xlabel('Input Vector Size (number of 32 bit floats)', fontsize=14)
plt.ylabel('Time (seconds, log scale)', fontsize=14)

sns.lineplot(x='input_size', y='micrograd_time', data=grouped, label='micrograd', color='blue', marker='o')
sns.lineplot(x='input_size', y='no_torch_time', data=grouped, label='NoTorch', color='green', marker='s')
sns.lineplot(x='input_size', y='pytorch_time', data=grouped, label='PyTorch', color='red', marker='^')

plt.yscale('log')
plt.grid(True, which="both", ls="--", alpha=0.7)
plt.legend(fontsize=12)
plt.tight_layout()

plt.savefig('imgs/results.png', dpi=300)
print("Plot saved as 'results.png'")

plt.show()