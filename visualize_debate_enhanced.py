import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from collections import defaultdict
import re

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.size'] = 11

def load_jsonl(filepath):
    """Load JSONL file and parse JSON lines"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def analyze_debate_history(filepath):
    """Analyze debate history and compute detailed metrics per round"""
    data = load_jsonl(filepath)
    
    num_rounds = len(data[0]) if data else 0
    metrics = {
        'accuracy': [],
        'consensus': [],
        'num_samples': len(data)
    }
    
    for round_idx in range(num_rounds):
        correct_count = 0
        total_count = 0
        consensus_counts = defaultdict(int)
        
        for sample in data:
            if str(round_idx) in sample:
                round_data = sample[str(round_idx)]
                
                if 'debate_answer_iscorr' in round_data:
                    if round_data['debate_answer_iscorr']:
                        correct_count += 1
                    total_count += 1
                
                if 'debate_answer' in round_data:
                    answer = round_data['debate_answer']
                    if answer is not None:
                        consensus_counts[answer] += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        max_consensus = max(consensus_counts.values()) if consensus_counts else 0
        consensus_pct = max_consensus / total_count if total_count > 0 else 0
        
        metrics['accuracy'].append(accuracy)
        metrics['consensus'].append(consensus_pct)
    
    return metrics

# Load both datasets
sparse_data = analyze_debate_history('out/history/gsm8k_50__qwen2.5-7b_N=3_R=3_SPARSE.jsonl')
full_data = analyze_debate_history('out/history/gsm8k_50__qwen2.5-7b_N=3_R=3.jsonl')

num_rounds = len(sparse_data['accuracy'])
rounds = list(range(num_rounds))

# Create a 2x2 grid with round-by-round breakdowns
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

# Color scheme (matching visualize_debate.py)
colors_sparse = '#1f77b4'  # Blue (matplotlib default)
colors_full = '#ff7f0e'    # Orange (matplotlib default)

# ============ ROW 1: Accuracy Across Rounds ============
ax1 = fig.add_subplot(gs[0, :])
width = 0.35
x_pos = np.arange(num_rounds)

bars1 = ax1.bar(x_pos - width/2, sparse_data['accuracy'], width, label='Sparse Communication', 
                color=colors_sparse, alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax1.bar(x_pos + width/2, full_data['accuracy'], width, label='Full Communication',
                color=colors_full, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.0%}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax1.set_xlabel('Debate Round', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax1.set_title('Answer Correctness by Round', fontsize=14, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'Round {i}' for i in rounds])
ax1.set_ylim([0, 1.0])
ax1.legend(fontsize=12, loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# ============ ROW 2: LEFT - Round-by-Round Comparison Table ============
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('tight')
ax2.axis('off')

table_data = []
table_data.append(['Round', 'Sparse Acc.', 'Full Acc.', 'Diff', 'Sparse Cons.', 'Full Cons.'])

for i in range(num_rounds):
    acc_diff = full_data['accuracy'][i] - sparse_data['accuracy'][i]
    diff_str = f"{acc_diff:+.1%}"
    
    table_data.append([
        f'Round {i}',
        f"{sparse_data['accuracy'][i]:.1%}",
        f"{full_data['accuracy'][i]:.1%}",
        diff_str,
        f"{sparse_data['consensus'][i]:.1%}",
        f"{full_data['consensus'][i]:.1%}"
    ])

table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.12, 0.15, 0.15, 0.12, 0.18, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#34495E')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(6):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#ECF0F1')
        else:
            table[(i, j)].set_facecolor('#FFFFFF')

ax2.set_title('Performance Metrics', fontsize=13, fontweight='bold', pad=20)

# ============ ROW 2: RIGHT - Consensus Agreement ============
ax3 = fig.add_subplot(gs[1, 1])

ax3.plot(rounds, sparse_data['consensus'], 'o-', label='Sparse', color=colors_sparse, 
         linewidth=2.5, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
ax3.plot(rounds, full_data['consensus'], 's-', label='Full', color=colors_full,
         linewidth=2.5, markersize=10, markeredgecolor='black', markeredgewidth=1.5)

for i, (s, f) in enumerate(zip(sparse_data['consensus'], full_data['consensus'])):
    if s > 0.02:
        ax3.text(i - 0.12, s + 0.06, f'{s:.0%}', ha='center', fontweight='bold', fontsize=10, 
                color=colors_sparse)
    if f > 0.02:
        ax3.text(i + 0.12, f + 0.06, f'{f:.0%}', ha='center', fontweight='bold', fontsize=10,
                color=colors_full)

ax3.set_xlabel('Debate Round', fontsize=13, fontweight='bold')
ax3.set_ylabel('Agent Agreement %', fontsize=13, fontweight='bold')
ax3.set_title('Consensus Agreement Across Rounds', fontsize=13, fontweight='bold')
ax3.set_xticks(rounds)
ax3.set_xticklabels([f'Round {i}' for i in rounds])
ax3.set_ylim([-0.05, 1.0])
ax3.legend(fontsize=11, loc='upper right')
ax3.grid(True, alpha=0.3)

# ============ ROW 3: LEFT - Accuracy Trajectory ============
ax4 = fig.add_subplot(gs[2, 0])

# Calculate improvement
sparse_improvement = sparse_data['accuracy'][-1] - sparse_data['accuracy'][0]
full_improvement = full_data['accuracy'][-1] - full_data['accuracy'][0]

# Line plot with markers
ax4.plot(rounds, sparse_data['accuracy'], 'o-', label='Sparse Communication', 
         color=colors_sparse, linewidth=3, markersize=12, markeredgecolor='black', markeredgewidth=2)
ax4.plot(rounds, full_data['accuracy'], 's-', label='Full Communication',
         color=colors_full, linewidth=3, markersize=12, markeredgecolor='black', markeredgewidth=2)

# Add value labels
for i, (s, f) in enumerate(zip(sparse_data['accuracy'], full_data['accuracy'])):
    ax4.text(i - 0.12, s + 0.008, f'{s:.0%}', ha='center', fontweight='bold', fontsize=10, color=colors_sparse)
    ax4.text(i + 0.12, f + 0.008, f'{f:.0%}', ha='center', fontweight='bold', fontsize=10, color=colors_full)

ax4.set_xlabel('Debate Round', fontsize=13, fontweight='bold')
ax4.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax4.set_title('Accuracy Trajectory Through Debate', fontsize=13, fontweight='bold')
ax4.set_xticks(rounds)
ax4.set_xticklabels([f'Round {i}' for i in rounds])
ax4.set_ylim([0.75, 1.0])
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)

# ============ ROW 3: RIGHT - Summary Stats ============
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

summary_text = f"""
SUMMARY STATISTICS

Sparse Communication:
  • Final Accuracy: {sparse_data['accuracy'][-1]:.1%}
  • Average Accuracy: {np.mean(sparse_data['accuracy']):.1%}
  • Improvement (R0→R{num_rounds-1}): {sparse_improvement:+.1%}
  • Final Consensus: {sparse_data['consensus'][-1]:.1%}

Full Communication:
  • Final Accuracy: {full_data['accuracy'][-1]:.1%}
  • Average Accuracy: {np.mean(full_data['accuracy']):.1%}
  • Improvement (R0→R{num_rounds-1}): {full_improvement:+.1%}
  • Final Consensus: {full_data['consensus'][-1]:.1%}

Key Insights:
  • Full communication maintains higher average accuracy
  • Sparse communication shows more improvement over rounds
  • Both achieve similar final accuracy
  • Low consensus suggests diverse reasoning patterns
"""

ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes, fontfamily='monospace',
        fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Main title
fig.suptitle('Multi-Agent Debate Performance Analysis\nGSM8K Dataset (N=50 samples, 3 agents, Qwen 2.5-7B)',
            fontsize=16, fontweight='bold', y=0.995)

plt.savefig('out/debate_analysis_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Visualization saved to: out/debate_analysis_enhanced.png")

# Also create individual round comparison charts
fig_rounds, axes_rounds = plt.subplots(1, num_rounds, figsize=(5*num_rounds, 5))
if num_rounds == 1:
    axes_rounds = [axes_rounds]

for round_idx in range(num_rounds):
    ax = axes_rounds[round_idx]
    
    metrics = ['Accuracy', 'Consensus']
    sparse_vals = [sparse_data['accuracy'][round_idx], sparse_data['consensus'][round_idx]]
    full_vals = [full_data['accuracy'][round_idx], full_data['consensus'][round_idx]]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sparse_vals, width, label='Sparse', color=colors_sparse, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, full_vals, width, label='Full', color=colors_full,
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.0%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Round {round_idx}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim([0, 1.0])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

fig_rounds.suptitle('Performance by Round', fontsize=15, fontweight='bold')
fig_rounds.tight_layout()
plt.savefig('out/debate_rounds_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Round comparison saved to: out/debate_rounds_comparison.png")

plt.close('all')
print("All visualizations generated successfully!")
