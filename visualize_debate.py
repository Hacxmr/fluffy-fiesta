import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import re

def load_jsonl(filepath):
    """Load JSONL file and parse JSON lines"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def extract_final_answer(text):
    """Extract final answer from text"""
    match = re.search(r'\{final answer:\s*(\d+(?:\.\d+)?)\}', text)
    if match:
        return float(match.group(1))
    return None

def analyze_debate_history(filepath, name):
    """Analyze debate history and compute metrics"""
    data = load_jsonl(filepath)
    
    results = {
        'name': name,
        'num_samples': len(data),
        'accuracy_by_round': [],
        'consensus_by_round': [],
        'agreement_by_round': []
    }
    
    num_rounds = len(data[0]) if data else 0
    
    for round_idx in range(num_rounds):
        correct_count = 0
        total_count = 0
        consensus_counts = defaultdict(int)
        
        for sample_idx, sample in enumerate(data):
            if str(round_idx) in sample:
                round_data = sample[str(round_idx)]
                
                if 'debate_answer_iscorr' in round_data:
                    if round_data['debate_answer_iscorr']:
                        correct_count += 1
                    total_count += 1
                
                if 'final_answers' in round_data:
                    answer = round_data.get('debate_answer', None)
                    if answer is not None:
                        consensus_counts[answer] += 1
        
        # Calculate accuracy for this round
        accuracy = correct_count / total_count if total_count > 0 else 0
        results['accuracy_by_round'].append(accuracy)
        
        # Calculate consensus (highest count answer)
        if consensus_counts:
            max_consensus = max(consensus_counts.values())
            consensus_pct = max_consensus / total_count if total_count > 0 else 0
            results['consensus_by_round'].append(consensus_pct)
    
    return results

# Load both datasets
sparse_data = analyze_debate_history(
    'out/history/gsm8k_50__qwen2.5-7b_N=3_R=3_SPARSE.jsonl',
    'Sparse Communication'
)
full_data = analyze_debate_history(
    'out/history/gsm8k_50__qwen2.5-7b_N=3_R=3.jsonl',
    'Full Communication'
)

# Create visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy across rounds
rounds = range(len(sparse_data['accuracy_by_round']))
axes[0].plot(rounds, sparse_data['accuracy_by_round'], 'o-', label='Sparse', linewidth=2, markersize=8)
axes[0].plot(rounds, full_data['accuracy_by_round'], 's-', label='Full', linewidth=2, markersize=8)
axes[0].set_xlabel('Debate Round', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Answer Correctness Across Rounds', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1.05])

# Plot 2: Consensus agreement across rounds
axes[1].plot(rounds, sparse_data['consensus_by_round'], 'o-', label='Sparse', linewidth=2, markersize=8)
axes[1].plot(rounds, full_data['consensus_by_round'], 's-', label='Full', linewidth=2, markersize=8)
axes[1].set_xlabel('Debate Round', fontsize=12)
axes[1].set_ylabel('Consensus Agreement %', fontsize=12)
axes[1].set_title('Agent Agreement on Final Answer', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('out/debate_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: out/debate_comparison.png")

# Create summary statistics table
print("\n" + "="*60)
print("DEBATE PERFORMANCE SUMMARY")
print("="*60)
print(f"\n{'Metric':<30} {'Sparse':<15} {'Full':<15}")
print("-"*60)

# Final round accuracy
print(f"{'Final Accuracy (Round 3)':<30} {sparse_data['accuracy_by_round'][-1]:.2%}{'':<7} {full_data['accuracy_by_round'][-1]:.2%}")

# Average accuracy
avg_sparse = np.mean(sparse_data['accuracy_by_round'])
avg_full = np.mean(full_data['accuracy_by_round'])
print(f"{'Average Accuracy':<30} {avg_sparse:.2%}{'':<7} {avg_full:.2%}")

# Final consensus
print(f"{'Final Consensus Agreement':<30} {sparse_data['consensus_by_round'][-1]:.2%}{'':<7} {full_data['consensus_by_round'][-1]:.2%}")

# Improvement from round 0 to final
sparse_improvement = sparse_data['accuracy_by_round'][-1] - sparse_data['accuracy_by_round'][0]
full_improvement = full_data['accuracy_by_round'][-1] - full_data['accuracy_by_round'][0]
print(f"{'Accuracy Improvement (R0->R3)':<30} {sparse_improvement:+.2%}{'':<7} {full_improvement:+.2%}")

print("\n" + "="*60)
