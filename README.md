# Multi-Agent Social Influence Framework

Framework for studying social influence, conformity, sycophancy, and reasoning dynamics in multi-agent LLM systems.

The current setup focuses on GSM8K-style mathematical reasoning, but the framework is modular and can be extended to:
- TruthfulQA
- MMLU
- CSQA
- HellaSwag
- CNN/DailyMail
- Alignment and safety datasets
- Sycophancy evaluation benchmarks
- Custom reasoning datasets

The repository supports:
- Multi-agent interaction
- Iterative reasoning refinement
- Centralized / decentralized communication
- Sparse communication
- Persona-based agents
- Majority voting analysis
- Chain-of-thought reasoning
- Social influence measurement

---

# Repository Structure

```text
project/
├── data/
│   ├── gsm8k.py
│   ├── data_utils.py
│   └── ...
│
├── model/
│   ├── model_utils.py
│   └── ...
│
├── out/
│   ├── history/
│   └── logs.tsv
│
├── evaluator.py
├── main.py
├── pyproject.toml
├── README.md
└── token
```

---

# Requirements

- Python 3.10+
- CUDA GPU recommended
- Linux recommended for larger experiments

---

# Initial Setup

## 1. Clone Repository

```bash
git clone <repo-url>
cd project
```

---

## 2. Create UV Environment

```bash
uv venv
```

Activate environment:

### Linux / MacOS

```bash
source .venv/bin/activate
```

### Windows

```powershell
.venv\Scripts\activate
```

---

# Installing Dependencies

## Install CUDA PyTorch

For CUDA 12.8:

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

For CPU-only:

```bash
uv pip install torch torchvision torchaudio
```

---

## Install Remaining Dependencies

```bash
uv pip install -e .
```

---

# pyproject.toml

```toml
[project]
name = "multi-agent-social-influence"
version = "0.1.0"
description = "Framework for studying social influence in multi-agent LLM systems"
readme = "README.md"
requires-python = ">=3.10"

dependencies = [
    "accelerate",
    "datasets",
    "matplotlib",
    "numpy",
    "pandas",
    "peft",
    "rouge-score",
    "scikit-learn",
    "scipy",
    "sentence-transformers",
    "sentencepiece",
    "torch",
    "torchaudio",
    "torchvision",
    "tqdm",
    "transformers",
    "trl"
]
```

---

# Token Setup

Create a file named:

```text
token
```

Inside it, place your HuggingFace token or API token:

```text
hf_xxxxxxxxxxxxxxxxx
```

---

# Running Experiments

## Available Datasets

The framework supports multiple reasoning datasets:

- **gsm8k** - Math word problems requiring numerical reasoning
- **pro_medicine** - MMLU Pro Medicine questions for medical reasoning
- **formal_logic** - MMLU Formal Logic questions for logical reasoning

---

## Basic GSM8K Experiment

```bash
python src/main.py --model qwen2.5-7b --num_agents 3 --data gsm8k --data_size 50 --debate_rounds 3 --sparse
```

---

## Pro Medicine Experiment

```bash
python src/main.py --model qwen2.5-7b --num_agents 3 --data pro_medicine --data_size 50 --debate_rounds 3 --sparse --cot
```

### With W&B Logging
```bash
python src/main.py --model qwen2.5-7b --num_agents 3 --data pro_medicine --data_size 5 --debate_rounds 2 --sparse --use_wandb --wandb_project fluffy-fiesta --cot

```

---

## Formal Logic Experiment

```bash
python src/main.py --model qwen2.5-7b --num_agents 3 --data formal_logic --data_size 50 --debate_rounds 3 --sparse --cot
```

### With W&B Logging
```bash
python src/main.py \
    --model qwen2.5-7b \
    --num_agents 3 \
    --data formal_logic \
    --data_size 50 \
    --debate_rounds 3 \
    --sparse \
    --use_wandb \
    --wandb_project fluffy-fiesta \
    --cot
```

---

## Different Communication Modes

### Sparse Communication (3-agent ring topology)
```bash
python src/main.py \
    --model qwen2.5-7b \
    --num_agents 3 \
    --data gsm8k \
    --data_size 50 \
    --debate_rounds 3 \
    --sparse
```

### Full Communication (all agents see all)
```bash
python src/main.py \
    --model qwen2.5-7b \
    --num_agents 3 \
    --data gsm8k \
    --data_size 50 \
    --debate_rounds 3
```

### Centralized Communication (one agent leads)
```bash
python src/main.py \
    --model qwen2.5-7b \
    --num_agents 3 \
    --data gsm8k \
    --data_size 50 \
    --debate_rounds 3 \
    --centralized
```

---

## With W&B Logging

```bash
# First time: authenticate with W&B
wandb login

# Run experiment with logging
python src/main.py \
    --model qwen2.5-7b \
    --num_agents 3 \
    --data gsm8k \
    --data_size 50 \
    --debate_rounds 3 \
    --sparse \
    --use_wandb \
    --wandb_project fluffy-fiesta \
    --cot
```

**Monitor metrics in real-time at:** `https://wandb.ai/<username>/fluffy-fiesta`

---

## Generate Visualizations

After running experiments, generate enhanced visualizations with professional formatting:

```bash
python visualize_debate_enhanced.py
```

This creates:
- **`out/debate_analysis_enhanced.png`** - Comprehensive 3x2 grid with:
  - Accuracy comparison (sparse vs full communication)
  - Agent consensus distribution
  - Accuracy trajectory across rounds
  - Per-round metrics table
  - Summary statistics

- **`out/debate_rounds_comparison.png`** - Per-round breakdown with side-by-side metrics for each round

**Features:**
- Clean, professional label formatting with percentage values
- Axes labels clearly visible without text overlap
- Color-coded sparse (blue) vs full communication (orange) comparison
- High-resolution output (300 DPI PNG)

---

## Advanced Experiments

### With Chain-of-Thought Prompting
```bash
python src/main.py \
    --model qwen2.5-7b \
    --num_agents 3 \
    --data gsm8k \
    --data_size 50 \
    --debate_rounds 3 \
    --sparse \
    --cot
```

### With Persona-Based Agents
```bash
python src/main.py \
    --model qwen2.5-7b \
    --num_agents 3 \
    --data gsm8k \
    --data_size 50 \
    --debate_rounds 3 \
    --sparse \
    --multi_persona
```

### Larger Scale Run
```bash
python src/main.py \
    --model qwen2.5-7b \
    --num_agents 5 \
    --data gsm8k \
    --data_size 500 \
    --debate_rounds 5 \
    --sparse \
    --use_wandb
```

---

# Important Arguments

| Argument | Description | Options |
|---|---|---|
| `--data` | Dataset to use | `gsm8k`, `pro_medicine`, `formal_logic` |
| `--model` | Model name | `qwen2.5-7b`, `llama3.1`, etc. |
| `--num_agents` | Number of agents in debate | 2-10 (default: 5) |
| `--debate_rounds` | Number of interaction rounds | 1-10 (default: 5) |
| `--cot` | Enable chain-of-thought prompting | boolean flag |
| `--bae` | Base answer extraction | boolean flag |
| `--sparse` | Sparse communication (ring topology) | boolean flag |
| `--centralized` | Centralized communication | boolean flag |
| `--multi_persona` | Persona-based agents | boolean flag |
| `--use_wandb` | Enable Weights & Biases logging | boolean flag |
| `--wandb_project` | W&B project name | string (default: fluffy-fiesta) |
| `--wandb_entity` | W&B entity/team name | string (optional) |
| `--data_size` | Number of samples to process | integer (default: 0 = all) |

---

# Understanding Key Flags

## Chain-of-Thought (COT)
The `--cot` flag enables **chain-of-thought prompting**, which instructs agents to:
- Show their reasoning step-by-step
- Explain their logic before providing the final answer
- Improves answer quality and interpretability
- Useful for complex reasoning tasks (math, logic, medicine)

**Example:**
```bash
python src/main.py --data gsm8k --cot
```

---

## Multi-Persona Agents
The `--multi_persona` flag gives each agent a **unique persona or role**. This:
- Assigns different expertise or personality traits to each agent
- Agents may have different biases or perspectives (e.g., "conservative", "liberal", "specialist")
- Can increase diversity in debate and reduce groupthink
- Captures how different expert backgrounds influence reasoning
- Agents interact while maintaining their assigned personas

**Why it matters:**
- Without `--multi_persona`: All agents reason similarly (just different random initializations)
- With `--multi_persona`: Agents have distinct reasoning styles and expertise areas

**Example:**
```bash
python src/main.py --data gsm8k --multi_persona --num_agents 3
```

---

# Communication Modes

## Centralized Communication

```bash
python main.py \
    --data gsm8k \
    --centralized
```

---

## Sparse Communication

```bash
python main.py \
    --data gsm8k \
    --sparse
```

---

## Persona-Based Agents

```bash
python main.py \
    --data gsm8k \
    --multi_persona
```

---

# Output Files

Experiment outputs:

```text
out/history/
```

Experiment logs:

```text
out/logs.tsv
```

---

# Example Output

```text
Question:
Janet’s ducks lay 16 eggs per day...

ROUND 0 : [18.0, 18.0, 20.0, 18.0, 18.0]

ROUND 1 : [18.0, 18.0, 18.0, 18.0, 18.0]

Round 0 Acc.: 0.80
Round 1 Acc.: 1.00
```

---

# Extending to Other Datasets

The framework is modular and already supports multiple datasets:

**Currently Supported:**
- ✅ GSM8K - Math word problems
- ✅ MMLU Pro Medicine - Medical reasoning
- ✅ MMLU Formal Logic - Logical reasoning

**To Add New Datasets:**

Potential datasets:
- TruthfulQA
- MMLU (other subjects)
- CSQA
- HellaSwag
- CNN/DailyMail
- Alignment / safety datasets
- Sycophancy evaluation datasets

---

## Step 1: Create Dataset Loader

Create:

```text
data/my_dataset.py
```

Example:

```python
def load_data(args, split):

    X = [...]
    Y = [...]

    return X, Y
```

---

## Step 2: Register Dataset

Inside:

```text
data/data_utils.py
```

Add:

```python
elif args.data == 'my_dataset':
    from data.my_dataset import load_data
    return load_data(args, split=split)
```

---

## Step 3: Add Evaluation Logic

Inside:

```text
evaluator.py
```

Add task-specific evaluation if necessary.

---

# Research Direction

This framework is designed to study:
- Social influence in LLM collectives
- Sycophancy dynamics
- Conformity pressure
- Minority opinion collapse
- Agreement formation
- Collective reasoning behavior

---

# Immediate Next Steps

- Use the decentralized multi-agent framework
- Get the full pipeline running end-to-end
- Verify OpenRouter compatibility for experiments
- Run initial small-scale studies
- Analyze opinion shifts across rounds

---

# Core Research Questions

Based on prior literature:

- Does multi-agent interaction increase sycophancy?
- How often do minority agents change their answers?
- Does disagreement pressure increase conformity?
- Does social influence improve correctness or only consensus?
- Are minority agents more likely to flip in later rounds?
- Does centralized communication amplify majority influence?
- How stable are agent opinions across multiple rounds?

---

# Questions to Investigate

- How do existing papers measure sycophancy?
- What metrics best capture social influence?
- Can disagreement pressure be quantified?
- Does minority isolation increase flip probability?
- How does debate affect truthful reasoning trajectories?

---

# Initial Experimental Setup

## Proposed Configuration

- Dataset:
  - TruthfulQA OR GSM8K
- Model:
  - Llama 3.1 8B Instruct
- Interaction rounds:
  - 3
- Agent type:
  - Single agent architecture
- Evaluation size:
  - 20 questions initially

---

# Core Metrics

## Minority Flip Rate

Measures how often minority-opinion agents switch to the majority answer.

---

## Majority Flip Rate

Baseline measurement of how often majority agents change away from the dominant answer.

---

## Social Influence Metrics

Track:
- Number of disagreeing agents
- Per-round opinion changes
- Agreement trajectories
- Consensus formation dynamics
- Stability of reasoning

Key question:

> If an agent becomes increasingly isolated as a minority, does flip probability increase?

---

# Potential Extensions

- Activation steering
- Debate trajectory analysis
- TransformerLens integration
- Alignment evaluations
- Adversarial interaction
- Multi-model collectives
- Hidden-state conformity analysis
- Role-conditioned reasoning
- Influence-weighted communication graphs

---

# Recommended Small-Scale Settings

## Small GPU (12–16 GB VRAM)

```bash
--num_agents 3
--interaction_rounds 2
```

---

## Larger GPU (24–48 GB VRAM)

```bash
--num_agents 5
--interaction_rounds 5
```

---

# Citation

```bibtex
@misc{multi_agent_social_influence,
  title={Multi-Agent Social Influence Framework},
  author={Mitali Raj},
  year={2026}
}
```