# SemEval 2026 Task 11 - Disentangling Content and Formal Reasoning
## Student Research Project - Complete Guide

> **Research Goal:** Build models that can assess formal validity of syllogistic arguments independent of their plausibility  
> **Competition:** [SemEval 2026 Task 11](https://sites.google.com/view/semeval-2026-task-11)  
> **Evaluation Period:** January 2026  
> **Paper Deadline:** February 2026  
> **Target:** High accuracy with low content effect bias

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Training Data Overview

**Official Dataset:** Provided by SemEval 2026 Task 11 organizers

| Subtask | Description | Languages | Key Challenge |
|---------|-------------|-----------|---------------|
| **Subtask 1** | Syllogistic Reasoning | English | Validity prediction independent of plausibility |
| **Subtask 2** | + Irrelevant Premises | English | Identify relevant premises + validity |
| **Subtask 3** | Multilingual Reasoning | 12 languages | Cross-lingual validity prediction |
| **Subtask 4** | Multilingual + Irrelevant | 12 languages | Multilingual premise selection + validity |

**Languages:** English, German, Spanish, French, Italian, Dutch, Portuguese, Russian, Chinese, Swahili, Bengali, Telugu

**Data Format:**
- **Format:** JSON files
- **Fields:** `id`, `syllogism`, `validity` (bool), `plausibility` (bool, training only)
- **Training:** English only (low-resource setting)
- **Test:** Multilingual evaluation

**Example:**
```json
{
  "id": "0",
  "syllogism": "Not all canines are aquatic creatures known as fish. It is certain that no fish belong to the class of mammals. Therefore, every canine falls under the category of mammals.",
  "validity": false,
  "plausibility": true
}
```

**Sample Data Generation:**
```bash
python3 src/generate_data.py --subtask 1  # Generate sample data
```

---

## 📑 Quick Navigation

**🆕 New here?** → [Quick Start](#-quick-start-5-minutes) (5 minutes)  
**👥 Working in a team?** → [Team Guide](#-contributing-for-teams) (avoid merge conflicts!)  
**🎯 Ready to contribute?** → [Pick Your Level](#-how-to-contribute) (⭐, ⭐⭐, or ⭐⭐⭐)  
**📚 Need to learn?** → [Learning Resources](#-learning-resources) (tutorials by level)  
**❓ Need help?** → [Available Commands](#-available-commands) (CLI reference)

---

## 🎯 The Challenge

**Main Problem:** Language models exhibit **content effect** - they confuse formal validity with plausibility.

**Example:**
```
Syllogism: "All dogs are mammals. All mammals breathe. Therefore, all dogs breathe."
- Validity: TRUE (logically valid)
- Plausibility: TRUE (matches world knowledge)
- Model prediction: ✅ Correct (easy case)

Syllogism: "All cats are reptiles. All reptiles are cold-blooded. Therefore, all cats are cold-blooded."
- Validity: TRUE (logically valid structure)
- Plausibility: FALSE (contradicts world knowledge)
- Model prediction: ❌ Often wrong! (content effect bias)
```

**Our Goal:** Build models that predict validity **independent** of plausibility.

### Subtasks

**Subtask 1: English Syllogistic Reasoning**
- Predict validity (True/False)
- Minimize content effect bias
- Metric: Accuracy / Content Effect ratio

**Subtask 2: English + Irrelevant Premises**
- Predict validity
- Identify relevant premises
- Metric: (Accuracy + F1) / Content Effect ratio

**Subtask 3: Multilingual Reasoning**
- Same as Subtask 1, but 12 languages
- Zero-shot transfer from English training

**Subtask 4: Multilingual + Irrelevant Premises**
- Same as Subtask 2, but 12 languages
- Most challenging task

---

## 🚀 Quick Start (5 minutes)

### 1. Setup (Reproducible Environment)

**Recommended: Use our setup script**
```bash
# Clone the repository
git clone <your-repo-url>
cd semeval_task11_complete

# Run automated setup
./setup_env.sh

# Activate environment
source venv/bin/activate

# Verify everything is correct
python3 verify_environment.py
```

### 2. Train Your First Model

```bash
# Train baseline model on Subtask 1
python3 src/pipeline.py train --subtask 1

# Expected output: ~60-70% accuracy with content effect

# Check dataset info
python3 src/pipeline.py info --subtask 1
```

### 3. Explore Data

```bash
# Open Jupyter notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

### 4. Run Tests

```bash
./run_all_tests.sh
# All tests should pass ✅
```

---

## 📁 Project Structure

```
semeval_task11_complete/
├── README.md                 # This file
├── RESEARCH.md               # Research direction & hypotheses
├── CONTRIBUTING.md           # Contribution workflow
├── requirements.txt          # Pinned dependencies
├── setup_env.sh              # Environment setup script
├── verify_environment.py     # Environment verification
│
├── src/                      # Core code
│   ├── pipeline.py          # Main CLI
│   ├── features/            # ⭐⭐ Add YOUR feature file here (no conflicts!)
│   │   ├── basic.py         # Basic linguistic features
│   │   ├── logical.py       # Logical structure features
│   │   ├── _template.py     # Copy this to start
│   │   └── yourname.py      # ← Create your own!
│   ├── models/              # ⭐⭐⭐ Add YOUR model file here (no conflicts!)
│   │   ├── baseline.py      # Baseline models
│   │   ├── _template.py     # Copy this to start
│   │   └── yourname.py      # ← Create your own!
│   ├── data_loader.py       # Load syllogism data
│   ├── evaluate.py          # Evaluation metrics
│   └── generate_data.py     # Sample data generation
│
├── notebooks/                # Jupyter notebooks
│   ├── 01_data_exploration.ipynb    # ⭐ Start here
│   └── 02_baseline_training.ipynb   # ⭐⭐ Train models
│
├── experiments/              # ⭐⭐⭐ Large research projects
├── tests/                    # Unit tests
├── data/                     # Datasets
└── models/                   # Saved models
```

**Your contributions fit here:**
- **⭐ Beginner** → `notebooks/` (explore data, visualize)
- **⭐⭐ Intermediate** → `src/features/yourname.py` (add features - **no merge conflicts!**)
- **⭐⭐⭐ Advanced** → `src/models/yourname.py`, `experiments/` (transformers, novel methods)

---

## 👥 Contributing (For Teams)

**Working in a team of 10 students?** Here's how to organize:

### Role Distribution
- **Data/Features (3 students)** → `notebooks/`, `src/features/alice.py`, `src/features/bob.py`
- **Modeling (3 students)** → `src/models/dave.py`, `experiments/transformer_models/`
- **Multilingual (2 students)** → Cross-lingual transfer, translation augmentation
- **Research (2 students)** → Documentation, paper writing

**Key:** Each student gets their own file/folder → **Zero merge conflicts!**

**How to add features/models:**
```bash
# Features: Copy template
cp src/features/_template.py src/features/yourname.py
# Edit, then register in src/features/__init__.py (2 lines)

# Models: Copy template
cp src/models/_template.py src/models/yourname.py
# Edit, then register in src/models/__init__.py (2 lines)
```

---

## 👤 How to Contribute

**Everyone can contribute!** We have tasks for all skill levels.

### ⭐ Beginner Tasks (No ML experience needed)

#### 1. Data Exploration (2-5 hours)
- **Task:** Analyze syllogism patterns in training data
- **What to do:**
  - Count valid vs invalid arguments
  - Analyze plausible vs implausible distribution
  - Visualize content effect in training data
- **Skills learned:** Data analysis, visualization
- **File:** `notebooks/yourname_exploration.ipynb`

#### 2. Documentation (1-2 hours)
- **Task:** Improve README or add examples
- **What to do:**
  - Add more syllogism examples
  - Document evaluation metrics clearly
  - Create beginner-friendly guides
- **Skills learned:** Technical writing

### ⭐⭐ Intermediate Tasks (Some ML/NLP experience)

#### 1. Linguistic Features (5-10 hours)
- **Task:** Extract linguistic features from syllogisms
- **What to do:**
  - POS tagging, dependency parsing
  - Negation detection
  - Quantifier extraction ("all", "some", "no")
- **Skills learned:** NLP, feature engineering
- **Expected impact:** +5-10% accuracy

#### 2. Logical Structure Features (5-10 hours)
- **Task:** Extract formal logic patterns
- **What to do:**
  - Identify premise-conclusion structure
  - Extract logical connectives
  - Detect syllogistic figures (AAA-1, EAE-2, etc.)
- **Skills learned:** Logic, pattern matching
- **Expected impact:** +10-15% accuracy

#### 3. Multilingual Evaluation (5-8 hours)
- **Task:** Test models across languages
- **What to do:**
  - Evaluate on different language subsets
  - Analyze cross-lingual performance
  - Identify language-specific biases
- **Skills learned:** Multilingual NLP

### ⭐⭐⭐ Advanced Tasks (Research/Deep Learning)

#### 1. Transformer Fine-tuning (10-20 hours)
- **Task:** Fine-tune multilingual transformers
- **What to do:**
  - Fine-tune mBERT, XLM-R on syllogisms
  - Experiment with prompt engineering
  - Implement chain-of-thought reasoning
- **Skills learned:** Transformers, prompt engineering
- **Expected impact:** +20-30% accuracy

#### 2. Debiasing Methods (15-25 hours)
- **Task:** Reduce content effect bias
- **What to do:**
  - Implement contrastive learning
  - Try adversarial training
  - Explore causal intervention methods
- **Skills learned:** Advanced ML, bias mitigation
- **Expected impact:** -20-40% content effect

#### 3. Premise Selection (10-15 hours)
- **Task:** Build models for Subtask 2/4
- **What to do:**
  - Implement attention-based selection
  - Try graph neural networks
  - Multi-task learning (validity + selection)
- **Skills learned:** Structured prediction

---

## 🔬 Research & Advanced Topics

**For detailed research direction, hypotheses, and experimental plans:**
👉 **See [RESEARCH.md](RESEARCH.md)**

**Key research questions:**
- How to disentangle content from formal reasoning?
- Can we achieve zero-shot multilingual transfer?
- What features best predict validity independent of plausibility?

---

## 🤝 How to Contribute

**For detailed contribution workflow and git guidelines:**
👉 **See [CONTRIBUTING.md](CONTRIBUTING.md)**

**Quick workflow:**
1. Pick a task from GitHub Issues
2. Create branch: `git checkout -b feature/your-task`
3. Make changes and test
4. Push and open Pull Request
5. Git tracks your contributions automatically!

---

## 🔧 Available Commands

### Training

```bash
# Train on Subtask 1 (English)
python3 src/pipeline.py train --subtask 1

# Train on Subtask 2 (English + irrelevant premises)
python3 src/pipeline.py train --subtask 2

# Train specific model
python3 src/pipeline.py train --subtask 1 --model-type transformer
```

### Evaluation

```bash
# Evaluate on validation set
python3 src/pipeline.py evaluate --subtask 1 --model-path models/subtask1_model.pkl

# Calculate content effect
python3 src/evaluate.py --predictions results/predictions.json --gold data/val.json
```

### Data Info

```bash
# Show dataset statistics
python3 src/pipeline.py info --subtask 1
```

### Testing

```bash
# Run all tests
./run_all_tests.sh

# Run specific test
python3 tests/test_features.py
```

---

## 🏆 Competition Details

### Evaluation Metrics

**Subtask 1 & 3:**
- **Accuracy:** Correct validity predictions
- **Content Effect:** Bias towards plausibility
- **Ranking:** Accuracy / Content Effect ratio (higher is better)

**Subtask 2 & 4:**
- **Accuracy:** Correct validity predictions
- **F1 Score:** Premise selection quality
- **Content Effect:** Bias towards plausibility
- **Ranking:** (Accuracy + F1) / 2 / Content Effect ratio

### Timeline

| Date | Event |
|------|-------|
| **Now - Dec 2025** | Training phase |
| **Dec 2025** | Test data released |
| **Jan 2026** | Evaluation period |
| **Feb 2026** | Paper submission |
| **Summer 2026** | SemEval workshop |

---

## 👥 Team & Authorship

**Authorship will be determined during paper writing based on:**
- Quality and impact of contributions
- Sustained engagement throughout the project
- Collaboration with team members
- Git commit history as supporting evidence

**Typical tiers:**
- **🥇 Lead/Co-Authors:** Major features, sustained engagement, significant impact
- **🥈 Contributing Authors:** Meaningful contributions, consistent involvement
- **🥉 Acknowledged:** Helpful contributions, bug fixes, documentation

---

## 📚 Learning Resources

### Logical Reasoning
- **Syllogistic Logic:** [Stanford Encyclopedia](https://plato.stanford.edu/entries/logic-classical/)
- **Formal Logic Basics:** [Khan Academy](https://www.khanacademy.org/math/logic)

### NLP & Transformers
- **Transformers:** [HuggingFace Course](https://huggingface.co/course)
- **Multilingual NLP:** [mBERT Paper](https://arxiv.org/abs/1810.04805)
- **XLM-RoBERTa:** [Paper](https://arxiv.org/abs/1911.02116)

### Bias & Debiasing
- **Content Effect:** [Task 11 References](https://sites.google.com/view/semeval-2026-task-11)
- **Debiasing Methods:** [Survey Paper](https://arxiv.org/abs/2103.00453)

### Tools
- **spaCy:** [NLP Library](https://spacy.io/)
- **Transformers:** [HuggingFace](https://huggingface.co/transformers/)
- **PyTorch:** [Deep Learning](https://pytorch.org/)

---

## 💬 Getting Help

- **Questions?** Open an issue with `question` label
- **Found a bug?** Open an issue with `bug` label
- **Have an idea?** Open an issue with `enhancement` label
- **Task Slack:** [Join here](https://sites.google.com/view/semeval-2026-task-11)

---

## 🎓 Learning Outcomes

By contributing to this project, you will:
- ✅ Understand formal logic and reasoning
- ✅ Learn about content bias in language models
- ✅ Work with multilingual NLP
- ✅ Gain experience with transformers
- ✅ Contribute to a published research paper
- ✅ Build your research portfolio

---

## 🚀 Ready to Start?

1. **Read this README** (you're doing it! ✅)
2. **Pick your difficulty level** (⭐, ⭐⭐, or ⭐⭐⭐)
3. **Follow Quick Start** (5 minutes)
4. **Choose a task** (check Issues)
5. **Make your contribution** (follow workflow)
6. **Commit and push** (Git tracks everything automatically!)

**Questions?** Run `python3 src/pipeline.py --help` or open an issue.

---

## 🔒 Reproducibility

### Environment Reproducibility

**We ensure "works on my machine" NEVER happens:**

1. **Pinned Dependencies** - All package versions exact in `requirements.txt`
2. **Python Version Control** - `.python-version` specifies Python 3.9.18
3. **Virtual Environment** - Isolated from system Python
4. **Environment Verification** - `verify_environment.py` checks everything

### Experimental Reproducibility

All experiments are **fully reproducible** with seed `42`:
- ✅ Data generation produces identical datasets
- ✅ Model training is deterministic
- ✅ Results are consistent across runs

**To reproduce results:**
```bash
# 1. Setup environment (once)
./setup_env.sh
source venv/bin/activate

# 2. Verify environment
python3 verify_environment.py

# 3. Run experiments (same results every time)
python3 src/generate_data.py --subtask 1
python3 src/pipeline.py train --subtask 1
```

**Guaranteed:** Same environment + same seed = same results!

---

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

---

**Let's disentangle content from reasoning together!** 🎯
