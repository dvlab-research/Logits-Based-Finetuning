# Logits-Based Finetuning
<p align="center">
• 🤗 <a href="https://huggingface.co/datasets/JingyaoLi/Science-Logits-1.2M" target="_blank">Data </a> 
• 🤗 <a href="https://huggingface.co/JingyaoLi/ScienceLLaMA-3b" target="_blank">ScienceLLaMA-3B </a> 
• 🤗 <a href="https://huggingface.co/JingyaoLi/ScienceLLaMA-1b" target="_blank">ScienceLLaMA-1B </a> 
• 🐱 <a href="Logits-based Finetuning" target="_blank">Code</a> 
• 📃 Paper (TO be released) <br>
</p>

In recent years, developing compact and efficient large language models (LLMs) has emerged as a thriving area of research. Traditional Supervised Fine-Tuning (SFT), which relies on singular ground truth labels, often fails to capture  token-level dependencies and linguistic diversity. To address these limitations, we propose a logits-based fine-tuning framework that integrates the strengths of supervised learning and knowledge distillation. Our approach constructs enriched training targets by combining teacher logits with ground truth labels, preserving both correctness and linguistic diversity. This ensures more reliable and effective training. We constructed a large-scale 1.2M logits dataset and trained a series of science-focused models. Experimental results demonstrate that our method achieves significant improvements, with accuracy gains of 18% on Mawps and 22.7% on TabMWP. Across nine widely used mathematical benchmarks, our method consistently outperforms prior SFT models, achieving an average improvement of 7.28%. 

## Train
- **Data**: [huggingface](https://huggingface.co/datasets/JingyaoLi/Science-Logits-1.2M)
- **Readme**: [Installation Guide](https://github.com/hiyouga/LLaMA-Factory?tab=readme-ov-file#installation)
- **Installation**:
```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
- **Run**
```bash
# 1b
llamafactory-cli train llamafactory/scripts/llama3.2_1b_instruct_pkl_1300k_e1_warmup0.1_cosinelr1e-6_seed42_maxl2048_a0.9_t1.0_logp5_freqt_0_b1.0_r1.0.yaml
# 3b
llamafactory-cli train llamafactory/scripts/llama3.2_3b_instruct_pkl_1300k_e1_warmup0.1_cosinelr1e-6_seed42_maxl2048_a0.9_t1.0_logp5_freqt_0_b1.0_r1.0.yaml
```

- **Hyperparatemers**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_distill` | `bool` | `False` | Whether to enable distillation. |
| `distill_alpha` | `float` | `0.9` | Balance weight for the distillation loss. |
| `distill_t` | `float` | `1.0` | Temperature for the distillation loss. |
| `distill_gamma` | `float` | `1.0` | Balance weight for teacher model logits. |

## Evaluation

- **Installation**
```bash
cd evaluation/latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
```

- **Run**
```bash
bash evaluation/sh/eval.sh "qwen25-math-cot" $MODEL_NAME_OR_PATH
```