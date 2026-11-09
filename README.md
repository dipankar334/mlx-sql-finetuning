# Domain-Specific Text-to-SQL Fine-Tuning with Apple MLX (LoRA)

Fine-tuned **Qwen2.5-Coder-3B-Instruct (4-bit)** natively on Apple Silicon using **LoRA (Low-Rank Adaptation)** and Apple's **MLX** framework to master proprietary warehouse schemas and domain-specific financial KPIs without prompt-level schema injection.

## Benchmark Results (Unseen Test Set)

| Model Variant | Schema in Prompt? | Exact-Match Accuracy |
| :--- | :--- | :--- |
| Base `Qwen2.5-Coder-3B-Instruct-4bit` | No | **0.0% (0/8)** |
| Fine-Tuned `Qwen2.5-Coder-3B + LoRA` | No | **100.0% (8/8)** |

## Key Engineering Highlights
* **Parameter-Efficient Fine-Tuning (LoRA):** Trained 8 transformer layers over 200 iterations on 16 GB Unified Memory (~4.5 GB peak footprint), producing a lightweight `adapters.safetensors` checkpoint.
* **Proprietary Business Logic Encoding:** Taught the model internal definitions (`Net ARR`, `Churn Risk`, `APAC Tier-1`) and strict raw-SQL formatting rules (`LIMIT 100` on non-aggregated queries).
* **Weight Fusion:** Merged trained LoRA matrices directly into the quantized base weights (`mlx_lm.fuse`) for zero-overhead standalone inference.

## Reproduce Locally
1. Create environment: `python3 -m venv .venv && source .venv/bin/activate && pip install mlx-lm`
2. Generate dataset: `python3 build_dataset.py`
3. Evaluate baseline: `python3 evaluate.py`
4. Fine-tune LoRA: `python3 -m mlx_lm.lora --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit --train --data ./data --iters 200 --batch-size 2 --num-layers 8 --adapter-path ./adapters`
5. Evaluate fine-tuned model: `python3 evaluate.py mlx-community/Qwen2.5-Coder-3B-Instruct-4bit ./adapters`
