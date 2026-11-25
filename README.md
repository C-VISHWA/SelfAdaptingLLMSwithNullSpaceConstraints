Mitigating Catastrophic Forgetting in Self-Adapting LLMs

Repository description (GitHub "About" box): Measuring and reducing catastrophic forgetting in self-adapting language models (SEAL) with null-space constrained editing (NSCE), replay and HiPPO baselines.

Topics: llm continual-learning catastrophic-forgetting lora seal null-space qwen2-5 pytorch nlp

Overview

Self-adapting language models such as SEAL improve by writing their own training data ("self-edits") and fine-tuning on it. Repeating this merges many LoRA updates into the weights, and knowledge learned earlier can be overwritten. This project builds a way to measure that forgetting and compares ways to reduce it.

We built:

A retention harness. After every update it evaluates all articles seen so far and fills a lower-triangular accuracy matrix (row 0 is the untouched model). From it we compute plasticity, final accuracy, retention ratio and collision fraction.
Three interchangeable update methods under identical settings:
Plain SEAL: merge each LoRA update as is.
Replay: mix two earlier training sequences into each update.
NSCE (Null-Space Constrained Editing): project the LoRA A matrix away from the directions earlier knowledge uses, A' = A − (A·U)·Uᵀ, where U holds the top eigenvectors (90% energy, rank budget 64) of an uncentred activation covariance.
A preliminary HiPPO (LegS/LegT) memory module in SEAL's selection step (single run on Qwen2.5-3B; not a headline result).
Experiments

Four controlled setups, each with 3 seeds × {plain, replay, NSCE} = 36 runs:

Setup	Model	SQuAD articles
small_15	Qwen2.5-0.5B	15
small_50	Qwen2.5-0.5B	50
big_15	Qwen2.5-7B	15
big_50	Qwen2.5-7B	50

Settings: LoRA r=16, α=32 on q/v; AdamW 5e-5; 2 epochs; batch size 1; max length 256; bf16. Self-edits come from qwen2.5:1.5b through Ollama (temperature 1.0), 8 questions per article.

Results (mean ± SEM over 3 seeds, %)
Setup	Method	Plasticity	Final acc.	Retention
small_15	Plain	6.4 ± 2.0	3.9 ± 1.2	57.1 ± 5.5
	Replay	7.2 ± 1.1	3.6 ± 0.3	52.4 ± 9.9
	NSCE	8.6 ± 2.4	5.8 ± 0.5	60.0 ± 12.6
small_50	Plain	6.5 ± 0.5	3.3 ± 1.0	36.7 ± 12.1
	Replay	7.1 ± 1.1	3.2 ± 1.4	38.1 ± 20.7
	NSCE	9.8 ± 1.3	6.1 ± 1.5	55.6 ± 8.3
big_15	Plain	28.3 ± 1.4	21.9 ± 1.0	81.5 ± 6.1
	Replay	27.5 ± 2.4	23.3 ± 1.4	78.8 ± 5.8
	NSCE	26.7 ± 0.8	20.0 ± 2.1	74.0 ± 6.8
big_50	Plain	26.1 ± 1.1	19.0 ± 1.3	81.7 ± 7.7
	Replay	27.4 ± 1.1	18.7 ± 0.3	68.9 ± 4.9
	NSCE	27.7 ± 1.5	20.8 ± 1.0	80.1 ± 1.2

Untouched-model accuracy (row 0): 10.6, 9.7, 22.5, 23.7.

Takeaways
Forgetting is real and grows with stream length on the 0.5B model: plain-merge retention falls from 57% to 37%. The 7B model retains about 82%.
NSCE gives the best final accuracy on 0.5B, though most gaps are within one SEM. It shows no clear benefit on 7B.
The collision fraction, 64/d (7.1% for 0.5B, 1.8% for 7B), is chance-level overlap. This helps explain why larger models need less protection.
No method ends meaningfully above the untouched model. NSCE reduces degradation; it does not produce net learning.
Limitations

Small effect sizes with 3 seeds; no raw-passage baseline; HiPPO has no matched-scale result yet (NSCE + HiPPO integration is under testing); an earlier laptop run on a 16 GB Mac (MPS) gave different numbers and is treated as preliminary.
