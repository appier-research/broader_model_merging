# A Broader Look at Model Merging: Rethinking Implicit Regularization Induced by Task Arithmetic

This is the official codebase for the paper "[A Broader Look at Model Merging: Rethinking Implicit Regularization Induced by Task Arithmetic](https://arxiv.org/abs/2610.07990)".

## Structure 

| Folder | Contents |
|---|---|
| [`vision_exp/`](vision_exp/README.md) | Model merging on CLIP ViT-B/32, ViT-B/16, ViT-L/14 (9 tasks) |
| [`llm_exp/`](llm_exp/README.md) | Model merging on Qwen3-0.6B/1.7B/4B and Llama-3.2-1B (4 tasks) |
| [`visualization/`](visualization) | Main-result figures (`plot_main_result.py`, `plot_vit32_fig1.py`) |

Each experiment folder has a `README.md` (how to run) and a `design.md`
(code map and protocol details).

## Setup

```bash
conda create -n ta python=3.12.13
conda activate ta
pip install -r requirements.txt
```

## Main result figures

```bash
cd visualization
python plot_main_result.py            # -> figures/ (vision + LLM bars)
python plot_vit32_fig1.py             # ViT-B/32 one-panel version
```

Both read the numbers in `main_result_{vision,llm}.csv`.
