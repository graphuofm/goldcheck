# Data sources

Place the data under `exp/` as follows. None of it is redistributed in this repository.

| Target path | Source | License |
|---|---|---|
| `exp/data/dev_databases/*.sqlite` | BIRD Dev databases, https://bird-bench.github.io | CC BY-SA 4.0 |
| `exp/data_ext/bird_dev_20240627/dev.json` | BIRD Dev gold, 2024-06-27 release (same site) | CC BY-SA 4.0 |
| `exp/data_ext/bird_dev_20251106/dev.json` | BIRD Dev gold, 2025-11-06 release, https://huggingface.co/datasets/birdsql/bird_sql_dev_20251106 | CC BY-SA 4.0 |
| `exp/data/minidev.zip`, `exp/data/arcwise_orig.json` | BIRD Mini-Dev (T0) and the Arcwise correction (T1) | CC BY-SA 4.0 |
| `exp/data/text_to_sql_benchmarks/` | repository of Jin et al. (PVLDB 2026): Mini-Dev audit (T2), SAR-Agent prompt, schemas | see that repository |
| `exp/data/dev100/`, `exp/data/agent_preds/` | Dev-100 gold and system predictions released by Jin et al. | see that repository |
| `exp/data/bird_train/` (`train.json`, `train_databases/`) | https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip | CC BY-SA 4.0 |

Software: Python 3.8+, `sqlglot`, `openpyxl`, `matplotlib`; `duckdb==1.1.3` for `40_tie_engines.py`
(its SQLite extension is installed into `exp/data/duckdb_ext/` on first use with
`INSTALL sqlite`). LLM runs used vLLM 0.11 on one NVIDIA H100 and Ollama on one Quadro RTX 6000.
