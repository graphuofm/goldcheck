# GoldCheck

A deterministic checker for **state-dependent gold queries** in text-to-SQL benchmarks,
and all artifacts of the paper

> *Right by Accident: Text-to-SQL Gold Queries Whose Answers Depend on the Database State*
> (submitted to ICDE 2027, Experiment, Analysis, and Benchmark track).

A gold query is state-dependent when its answer on the benchmark database depends on the
data in a way the question does not intend: it cuts a ranked list at a tie, counts join rows
instead of entities, puts a `NULL` at the top of a ranking, or returns nothing.
GoldCheck reports a finding only with a proof obtained by running SQL.

## Quick start

```bash
pip install sqlglot
cd exp
python3 -c "import goldcheck, json; print(json.dumps(goldcheck.check('path/to/db.sqlite', 'SELECT ...'), indent=1))"
```

`goldcheck.check(db_path, sql)` opens the database read-only and returns the flags
(`topk_tie`, `topk_null`, `count_fanout`, `empty_answer`, `topk_latent`) with evidence.

## Layout

| Path | Content |
|---|---|
| `exp/goldcheck.py` | the checker (about 240 lines, needs `sqlglot` and Python's `sqlite3`) |
| `exp/30`–`42_*.py` | experiments of the paper (see table below) |
| `exp/01`–`29_*.py` | earlier analyses; `07`, `22`, `25`, `26`, `29`, `33` feed Sections VI–VII and the figures |
| `exp/results/` | every result file behind the numbers, tables, and figures |
| `exp/results/sar_local/` | SAR-Agent re-runs and model predictions: raw outputs, run metadata, server logs |
| `review/` | annotation sheets, both annotators' returned labels, and the agreement script |
| `paper/` | LaTeX source, figures, and the submitted PDF |

| Script | Paper |
|---|---|
| `32_run_goldcheck.py` | findings per release, agreement with reviews, survival |
| `35_baselines.py`, `42_uncertainty.py` | baselines, bootstrap intervals, held-out test, kappa interval |
| `34_sar_local.py`, `36_sar_analysis.py`, `37_make_itiger_inputs.py` | SAR-Agent on open models, with and without GoldCheck evidence |
| `38_predict_sql.py`, `39_score_predictions.py` | case study on model predictions |
| `40_tie_engines.py` | engine dependence of tie answers |
| `41_goldcheck_train.py` | case study on the BIRD training split |
| `29_figures.py` | figures |

## Data

The benchmark data are third-party and are not redistributed here. See `DATA.md` for
sources and where to place them. All result files in `exp/results/` were produced from
these sources and let you check every number without re-running.

## License

Code: MIT (see `LICENSE`). Result files derived from BIRD inherit CC BY-SA 4.0.
