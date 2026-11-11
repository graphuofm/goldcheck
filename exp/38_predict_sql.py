#!/usr/bin/env python3
"""
ICDE Exp 38 -- downstream: generate text-to-SQL predictions for every Dev question with open
models served by vLLM (OpenAI-compatible API), so that we can measure how GoldCheck findings
change the scoring of real predictions (scored locally afterwards, no GPU needed).

Plain zero-shot prompt: schema (CREATE statements + BIRD column descriptions, as in 34),
evidence note, question. Temperature 0. One query per question, no execution feedback.
Usage (on the GPU node, with a vLLM server on $OPENAI_URL serving the model as <tag>):
  python3 38_predict_sql.py --model <tag> --version dev_2025 --ids results/sar_local/ids_dev2025_all.txt \
      --out results/sar_local/pred_<tag>_dev_2025.jsonl --workers 16
Resumes from an existing output file.
"""
import argparse, importlib.util, json, os, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sar", os.path.join(HERE, "34_sar_local.py"))
sar = importlib.util.module_from_spec(spec); spec.loader.exec_module(sar)      # reuse schema() and load_version()
OPENAI_URL = os.environ.get("OPENAI_URL", "http://localhost:8000/v1")
PROMPT = """You are an expert in SQLite. Write one SQLite query that answers the question over the database below.

Database schema:
{schema}

External knowledge: {evidence}

Question: {question}

Return only the SQL query inside a ```sql code block."""


def ask(model, text):
    if os.environ.get("SAR_BACKEND") == "ollama":                  # local runs: native API so num_ctx can be set
        body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}], "stream": False,
                           "options": {"num_ctx": 16384, "temperature": 0}}).encode()
        req = urllib.request.Request(os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
        return d["message"]["content"], {"prompt_tokens": d.get("prompt_eval_count", 0),
                                         "completion_tokens": d.get("eval_count", 0)}
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": text}],
                       "temperature": 0, "max_tokens": 8192}).encode()
    req = urllib.request.Request(f"{OPENAI_URL}/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"].get("content") or ""), d.get("usage", {})


def extract(txt):
    txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)
    m = re.findall(r"```sql\s*(.*?)```", txt, flags=re.S | re.I) or re.findall(r"```\s*(.*?)```", txt, flags=re.S)
    s = (m[-1] if m else txt).strip()
    return s.rstrip(";").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--version", default="dev_2025")
    ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    data = sar.load_version(a.version)
    ids = [int(x) for x in open(a.ids).read().split()]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["qid"] for l in open(a.out) if l.strip() and "error" not in json.loads(l)}
    lock = threading.Lock()

    def one(q):
        e = data[q]; t0 = time.time()
        try:
            txt, use = ask(a.model, PROMPT.format(schema=sar.schema(e["db_id"]), evidence=e.get("evidence", ""),
                                                  question=e["question"]))
            r = {"qid": q, "db": e["db_id"], "sql": extract(txt), "raw": txt[-4000:], "secs": round(time.time() - t0, 1),
                 "tokens_in": use.get("prompt_tokens", 0), "tokens_out": use.get("completion_tokens", 0)}
        except Exception as ex:
            r = {"qid": q, "error": str(ex)[:200]}
        r["model"] = a.model; r["version"] = a.version
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(q, "error" in r, r.get("secs"), flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, [q for q in ids if q not in done]))


if __name__ == "__main__":
    main()
