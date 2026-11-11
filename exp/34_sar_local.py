#!/usr/bin/env python3
"""
ICDE Exp 34 -- re-run the SAR-Agent protocol of Jin et al. (PVLDB 2026) with open
models served locally by ollama, at zero API cost.

Faithful to the released code (text_to_sql_benchmarks/SAR-Agent):
  * the BIRD user prompt is used verbatim (prompts/prompt_user_bird.txt);
  * the loop is the same: the model proposes one verification query, it is executed,
    the result is fed back ("Query running result: ..."), at most 30 steps, until the
    model calls terminate with "Correctness / Is_ambiguous / Explanation / Revised".
Deviations (all forced, all reported in the paper):
  * full_schema.json is not released; we rebuild it per table from the CREATE
    statement plus BIRD's database_description CSV;
  * function calls are emitted as one JSON object in the reply (same protocol for
    every model), because not every open model supports OpenAI function calling;
  * queries run on a READ-ONLY connection with a 20 s timeout; results are cut to
    2,000 characters (the prompt already asks for LIMIT 5).
Usage:
  python3 34_sar_local.py --model qwen2.5:3b --version dev_2024 --ids ids.txt --out results/sar_local/x.jsonl
"""
import argparse, csv, glob, json, os, re, sqlite3, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DBDIR = os.path.join(HERE, "data", "dev_databases")
REPO = os.path.join(HERE, "data", "text_to_sql_benchmarks")
PROMPT = open(os.path.join(REPO, "SAR-Agent", "prompts", "prompt_user_bird.txt")).read()
PROTOCOL = """

**Function calling protocol (this deployment):** In every step reply with exactly ONE JSON object and nothing else, either
{"name": "read_sqlite_query", "arguments": {"query": "<sqlite query>", "explaination": "<why>"}}
or
{"name": "terminate", "arguments": {"analyze_result": "Correctness: ...\\nIs_ambiguous: ...\\nExplanation: ...\\nRevised: ..."}}"""
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
HINT_TEXT = """

**Automated check (GoldCheck) for this gold query:** {hint}
This evidence was obtained by running SQL on the official database. It shows that two readings of the gold query give different results; it does not say which reading the question intends. Take it into account in your judgment."""
HINTS = {}           # qid -> evidence text; set with --hints or the SAR_HINTS environment variable


def load_version(v):
    if v.startswith("dev_"):
        p = os.path.join(HERE, "data_ext", "bird_dev_20240627" if v == "dev_2024" else "bird_dev_20251106", "dev.json")
        raw = open(p).read().strip()
        rows = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines()]
        return {e["question_id"]: e for e in rows}
    if v == "mini_T1":
        return {e["question_id"]: e for e in json.load(open(os.path.join(HERE, "data", "arcwise_orig.json")))}
    raise ValueError(v)


_schema = {}
def schema(db):
    if db in _schema:
        return _schema[db]
    con = sqlite3.connect(f"file:{os.path.join(DBDIR, db + '.sqlite')}?mode=ro", uri=True)
    parts = []
    for name, sql in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"):
        desc = ""
        for f in glob.glob(os.path.join(REPO, "data", "schemas", db, "database_description", "*.csv")):
            if os.path.splitext(os.path.basename(f))[0].lower() == name.lower():
                try:
                    rows = list(csv.reader(open(f, encoding="utf-8", errors="replace")))
                except Exception:
                    rows = []
                desc = " | ".join("; ".join(c.strip() for c in r if c and c.strip()) for r in rows[1:])
        parts.append(f"{name}: {' '.join(sql.split())}" + (f" -- column descriptions: {desc}" if desc else ""))
    con.close()
    _schema[db] = "\n".join(parts)
    return _schema[db]


def run_query(db, sql, secs=20):
    con = sqlite3.connect(f"file:{os.path.join(DBDIR, db + '.sqlite')}?mode=ro", uri=True)
    con.create_function("IIF", 3, lambda c, x, y: x if c else y)
    t = threading.Timer(secs, con.interrupt); t.start()
    try:
        res = str(con.execute(sql).fetchmany(50))
    except Exception as e:
        res = f"Error: {e}"
    finally:
        t.cancel(); con.close()
    return res[:2000]


BACKEND = os.environ.get("SAR_BACKEND", "ollama")          # "ollama" or "openai" (vLLM server)
OPENAI_URL = os.environ.get("OPENAI_URL", "http://localhost:8000/v1")


def chat(model, messages, num_ctx):
    if BACKEND == "openai":
        body = json.dumps({"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 8192}).encode()
        req = urllib.request.Request(f"{OPENAI_URL}/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            d = json.loads(r.read())
        m = d["choices"][0]["message"]
        txt = (m.get("content") or "")
        return {"message": {"content": txt}, "prompt_eval_count": d.get("usage", {}).get("prompt_tokens", 0),
                "eval_count": d.get("usage", {}).get("completion_tokens", 0)}
    body = json.dumps({"model": model, "messages": messages, "stream": False,
                       "options": {"num_ctx": num_ctx, "temperature": 0.7}}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def parse_call(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)       # reasoning models
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    s = m.group(0)
    for cand in (s, s.replace("\n", "\\n")):
        try:
            return json.loads(cand)
        except Exception:
            pass
    nm = re.search(r'"name"\s*:\s*"(\w+)"', s)
    if nm and nm.group(1) == "terminate":
        a = re.search(r'"analyze_result"\s*:\s*"(.*)"\s*\}\s*\}?\s*$', s, flags=re.S)
        return {"name": "terminate", "arguments": {"analyze_result": a.group(1) if a else s}}
    q = re.search(r'"query"\s*:\s*"(.*?)"\s*,\s*"explaination"', s, flags=re.S)
    if nm and q:
        return {"name": nm.group(1), "arguments": {"query": q.group(1).replace('\\"', '"'), "explaination": ""}}
    return None


def verdict(analysis):
    a = analysis or ""
    c = re.search(r"Correctness\W*\s*(yes|no)", a, flags=re.I)
    m = re.search(r"Is_ambiguous\W*\s*(yes|no)", a, flags=re.I)
    return (c.group(1).lower() if c else None), (m.group(1).lower() if m else None)


def review(model, e, num_ctx=16384, max_steps=30):
    prompt = PROMPT.format(question=e["question"], schema=schema(e["db_id"]), external_knowledge=e.get("evidence", ""),
                           gold_query=e["SQL"]) + PROTOCOL
    if HINTS.get(str(e["question_id"])):
        prompt += HINT_TEXT.format(hint=HINTS[str(e["question_id"])])
    msgs = [{"role": "system", "content": prompt}]
    user = "Please judge the correctness and ambiguity of the text-to-sql pair."
    t0 = time.time(); tok_in = tok_out = 0; analysis = None; steps = 0; bad = 0
    for steps in range(1, max_steps + 1):
        msgs.append({"role": "user", "content": user})
        r = chat(model, msgs, num_ctx)
        txt = r["message"]["content"]; msgs.append({"role": "assistant", "content": txt})
        tok_in += r.get("prompt_eval_count", 0); tok_out += r.get("eval_count", 0)
        call = parse_call(txt)
        if call is None:
            bad += 1
            if re.search(r"Correctness\W*\s*(yes|no)", txt, flags=re.I):
                analysis = txt; break
            user = "Your reply was not a single JSON function call. Reply with exactly one JSON object as specified."
            continue
        if call.get("name") == "terminate":
            analysis = call.get("arguments", {}).get("analyze_result", ""); break
        q = call.get("arguments", {}).get("query", "")
        user = "Query running result: " + run_query(e["db_id"], q)
    c, a = verdict(analysis)
    return {"qid": e["question_id"], "db": e["db_id"], "model": model, "correct": c, "ambiguous": a, "steps": steps,
            "bad_format": bad, "terminated": analysis is not None, "secs": round(time.time() - t0, 1),
            "tokens_in": tok_in, "tokens_out": tok_out, "analysis": analysis}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--version", default="dev_2024")
    ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--num_ctx", type=int, default=16384)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--hints", default=os.environ.get("SAR_HINTS", ""))
    a = ap.parse_args()
    if a.hints:
        HINTS.update(json.load(open(a.hints)))
    data = load_version(a.version)
    ids = [int(x) for x in open(a.ids).read().split()]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["qid"] for l in open(a.out) if l.strip()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    todo = [q for q in ids if q not in done]
    lock = threading.Lock()

    def one(q):
        try:
            r = review(a.model, data[q], a.num_ctx)
        except Exception as ex:
            r = {"qid": q, "model": a.model, "error": str(ex)[:200]}
        r["version"] = a.version; r["tag"] = a.tag; r["hinted"] = str(q) in HINTS
        with lock:
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(q, r.get("correct"), r.get("ambiguous"), r.get("steps"), r.get("secs"), flush=True)

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
