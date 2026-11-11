#!/usr/bin/env python3
"""
ICDE Exp 33 -- check two facts about the released Dev-100 artifacts of Jin et al.
(PVLDB 2026), which our ranking analysis builds on.

(a) For each system, how many released predictions are verbatim copies of the gold
    of the same version (whitespace/case/';' normalized)?
(b) For each system, on how many of the 100 examples do the two prediction files
    (original inputs vs corrected inputs) differ? Jin et al. corrected the question
    in 19 examples and the external knowledge in 17 (their Table 4), so at most 36
    examples had changed inputs; differences beyond 36 must come from regenerating
    predictions, not from changed inputs.
Output: results/prior_claims.json
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")
norm = lambda s: re.sub(r"\s+", " ", s.split("\t")[0]).strip().rstrip(";").lower()
gold = {"dev": json.load(open(os.path.join(D, "dev100", "dev.json"))),
        "dev_old": json.load(open(os.path.join(D, "dev100", "dev_old.json")))}
out = {"copies_of_gold": {}, "differing_predictions": {}, "max_examples_with_changed_inputs": 36}
for f in sorted(os.listdir(os.path.join(D, "agent_preds"))):
    which = "dev_old" if f.endswith("_dev_old.json") else "dev"
    sysname = f[: -len(f"_{which}.json")]
    p = json.load(open(os.path.join(D, "agent_preds", f)))
    out["copies_of_gold"][f] = sum(norm(p[k]) == norm(gold[which][k]) for k in p if k in gold[which])
    if which == "dev":
        q = json.load(open(os.path.join(D, "agent_preds", f"{sysname}_dev_old.json")))
        out["differing_predictions"][sysname] = sum(norm(p[k]) != norm(q[k]) for k in p)
d = [v for k, v in out["differing_predictions"].items() if k != "CHESS"]
out["summary"] = {"systems_excl_CHESS": len(d), "mean_differing": round(sum(d) / len(d), 1),
                  "systems_differing_on_more_than_36": sum(x > 36 for x in d)}
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(HERE, "results", "prior_claims.json"), "w"), indent=1)
