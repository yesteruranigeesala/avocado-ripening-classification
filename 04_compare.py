import glob, json, pandas as pd
rows = []
for f in sorted(glob.glob("results/*_metrics.json")):
    m = json.load(open(f))
    rows.append({k: round(m[k], 4) if isinstance(m[k], float) else m[k]
                 for k in ("model", "accuracy", "precision_macro", "recall_macro", "f1_macro")}
                | {"cpu_ms": round(m.get("latency_ms_cpu", float("nan")), 1)})
df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
print(df.to_string(index=False)); df.to_csv("results/comparison.csv", index=False)
open("results/comparison.md", "w").write(df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string())
print("\nBest by macro-F1:", df.iloc[0]["model"])
