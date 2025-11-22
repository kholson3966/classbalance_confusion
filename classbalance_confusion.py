import pandas as pd
import numpy as np

df = pd.read_csv("analysis_out/sessions_posterior.csv")
lab = df["label"].astype(str).str.lower()
human_vals = {"human","1","true","t","h"}
known = lab.isin(human_vals | {"scripted","0","false","f","s"})
dfk = df[known].copy()
dfk["y_true"] = lab.isin(human_vals)
dfk["y_pred"] = dfk["p_human"] >= 0.5

# Majority-class baseline
maj = dfk["y_true"].mean()
baseline_acc = max(maj, 1-maj)

acc = (dfk["y_pred"] == dfk["y_true"]).mean()
tp = ((dfk["y_pred"]==1) & (dfk["y_true"]==1)).sum()
tn = ((dfk["y_pred"]==0) & (dfk["y_true"]==0)).sum()
fp = ((dfk["y_pred"]==1) & (dfk["y_true"]==0)).sum()
fn = ((dfk["y_pred"]==0) & (dfk["y_true"]==1)).sum()

ths = np.linspace(0,1,101)
best = max(
    (( (dfk["p_human"]>=t).eq(dfk["y_true"]).mean(), t) for t in ths),
    key=lambda x: x[0]
)

print(f"Accuracy: {acc:.3f} vs baseline {baseline_acc:.3f}")
print(f"Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"Precision(H): {tp/max(tp+fp,1):.3f}  Recall(H): {tp/max(tp+fn,1):.3f}")
print("Best accuracy threshold:", best)
