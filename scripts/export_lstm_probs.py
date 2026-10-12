# LSTM(과제 1) 예측 확률 저장용 스크립트
# '01_lstm_from_scratch.ipynb'의 코드 셀 0~23(데이터 준비 ~ 학습)을 노트북에서 그대로 읽어 실행한 뒤
# val/test 예측 확률, best 모델, vocab, 학습 정보를 저장한다.
# 하이퍼파라미터 비교 실험(셀 40 이후)은 실행하지 않는다.
#
# 실행: 프로젝트 폴더에서  .venv\Scripts\python.exe scripts\export_lstm_probs.py
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 그래프 창을 띄우지 않음

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
NB_PATH = ROOT / "01_lstm_from_scratch.ipynb"
LAST_CELL = 23   # 학습 루프 셀

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"][:LAST_CELL + 1] if c["cell_type"] == "code"]

from IPython.display import display   # 노트북의 display() 대체
ns = {"display": display, "__name__": "__main__"}

for i, src in enumerate(cells):
    if i == LAST_CELL:
        ns["_train_start"] = time.time()
    exec(compile(src, f"<cell {i}>", "exec"), ns)
train_seconds = time.time() - ns["_train_start"]

import numpy as np
import torch
from sklearn.metrics import f1_score

model, criterion, DEVICE = ns["model"], ns["criterion"], ns["DEVICE"]
_, val_prob, val_true = ns["predict_proba"](model, ns["val_loader"], criterion, DEVICE)
_, test_prob, test_true = ns["predict_proba"](model, ns["test_loader"], criterion, DEVICE)

results = ROOT / "results"
results.mkdir(exist_ok=True)
np.save(results / "lstm_val_probs.npy", val_prob.astype(np.float32))
np.save(results / "lstm_test_probs.npy", test_prob.astype(np.float32))
(ROOT / "model").mkdir(exist_ok=True)
torch.save(model.state_dict(), ROOT / "model" / "lstm_kote_best.pt")
(results / "lstm_vocab.json").write_text(json.dumps(ns["vocab"], ensure_ascii=False), encoding="utf-8")

def scores(prob, true, t):
    pred = (prob >= t).astype(int)
    return {"micro": f1_score(true, pred, average="micro", zero_division=0),
            "macro": f1_score(true, pred, average="macro", zero_division=0)}

info = {
    "train_seconds": round(train_seconds, 1),
    "epochs_run": len(ns["train_history"]),
    "best_epoch": int(np.argmax(ns["val_f1_history"])) + 1,
    "best_val_macro_f1": float(ns["best_val_f1"]),
    "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "val_f1_history": [float(x) for x in ns["val_f1_history"]],
    "train_loss_history": [float(x) for x in ns["train_history"]],
    "val_loss_history": [float(x) for x in ns["val_history"]],
    "test_t0.3": scores(test_prob, test_true, 0.3),
    "test_t0.2": scores(test_prob, test_true, 0.2),
    "torch": torch.__version__,
    "device": str(DEVICE),
}
(results / "lstm_train_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in info.items() if "history" not in k}, ensure_ascii=False, indent=2))
