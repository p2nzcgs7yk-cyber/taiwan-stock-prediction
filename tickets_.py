# -*- coding: utf-8 -*-
"""
tickets_clean.py
整理自 Colab 版 tickets.py，並補上報告建議加入的評估：
  1. 各資料集樣本數與類別分布（上漲日比例）
  2. 「永遠預測上漲」基準線（Accuracy、F1）
  3. 混淆矩陣、AUC
  4. Accuracy 與 F1 的 bootstrap 95% 信賴區間
  5. LSTM 以多個亂數種子重複訓練（平均數與標準差）
  6. 簡易回測（含單邊交易成本），與買進持有比較

與原始程式相同的設定：
  - 資料：yfinance，2021-01-01 ~ 2025-12-31，auto_adjust=False（Close 未調整股利）
  - 特徵：Return、MA5_ratio、MA20_ratio、Volatility5、Volume_change
  - 切分：Train 2021-2023、Validation 2024、Test 2025
  - RF：n_estimators=100, random_state=42
  - XGB：n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
  - LSTM：window=20, LSTM(32)+Dropout(0.2)+Dense(1,sigmoid), Adam, 30 epochs, batch 32
  - LSTM 的視窗是在各資料集「內部」建立（與原程式一致），所以 Validation/Test 會各少 20 筆

用法：
  pip install yfinance scikit-learn xgboost tensorflow pandas numpy
  python tickets_clean.py                 # 全部模型
  python tickets_clean.py --models rf xgb # 只跑 RF 與 XGBoost（較快）
  python tickets_clean.py --synthetic     # 不連網，用模擬資料測試程式能否跑通

輸出：終端機報表 + results_all.csv + results_summary.csv
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- 設定
SEED = 42
WINDOW = 20
N_LSTM_RUNS = 5          # LSTM 重複訓練次數（不同亂數種子）
N_BOOT = 2000            # bootstrap 次數
ONE_WAY_COST = 0.002     # 回測的單邊交易成本（約略值，請依實際手續費與證交稅自行調整）
START, END = "2021-01-01", "2026-01-01"   # yfinance 的 end 不含當日
TICKERS = {"2330": "2330.TW", "0050": "0050.TW"}
FEATURES = ["Return", "MA5_ratio", "MA20_ratio", "Volatility5", "Volume_change"]
SPLITS = {
    "train": ("2021-01-01", "2023-12-31"),
    "val": ("2024-01-01", "2024-12-31"),
    "test": ("2025-01-01", "2025-12-31"),
}


# ---------------------------------------------------------------- 資料
def load_prices(ticker):
    import yfinance as yf
    df = yf.download(ticker, start=START, end=END, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def make_synthetic_prices(seed=0):
    """不連網測試用：隨機漫步價格與成交量。"""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(START, "2025-12-31")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(idx))))
    volume = rng.integers(1_000, 5_000, len(idx)).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=idx)


def create_features(df):
    """與報告附錄一相同；另外保留 Next_Return 供回測使用。"""
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Volatility5"] = df["Return"].rolling(5).std()
    df["MA5_ratio"] = df["Close"] / df["MA5"] - 1
    df["MA20_ratio"] = df["Close"] / df["MA20"] - 1
    df["Volume_change"] = df["Volume"].pct_change()
    # 前一日成交量為 0 時會得到 inf：視為缺失值並刪除該列
    df["Volume_change"] = df["Volume_change"].replace([np.inf, -np.inf], np.nan)
    df["Next_Close"] = df["Close"].shift(-1)
    df["Target"] = (df["Next_Close"] > df["Close"]).astype(int)
    df["Next_Return"] = df["Next_Close"] / df["Close"] - 1
    return df.dropna()   # 最後一列沒有 Next_Close，會在這裡被刪除


def split_data(df):
    return {name: df.loc[a:b].copy() for name, (a, b) in SPLITS.items()}


def create_sequences(X, y, window=WINDOW):
    Xs, ys = [], []
    for i in range(window, len(X)):
        Xs.append(X[i - window:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


# ---------------------------------------------------------------- 評估
def baseline_always_up(y):
    """永遠預測上漲：Accuracy = p，F1 = 2p/(1+p)。"""
    p = float(np.mean(y))
    return p, 2 * p / (1 + p)


def bootstrap_ci(y, pred, metric, n_boot=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric(y[idx], pred[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def backtest(pred, next_ret, one_way_cost=ONE_WAY_COST):
    """預測上漲就持有一天，否則空手；持倉改變時扣單邊成本。回傳 (策略報酬, 買進持有報酬)。"""
    pos = np.asarray(pred, dtype=float)
    prev = np.concatenate([[0.0], pos[:-1]])
    trades = np.abs(pos - prev)
    strat = pos * np.asarray(next_ret) - trades * one_way_cost
    return float(np.prod(1 + strat) - 1), float(np.prod(1 + np.asarray(next_ret)) - 1)


def evaluate(ticker, model, run, split, y, pred, prob=None, next_ret=None):
    base_acc, base_f1 = baseline_always_up(y)
    row = {
        "ticker": ticker, "model": model, "run": run, "split": split,
        "n": len(y),
        "up_ratio": round(float(np.mean(y)), 4),
        "pred_up_ratio": round(float(np.mean(pred)), 4),
        "acc": accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "baseline_acc": base_acc, "baseline_f1": base_f1,
    }
    if prob is not None and len(np.unique(y)) == 2:
        row["auc"] = roc_auc_score(y, prob)
    if split == "test":
        lo, hi = bootstrap_ci(y, pred, accuracy_score)
        row["acc_ci_low"], row["acc_ci_high"] = lo, hi
        lo, hi = bootstrap_ci(y, pred, lambda a, b: f1_score(a, b, zero_division=0))
        row["f1_ci_low"], row["f1_ci_high"] = lo, hi
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        row.update({"TN": tn, "FP": fp, "FN": fn, "TP": tp})
        if next_ret is not None:
            strat, bh = backtest(pred, next_ret)
            row["backtest_strategy"], row["backtest_buy_hold"] = strat, bh
    return row


# ---------------------------------------------------------------- 模型
def run_rf(parts):
    m = RandomForestClassifier(n_estimators=100, random_state=SEED)
    m.fit(parts["train"][FEATURES], parts["train"]["Target"])
    out = {}
    for s in ("val", "test"):
        X = parts[s][FEATURES]
        out[s] = (m.predict(X), m.predict_proba(X)[:, 1])
    return out


def run_xgb(parts):
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=SEED)
    m.fit(parts["train"][FEATURES], parts["train"]["Target"])
    out = {}
    for s in ("val", "test"):
        X = parts[s][FEATURES]
        out[s] = (m.predict(X), m.predict_proba(X)[:, 1])
    return out


def run_lstm(parts, seed):
    import tensorflow as tf
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import LSTM, Dropout, Dense

    tf.keras.utils.set_random_seed(seed)
    seqs = {s: create_sequences(parts[s][FEATURES].values, parts[s]["Target"].values)
            for s in ("train", "val", "test")}
    scaler = StandardScaler().fit(seqs["train"][0].reshape(-1, len(FEATURES)))   # 只用 Train fit

    def scale(X):
        return scaler.transform(X.reshape(-1, len(FEATURES))).reshape(X.shape)

    model = Sequential([
        Input(shape=(WINDOW, len(FEATURES))),
        LSTM(32), Dropout(0.2), Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    hist = model.fit(scale(seqs["train"][0]), seqs["train"][1], epochs=30, batch_size=32,
                     validation_data=(scale(seqs["val"][0]), seqs["val"][1]), verbose=0)
    out = {"train_acc_last": hist.history["accuracy"][-1],
           "val_acc_last": hist.history["val_accuracy"][-1]}
    for s in ("val", "test"):
        prob = model.predict(scale(seqs[s][0]), verbose=0).flatten()
        out[s] = ((prob >= 0.5).astype(int), prob, seqs[s][1])
    return out


# ---------------------------------------------------------------- 主程式
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "lstm"],
                    choices=["rf", "xgb", "lstm"])
    ap.add_argument("--synthetic", action="store_true", help="不連網，用模擬資料測試程式")
    args = ap.parse_args()

    rows = []
    for name, ticker in TICKERS.items():
        raw = make_synthetic_prices(int(name)) if args.synthetic else load_prices(ticker)
        data = create_features(raw)
        parts = split_data(data)

        print(f"\n===== {name} =====")
        for s in ("train", "val", "test"):
            y = parts[s]["Target"]
            print(f"{s:5s} n={len(y):4d}  上漲日比例={y.mean():.3f}")
        print("LSTM 視窗樣本數（各少 20 筆）：",
              {s: len(parts[s]) - WINDOW for s in ("train", "val", "test")})

        for model in args.models:
            if model in ("rf", "xgb"):
                out = (run_rf if model == "rf" else run_xgb)(parts)
                for s in ("val", "test"):
                    pred, prob = out[s]
                    rows.append(evaluate(name, model, 0, s, parts[s]["Target"].values, pred, prob,
                                         parts[s]["Next_Return"].values))
            else:
                for run in range(N_LSTM_RUNS):
                    out = run_lstm(parts, SEED + run)
                    for s in ("val", "test"):
                        pred, prob, y = out[s]
                        next_ret = parts[s]["Next_Return"].values[WINDOW:]
                        row = evaluate(name, "lstm", run, s, y, pred, prob, next_ret)
                        if s == "val":
                            row["train_acc_last"] = out["train_acc_last"]
                            row["val_acc_last"] = out["val_acc_last"]
                        rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv("results_all.csv", index=False, encoding="utf-8-sig")

    metric_cols = [c for c in ["acc", "f1", "auc", "baseline_acc", "baseline_f1",
                               "pred_up_ratio", "acc_ci_low", "acc_ci_high",
                               "backtest_strategy", "backtest_buy_hold"] if c in res.columns]
    summary = res.groupby(["ticker", "model", "split"])[["n"] + metric_cols].mean()
    if "lstm" in args.models:
        sd = res[res.model == "lstm"].groupby(["ticker", "model", "split"])[["acc", "f1"]].std()
        sd.columns = ["acc_std_over_runs", "f1_std_over_runs"]
        summary = summary.join(sd)
    summary.to_csv("results_summary.csv", encoding="utf-8-sig")

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n===== 摘要（LSTM 為多次訓練的平均） =====")
    print(summary.round(4))
    print("\n已輸出 results_all.csv 與 results_summary.csv")
    print("提醒：Accuracy 若未高於 baseline_acc、F1 若未高於 baseline_f1，"
          "表示模型沒有贏過「永遠預測上漲」。")


if __name__ == "__main__":
    main()
