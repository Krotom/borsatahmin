import yfinance as yf
import pandas as pd
import numpy as np
import ta
import feedparser
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
from sklearn.metrics import classification_report

# -------------------------
# 1. Haber Sentiment Analizi
# -------------------------
sentiment_model = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

def get_news_sentiment(query="THYAO"):
    feed_url = f"https://news.google.com/rss/search?q={query}+site:kap.org.tr&hl=tr&gl=TR&ceid=TR:tr"
    feed = feedparser.parse(feed_url)
    if not feed.entries:
        return 0.5  # veri yoksa nötr
    scores = []
    for entry in feed.entries:
        result = sentiment_model(entry.title)[0]
        label = result["label"]
        score = result["score"]
        if label.lower() == "positive":
            scores.append(score)
        elif label.lower() == "negative":
            scores.append(1 - score)
        else:
            scores.append(0.5)
    return np.mean(scores)

# -------------------------
# 2. Fiyat Verisi ve Teknikler
# -------------------------
ticker = "THYAO.IS"
data = yf.download(ticker, period="2y", interval="1d")

data["rsi"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
data["ema20"] = ta.trend.EMAIndicator(data["Close"], window=20).ema_indicator()
data["ema50"] = ta.trend.EMAIndicator(data["Close"], window=50).ema_indicator()
data["macd"] = ta.trend.MACD(data["Close"]).macd()
data["boll_high"] = ta.volatility.BollingerBands(data["Close"]).bollinger_hband()
data["boll_low"] = ta.volatility.BollingerBands(data["Close"]).bollinger_lband()

# Hedef değişken: ertesi gün tavan (%10 artış)
data["target"] = (data["Close"].pct_change().shift(-1) >= 0.099).astype(int)

# Sentiment sütunu (bugün için haber yoksa 0.5)
data["sentiment"] = 0.5
for i in range(len(data)):
    if i == len(data)-1:  # son gün için canlı sentiment
        data.iloc[i, data.columns.get_loc("sentiment")] = get_news_sentiment(ticker.split(".")[0])
    else:
        data.iloc[i, data.columns.get_loc("sentiment")] = 0.5  # geçmiş için nötr

data = data.dropna()

# -------------------------
# 3. LSTM için Zaman Serisi
# -------------------------
seq_len = 20
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[["Close", "Volume", "sentiment"]])

X_lstm, y_lstm = [], []
for i in range(len(scaled_data) - seq_len):
    X_lstm.append(scaled_data[i:i+seq_len])
    y_lstm.append(data["target"].iloc[i+seq_len])
X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

split_idx = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]

lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(seq_len, 3)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1, activation="sigmoid"))
lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=16, verbose=1)

lstm_preds = (lstm_model.predict(X_test_lstm) > 0.5).astype(int)

# -------------------------
# 4. XGBoost ile Teknik + Sentiment
# -------------------------
features = ["rsi", "ema20", "ema50", "macd", "boll_high", "boll_low", "Volume", "sentiment"]
X_xgb = data[features]
y_xgb = data["target"]

X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, shuffle=False)

xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
xgb_model.fit(X_train_xgb, y_train_xgb)

xgb_preds = xgb_model.predict(X_test_xgb)

# -------------------------
# 5. Hibrit Tahmin (Oylama)
# -------------------------
final_preds = []
for l, x in zip(lstm_preds.flatten(), xgb_preds):
    final_preds.append(1 if (l + x) >= 1 else 0)

print("\n--- Model Performansı ---")
print(classification_report(y_test_xgb[-len(final_preds):], final_preds))

# -------------------------
# 6. Canlı Tahmin
# -------------------------
last_lstm_input = np.array([scaled_data[-seq_len:]])
lstm_prob = lstm_model.predict(last_lstm_input)[0][0]

last_xgb_input = np.array([X_xgb.iloc[-1]])
xgb_prob = xgb_model.predict_proba(last_xgb_input)[0][1]

final_prob = (lstm_prob + xgb_prob) / 2
print(f"\nBugün için tavan olma olasılığı: %{final_prob*100:.2f}")
