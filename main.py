import os
import warnings
import logging

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)

import yfinance as yf
import numpy as np
import ta
import feedparser
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
from telegram import Bot
from datetime import datetime
import asyncio

# -------------------------
# Telegram Bot Configuration
# -------------------------
TELEGRAM_BOT_TOKEN = "8389484759:AAEzi-nJxb-OHwEo3lg5i8m1tv3eiY3Np4k"  # Replace with your bot token
TELEGRAM_CHAT_ID = "-1002758348312"     # Replace with your chat ID

async def send_telegram_message(message):
    """Send message via Telegram bot using python-telegram-bot"""
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message,
            parse_mode='HTML'
        )
        return True
    except Exception as e:
        print(f"Telegram mesaj gönderme hatası: {e}")
        return False

def send_telegram_message_sync(message):
    """Synchronous wrapper for sending Telegram messages"""
    try:
        return asyncio.run(send_telegram_message(message))
    except Exception as e:
        print(f"Telegram mesaj gönderme hatası: {e}")
        return False

# -------------------------
# 1. Haber Sentiment Analizi
# -------------------------
# noinspection PyTypeChecker
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

def analyze_ticker(ticker):
    """Analyze a single ticker and return prediction probability"""
    try:
        print(f"\n{ticker} analiz ediliyor...")
        
        # -------------------------
        # 2. Fiyat Verisi ve Teknikler
        # -------------------------
        try:
            data = yf.download(ticker, period="2y", progress=False)
        except Exception as e:
            print(f"❌ {ticker}: Veri indirme hatası - {str(e)}")
            return None
        
        if data.empty:
            print(f"❌ {ticker}: Veri bulunamadı!")
            return None
            
        if len(data) < 100:
            print(f"❌ {ticker}: Yetersiz veri ({len(data)} gün)")
            return None
            
        # Technical indicators with error handling
        try:
            data["rsi"] = ta.momentum.RSIIndicator(data["Close"].squeeze()).rsi()
            data["ema20"] = ta.trend.EMAIndicator(data["Close"].squeeze(), window=20).ema_indicator()
            data["ema50"] = ta.trend.EMAIndicator(data["Close"].squeeze(), window=50).ema_indicator()
            data["macd"] = ta.trend.MACD(data["Close"].squeeze()).macd()
            data["boll_high"] = ta.volatility.BollingerBands(data["Close"].squeeze()).bollinger_hband()
            data["boll_low"] = ta.volatility.BollingerBands(data["Close"].squeeze()).bollinger_lband()
        except Exception as e:
            print(f"❌ {ticker}: Teknik gösterge hesaplama hatası - {str(e)}")
            return None

        # Hedef değişken: ertesi gün tavan (%5 artış)
        try:
            data["target"] = (data["Close"].pct_change().shift(-1) >= 0.049).astype(int)
        except Exception as e:
            print(f"❌ {ticker}: Hedef değişken hesaplama hatası - {str(e)}")
            return None

        # Sentiment sütunu
        try:
            data["sentiment"] = 0.5
            for i in range(len(data)):
                if i == len(data)-1:  # son gün için canlı sentiment
                    try:
                        sentiment_score = get_news_sentiment(ticker.split(".")[0])
                        data.iloc[i, data.columns.get_loc("sentiment")] = sentiment_score
                    except Exception as e:
                        print(f"⚠️ {ticker}: Sentiment analizi hatası - {str(e)}, nötr değer kullanılıyor")
                        data.iloc[i, data.columns.get_loc("sentiment")] = 0.5
                else:
                    data.iloc[i, data.columns.get_loc("sentiment")] = 0.5
        except Exception as e:
            print(f"❌ {ticker}: Sentiment hesaplama hatası - {str(e)}")
            return None

        data = data.dropna()
        
        if len(data) < 50:  # Yeterli veri yoksa
            print(f"❌ {ticker}: Temizleme sonrası yetersiz veri ({len(data)} satır)")
            return None

        # -------------------------
        # 3. LSTM için Zaman Serisi
        # -------------------------
        try:
            seq_len = 20
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[["Close", "Volume", "sentiment"]])

            X_lstm, y_lstm = [], []
            for i in range(len(scaled_data) - seq_len):
                X_lstm.append(scaled_data[i:i+seq_len])
                y_lstm.append(data["target"].iloc[i+seq_len])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

            if len(X_lstm) < 20:  # Yeterli veri yoksa
                print(f"❌ {ticker}: LSTM eğitimi için yetersiz veri ({len(X_lstm)} örnek)")
                return None

            split_idx = int(len(X_lstm) * 0.8)
            X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
            y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]

            # Check if we have positive examples in training data
            positive_examples = np.sum(y_train_lstm)
            total_examples = len(y_train_lstm)
            print(f"  📊 {ticker}: {positive_examples}/{total_examples} pozitif örnek (%{positive_examples/total_examples*100:.1f})")
            
            if positive_examples == 0:
                print(f"  ⚠️ {ticker}: Hiç pozitif örnek yok, LSTM baseline kullanılıyor!")
                lstm_prob = 0.01  # Very low but not zero
            else:
                lstm_model = Sequential()
                lstm_model.add(LSTM(50, return_sequences=True, input_shape=(seq_len, 3)))
                lstm_model.add(Dropout(0.2))
                lstm_model.add(LSTM(50))
                lstm_model.add(Dropout(0.2))
                lstm_model.add(Dense(1, activation="sigmoid"))
                lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                
                lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=8, verbose=0)
                # Get LSTM prediction
                last_lstm_input = np.array([scaled_data[-seq_len:]])
                lstm_prob = lstm_model.predict(last_lstm_input, verbose=0)[0][0]
                
        except Exception as e:
            print(f"❌ {ticker}: LSTM model hatası - {str(e)}")
            return None

        # -------------------------
        # 4. XGBoost ile Teknik + Sentiment
        # -------------------------
        try:
            features = ["rsi", "ema20", "ema50", "macd", "boll_high", "boll_low", "Volume", "sentiment"]
            X_xgb = data[features]
            y_xgb = data["target"]

            X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, shuffle=False)

            xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, base_score=0.5)
            xgb_model.fit(X_train_xgb, y_train_xgb)

            # -------------------------
            # 5. Canlı Tahmin
            # -------------------------
            last_xgb_input = np.array([X_xgb.iloc[-1]])
            xgb_prob = xgb_model.predict_proba(last_xgb_input)[0][1]
            
        except Exception as e:
            print(f"❌ {ticker}: XGBoost model hatası - {str(e)}")
            return None

        final_prob = (lstm_prob + xgb_prob) / 2
        
        current_price = data["Close"].iloc[-1].item()
        
        return {
            "ticker": ticker,
            "probability": final_prob,
            "current_price": current_price,
            "lstm_prob": lstm_prob,
            "xgb_prob": xgb_prob
        }
        
    except Exception as e:
        print(f"{ticker} analiz hatası: {e}")
        return None

def analyze_multiple_tickers(tickers):
    """Analyze multiple tickers and return results"""
    results = []
    
    for ticker in tickers:
        result = analyze_ticker(ticker)
        if result:
            results.append(result)
    
    return results

def format_telegram_message(results):
    """Format analysis results for Telegram message"""
    if not results:
        return "❌ Hiçbir hisse analiz edilemedi!"
    
    message = f"📊 <b>Borsa Tahmin Raporu</b>\n"
    message += f"🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"
    
    # Sort by probability (highest first)
    results.sort(key=lambda x: x["probability"], reverse=True)
    
    for result in results:
        ticker = result["ticker"]
        prob = result["probability"] * 100
        price = result["current_price"]
        
        if prob >= 70:
            emoji = "🚀"
        elif prob >= 50:
            emoji = "📈"
        else:
            emoji = "📉"
            
        message += f"{emoji} <b>{ticker}</b>\n"
        message += f"💰 Fiyat: {price:.2f} TL\n"
        message += f"📊 Tavan Olasılığı: %{prob:.1f}\n"
        message += f"🤖 LSTM: %{result['lstm_prob']*100:.1f} | XGB: %{result['xgb_prob']*100:.1f}\n\n"
    
    return message

# -------------------------
# Ana Program
# -------------------------
if __name__ == "__main__":
    # Analiz edilecek hisseler (BIST kodları)
    TICKERS = [
        "THYAO.IS",  # Türk Hava Yolları
        "AKBNK.IS",  # Akbank
        "GARAN.IS",  # Garanti BBVA
        "ISCTR.IS",  # İş Bankası
        "KCHOL.IS",  # Koç Holding
        "SAHOL.IS",  # Sabancı Holding
        "TCELL.IS",  # Turkcell
        "TUPRS.IS",  # Tüpraş
        "BIMAS.IS",  # BİM
        "ASELS.IS"   # Aselsan
    ]
    
    print("🚀 Çoklu hisse analizi başlatılıyor...")
    print(f"📋 Analiz edilecek hisseler: {', '.join([t.replace('.IS', '') for t in TICKERS])}")
    
    # Analiz yap
    results = analyze_multiple_tickers(TICKERS)
    
    # Sonuçları göster
    if results:
        print(f"\n✅ {len(results)} hisse başarıyla analiz edildi!")
        
        # Telegram mesajı hazırla
        telegram_message = format_telegram_message(results)
        print("\n" + "="*50)
        print("TELEGRAM MESAJI:")
        print("="*50)
        print(telegram_message.replace("<b>", "").replace("</b>", ""))
        
        # Telegram'a gönder (bot token ve chat ID ayarlanmışsa)
        if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE" and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID_HERE":
            if send_telegram_message_sync(telegram_message):
                print("✅ Telegram mesajı başarıyla gönderildi!")
            else:
                print("❌ Telegram mesajı gönderilemedi!")
        else:
            print("⚠️  Telegram bot ayarları yapılmamış. Lütfen TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID değerlerini güncelleyin.")
            print("📦 Gerekli paket: pip install python-telegram-bot")
    else:
        print("❌ Hiçbir hisse analiz edilemedi!")
