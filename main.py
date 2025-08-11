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
TELEGRAM_BOT_TOKEN = "8389484759:AAEzi-nJxb-OHwEo3lg5i8m1tv3eiY3Np4k"
TELEGRAM_CHAT_ID = "-1002758348312" 
SEND = True

if TELEGRAM_BOT_TOKEN is None or TELEGRAM_CHAT_ID is None:
    TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
    TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

target_percent = 0
# 2 - %15, 1 - %10, 0 - %5, -1 - %3

target = 0.149 if target_percent == 2 else 0.099 if target_percent == 1 else 0.049 if target_percent == 0 else 0.029
async def send_telegram_message(message):
    """Send message via Telegram bot using python-telegram-bot"""
    if not SEND:
        print("Telegram mesajları şu anda devre dışı")
        return False
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        sent_message = await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=message,
            parse_mode='HTML'
        )
        return sent_message.message_id
    except Exception as e:
        print(f"Telegram mesaj gönderme hatası: {e}")
        return False

async def edit_telegram_message(message_id, new_text):
    """Edit an existing Telegram message"""
    if not SEND:
        return False
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        await bot.edit_message_text(
            chat_id=TELEGRAM_CHAT_ID,
            message_id=message_id,
            text=new_text,
            parse_mode='HTML'
        )
        return True
    except Exception as e:
        print(f"Telegram mesaj düzenleme hatası: {e}")
        return False

def send_telegram_message_sync(message):
    """Synchronous wrapper for sending Telegram messages"""
    if not SEND:
        return False
    try:
        if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE" and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID_HERE":
            return asyncio.run(send_telegram_message(message))
        else:
            return False
    except Exception as e:
        print(f"Telegram mesaj gönderme hatası: {e}")
        return False

def edit_telegram_message_sync(message_id, new_text):
    """Synchronous wrapper for editing Telegram messages"""
    if not SEND:
        return False
    try:
        if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE" and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID_HERE":
            return asyncio.run(edit_telegram_message(message_id, new_text))
        else:
            print("⚠️  Telegram bot ayarları yapılmamış. Lütfen TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID değerlerini güncelleyin.")
            print("📦 Gerekli paket: pip install python-telegram-bot")
    except Exception as e:
        print(f"Telegram mesaj gönderme hatası: {e}")
        return False

# -------------------------
# 1. Haber Sentiment Analizi
# -------------------------
# noinspection PyTypeChecker
sentiment_model = pipeline("sentiment-analysis", model="savasy/bert-base-turkish-sentiment-cased")

def get_news_sentiment(query):
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
        print(f"\n{ticker.replace('.IS', '')} analiz ediliyor...")
        
        # -------------------------
        # 2. Fiyat Verisi ve Teknikler
        # -------------------------
        try:
            data = yf.download(ticker, period="2y", progress=False)
        except Exception as e:
            print(f"❌ {ticker.replace('.IS', '')}: Veri indirme hatası - {str(e)}")
            return None
        
        if data.empty:
            print(f"❌ {ticker.replace('.IS', '')}: Veri bulunamadı!")
            return None
            
        if len(data) < 100:
            print(f"❌ {ticker.replace('.IS', '')}: Yetersiz veri ({len(data)} gün)")
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
            print(f"❌ {ticker.replace('.IS', '')}: Teknik gösterge hesaplama hatası - {str(e)}")
            return None

        # Hedef değişken: ertesi gün tavan (%5 artış)
        try:
            data["target"] = (data["Close"].pct_change().shift(-1) >= target).astype(int)
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
                        print(f"  📊 {ticker}: Son gün sentiment puanı: %{sentiment_score*100:.1f}")
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
            print(f"  📊 {ticker.replace('.IS', '')}: {positive_examples}/{total_examples} pozitif örnek (%{positive_examples/total_examples*100:.1f})")
            
            if positive_examples == 0:
                print(f"  ⚠️ {ticker.replace('.IS', '')}: Hiç pozitif örnek yok, LSTM baseline kullanılıyor!")
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

def quick_screen_ticker(ticker):
    """Quick screening using basic technical indicators"""
    try:
        # Download only recent data for speed
        data = yf.download(ticker, period="3mo", progress=False)
        if data.empty or len(data) < 20:
            return None
            
        # Quick technical indicators
        close = data["Close"].squeeze()
        volume = data["Volume"].squeeze()
        
        # Basic momentum indicators
        rsi = ta.momentum.RSIIndicator(close).rsi().iloc[-1]
        ema20 = ta.trend.EMAIndicator(close, window=20).ema_indicator().iloc[-1]
        current_price = close.iloc[-1]
        
        # Volume analysis
        avg_volume = volume.rolling(20).mean().iloc[-1]
        recent_volume = volume.iloc[-1]
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        # Price momentum
        price_change_5d = (current_price / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
        price_above_ema = current_price > ema20
        
        # Screening criteria
        score = 0
        if 30 <= rsi <= 70:  # Not overbought/oversold
            score += 1
        if price_above_ema:  # Above EMA20
            score += 1
        if volume_ratio > 1.2:  # Above average volume
            score += 1
        if price_change_5d > 0:  # Positive momentum
            score += 1
            
        return {
            "ticker": ticker,
            "score": score,
            "rsi": rsi,
            "price": current_price,
            "volume_ratio": volume_ratio,
            "momentum_5d": price_change_5d,
            "above_ema": price_above_ema
        }
        
    except Exception as e:
        print(f"⚠️ {ticker}: Hızlı tarama hatası - {str(e)}")
        return None

def analyze_multiple_tickers(tickers):
    """Two-stage analysis: quick screening then detailed analysis"""
    print("🔍 1. Aşama: Hızlı tarama başlatılıyor...")
    send_telegram_message_sync("🔍 1. Aşama: Hızlı tarama başlatılıyor...")
    
    # Stage 1: Quick screening
    screening_results = []
    for ticker in tickers:
        result = quick_screen_ticker(ticker)
        if result:
            screening_results.append(result)
    
    # Sort by score and select top candidates
    screening_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Select top candidates - be more selective
    # First try score >= 3, then top 15 if not enough
    promising_tickers = [r for r in screening_results if r["score"] >= 3]
    if len(promising_tickers) < 5:
        promising_tickers = [r for r in screening_results if r["score"] >= 2][:15]
    if len(promising_tickers) > 20:  # Cap at 20 for efficiency
        promising_tickers = promising_tickers[:20]
    
    print(f"📊 Tarama tamamlandı: {len(screening_results)} hisse tarandı")
    print(f"🎯 Detaylı analiz için seçilen: {len(promising_tickers)} hisse")
    send_telegram_message_sync(f"📊 Tarama tamamlandı: {len(screening_results)} hisse tarandı")
    send_telegram_message_sync(f"🎯 Detaylı analiz için seçilen: {len(promising_tickers)} hisse")
    
    # Show screening results
    print("\n📋 Hızlı Tarama Sonuçları(İlk 10):")
    msg = "📋 Hızlı Tarama Sonuçları(İlk 10):\n"
    for result in screening_results[:10]:  # Show top 10
        ticker = result["ticker"]
        score = result["score"]
        rsi = result["rsi"]
        momentum = result["momentum_5d"]
        volume_ratio = result["volume_ratio"]
        
        status = "🎯" if result in promising_tickers else "📊"
        line = f"{status} {ticker.replace('.IS', '')}: Skor={score}/4, RSI={rsi:.1f}, Momentum={momentum:+.1f}%, Hacim={volume_ratio:.1f}x"
        print(line)
        msg += f"{line}\n"
    send_telegram_message_sync(msg)
    
    # Stage 2: Detailed analysis on promising tickers with live progress
    print(f"\n🔬 2. Aşama: Detaylı analiz başlatılıyor...")
    detailed_results = []
    
    # Send initial progress message
    progress_message_id = send_telegram_message_sync(
        f"🔬 <b>Detaylı Analiz Başlatılıyor</b>\n"
        f"📊 Seçilen hisse sayısı: {len(promising_tickers)}\n"
        f"⏳ İlerleme: 0/{len(promising_tickers)}\n"
        f"🎯 Hedef: %{target*100:.1f} artış"
    )
    
    for i, result in enumerate(promising_tickers, 1):
        ticker = result["ticker"]
        
        # Update progress message
        if progress_message_id:
            progress_bars = "█" * (i * 10 // len(promising_tickers))
            empty_bars = "░" * (10 - len(progress_bars))
            progress_text = (
                f"🔬 <b>Detaylı Analiz Devam Ediyor</b>\n"
                f"📊 Seçilen hisse sayısı: {len(promising_tickers)}\n"
                f"⏳ İlerleme: {i}/{len(promising_tickers)}\n"
                f"📈 [{progress_bars}{empty_bars}] %{i*100//len(promising_tickers)}\n"
                f"🔍 Şu an analiz edilen: <b>{ticker.replace('.IS', '')}</b>\n"
                f"🎯 Hedef: %{target*100:.1f} artış"
            )
            edit_telegram_message_sync(progress_message_id, progress_text)
        
        detailed_result = analyze_ticker(ticker)
        if detailed_result:
            # Add screening info to detailed result
            detailed_result["screening_score"] = result["score"]
            detailed_result["volume_ratio"] = result["volume_ratio"]
            detailed_result["momentum_5d"] = result["momentum_5d"]
            detailed_results.append(detailed_result)
    
    # Final progress update
    if progress_message_id:
        final_text = (
            f"✅ <b>Detaylı Analiz Tamamlandı!</b>\n"
            f"📊 Analiz edilen: {len(promising_tickers)} hisse\n"
            f"🎯 Başarılı: {len(detailed_results)} hisse\n"
            f"📈 [██████████] %100\n"
            f"🚀 Sonuçlar hazırlanıyor..."
        )
        edit_telegram_message_sync(progress_message_id, final_text)
    
    return detailed_results

def format_telegram_message(scan_results):
    """Format analysis results for Telegram message"""
    if not scan_results:
        return "❌ Hiçbir hisse analiz edilemedi!"
    
    message = f"📊 <b>Borsa Tahmin Raporu</b>\n"
    message += f"🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"
    
    # Sort by probability (highest first)
    scan_results.sort(key=lambda x: x["probability"], reverse=True)
    
    for result in scan_results:
        ticker = result["ticker"]
        prob = result["probability"] * 100
        price = result["current_price"]
        
        if prob >= 30:
            emoji = "🚀"
        elif prob >= 10:
            emoji = "📈"
        else:
            emoji = "📉"
            
        message += f"{emoji} <b>{ticker.replace('.IS', '')}</b>\n"
        message += f"💰 Fiyat: {price:.2f} TL\n"
        message += f"📊 Tavan Olasılığı: %{prob:.1f}\n"
        message += f"🤖 LSTM: %{result['lstm_prob']*100:.1f} | XGB: %{result['xgb_prob']*100:.1f}\n"
        
        # Add screening info if available
        if 'screening_score' in result:
            score = result['screening_score']
            volume_ratio = result.get('volume_ratio', 0)
            momentum = result.get('momentum_5d', 0)
            message += f"🔍 Tarama: {score}/4 | Hacim: {volume_ratio:.1f}x | Momentum: {momentum:+.1f}%\n"
        
        message += "\n"
    
    return message

# -------------------------
# Ana Program
# -------------------------
if __name__ == "__main__":
    # Comprehensive BIST stock list - Updated with more valid active stocks
    TICKERS = [
        # BIST 30 Core Stocks
        "THYAO.IS", "AKBNK.IS", "GARAN.IS", "ISCTR.IS", "KCHOL.IS", "SAHOL.IS",
        "TCELL.IS", "TUPRS.IS", "BIMAS.IS", "ASELS.IS", "HALKB.IS", "VAKBN.IS",
        "SISE.IS", "PETKM.IS", "KOZAL.IS", "KOZAA.IS", "EREGL.IS", "ARCLK.IS",
        "TOASO.IS", "PGSUS.IS", "MGROS.IS", "TAVHL.IS", "OYAKC.IS", "DOHOL.IS",
        
        # Banking & Finance
        "AKBNK.IS", "GARAN.IS", "HALKB.IS", "ISCTR.IS", "VAKBN.IS", "YKBNK.IS",
        "TSKB.IS", "ICBCT.IS", "ALBRK.IS", "SKBNK.IS", "KLNMA.IS", "GOZDE.IS",
        
        # Technology & Software
        "TCELL.IS", "TTKOM.IS", "NETAS.IS", "LOGO.IS", "KRONT.IS", "INDES.IS",
        "ARENA.IS", "DESPC.IS", "SMART.IS", "LINK.IS", "ESCOM.IS", "PATEK.IS",
        
        # Industrial & Manufacturing (removed delisted: TRKCM)
        "ASELS.IS", "EREGL.IS", "ARCLK.IS", "TOASO.IS", "OTKAR.IS", "FROTO.IS",
        "KARSN.IS", "VESTL.IS", "CEMTS.IS", "AKSA.IS", "PARSN.IS",
        "IHLAS.IS", "IHEVA.IS", "IHLGM.IS", "KARTN.IS", "KLMSN.IS", "KONYA.IS",
        
        # Energy & Utilities
        "TUPRS.IS", "PETKM.IS", "TRGYO.IS", "AKSEN.IS", "AKENR.IS", "ZOREN.IS",
        "ENKAI.IS", "GESAN.IS", "AYGAZ.IS", "ODAS.IS", "SNGYO.IS", "TKNSA.IS",
        "ENJSA.IS", "EPLAS.IS", "EGGUB.IS", "EGSER.IS",
        
        # Retail & Consumer Goods (removed delisted: MIPAZ)
        "BIMAS.IS", "MGROS.IS", "SOKM.IS", "MAVI.IS", "BIZIM.IS", "CRFSA.IS",
        "ULKER.IS", "KNFRT.IS", "PETUN.IS", "MPARK.IS", "METUR.IS",
        
        # Construction & Real Estate
        "TRGYO.IS", "GLYHO.IS", "RYGYO.IS", "VKGYO.IS", "ISGYO.IS", "AVGYO.IS",
        "DGGYO.IS", "EKGYO.IS", "FMIZP.IS", "GRNYO.IS", "HLGYO.IS", "ISFIN.IS",
        
        # Healthcare & Pharmaceuticals
        "SNGYO.IS", "DEVA.IS", "ECILC.IS", "LKMNH.IS", "SELEC.IS", "ALKIM.IS",
        "DAGI.IS", "DOGUB.IS", "DURDO.IS", "ETYAT.IS",
        
        # Transportation & Logistics
        "THYAO.IS", "PGSUS.IS", "CLEBI.IS", "DOCO.IS", "GSDHO.IS", "RYSAS.IS",
        "BEYAZ.IS", "BLCYT.IS", "BSOKE.IS", "BTCIM.IS",
        
        # Chemicals & Materials (removed delisted: AKKIM, ALTIN, ANACM)
        "GUBRF.IS", "BRSAN.IS", "BAGFS.IS", "DMSAS.IS", "PRKME.IS", "ALKIM.IS",
        "AKMGY.IS", "ALCAR.IS", "ALKA.IS",
        
        # Textiles & Fashion
        "YUNSA.IS", "BRMEN.IS", "DAGI.IS", "HATEK.IS", "KORDS.IS", "SNKRN.IS",
        "ATEKS.IS", "ARSAN.IS", "BLCYT.IS", "BOSSA.IS", "BRKO.IS", "DESA.IS",
        
        # Food & Beverage (removed delisted: ALYAG, KERVT)
        "ULKER.IS", "CCOLA.IS", "AEFES.IS", "BANVT.IS", "KENT.IS", "PINSU.IS",
        "AVOD.IS", "BANVT.IS", "CCOLA.IS", "ERSU.IS",
        
        # Mining & Metals (removed delisted: DGKLB)
        "KOZAL.IS", "KOZAA.IS", "CMENT.IS", "DOCO.IS",
        "EGEEN.IS", "EGEPO.IS", "EGSER.IS", "ERBOS.IS", "ETYAT.IS", "FENER.IS",
        
        # Media & Entertainment
        "IHLAS.IS", "IHEVA.IS", "IHLGM.IS", "IHYAY.IS", "INTEM.IS", "ISBIR.IS",
        
        # Additional High-Volume Stocks
        "ADEL.IS", "ADESE.IS", "AEFES.IS", "AFYON.IS", "AGESA.IS", "AGHOL.IS",
        "AGROT.IS", "AHGAZ.IS", "AKFGY.IS", "AKFYE.IS", "AKMGY.IS", "AKSGY.IS",
        "AKSUE.IS", "AKYHO.IS", "ALARK.IS", "ALBRK.IS", "ALCAR.IS", "ALCTL.IS"
    ]
    
    # Remove duplicates and sort
    TICKERS = sorted(list(set(TICKERS)))
    
    print("🚀 Çoklu hisse analizi başlatılıyor...")
    print(f"🚀 Hedef: %{target * 100:.1f} artış")
    print(f"📋 Analiz edilecek hisseler: {', '.join([t.replace('.IS', '') for t in TICKERS])}")
    send_telegram_message_sync("🚀 Çoklu hisse analizi başlatılıyor...")
    send_telegram_message_sync(f"🚀 Hedef: %{target * 100:.1f} artış")
    send_telegram_message_sync(f"📋 {len(TICKERS)} hisse analiz edilecek")
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
        
        send_telegram_message_sync(telegram_message)
    else:
        print("❌ Hiçbir hisse analiz edilemedi!")
