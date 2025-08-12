# Start time measurement
import time
start_time = time.time()

import os
import warnings
import logging
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from pathlib import Path

# Suppress warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Allow GPU memory growth
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

import yfinance as yf
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import xgboost as xgb
from telegram import Bot
from datetime import datetime
from typing import Any
import requests
import asyncio
import json
import psutil
from google import genai

# -------------------------
# Telegram Bot Configuration
# -------------------------
SEND = int(os.environ.get("SEND", 1))
SEND_ADVANCED = int(os.environ.get("SEND_ADVANCED", 0))

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "7950923465:AAFHznvCCp9-X99fqdX2RahAO7l92s5PIfE")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "-1002758348312")

# -------------------------
# LLM API Configuration (Google Gemini)
# -------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyD0xR1DWKj4IANbS2-DF1zdwtStlOclSK8")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash-lite")
USE_LLM = int(os.environ.get("USE_LLM", 1))

target_percent = int(os.environ.get("TARGET_PERCENT", 0))
# 2 - %15, 1 - %10, 0 - %5, -1 - %3

target = 0.149 if target_percent == 2 else 0.099 if target_percent == 1 else 0.049 if target_percent == 0 else 0.029

def log_memory_usage():
    """Log current memory usage for monitoring"""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"🔧 Bellek kullanımı: {memory_mb:.1f} MB", flush=True)
        return memory_mb
    except Exception:
        return 0

# Data caching system
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def get_cache_path(ticker, period):
    """Get cache file path for ticker data"""
    return CACHE_DIR / f"{ticker}_{period}.pkl"

def load_cached_data(ticker, period, max_age_hours=1):
    """Load cached data if it exists and is recent enough"""
    cache_path = get_cache_path(ticker, period)
    if cache_path.exists():
        try:
            # Check file age
            file_age = datetime.now().timestamp() - cache_path.stat().st_mtime
            if file_age < max_age_hours * 3600:  # Convert hours to seconds
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        except Exception:
            pass
    return None

def save_cached_data(ticker, period, data):
    """Save data to cache"""
    try:
        cache_path = get_cache_path(ticker, period)
        with open(cache_path, 'wb') as f:
            f: Any
            pickle.dump(data, f)
    except Exception:
        pass

def download_with_cache(ticker, period="1y", max_age_hours=1):
    """Download data with caching support"""
    # Try to load from cache first
    cached_data = load_cached_data(ticker, period, max_age_hours)
    if cached_data is not None:
        return cached_data
    
    # Download fresh data
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True, threads=False)
        # Save to cache
        save_cached_data(ticker, period, data)
        return data
    except Exception as e:
        print(f"❌ {ticker}: Veri indirme hatası - {str(e)}", flush=True)
        return None

async def send_telegram_message(msg):
    """Send message via Telegram bot using python-telegram-bot"""
    if not SEND:
        print("Telegram mesajları şu anda devre dışı", flush=True)
        return False
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN)
        sent_message = await bot.send_message(
            chat_id=TELEGRAM_CHAT_ID,
            text=msg,
            parse_mode='HTML'
        )
        return sent_message.message_id
    except Exception as e:
        print(f"Telegram mesaj gönderme hatası: {e}", flush=True)
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
        print(f"Telegram mesaj düzenleme hatası: {e}", flush=True)
        return False

def send_telegram_message_sync(msg):
    """Synchronous wrapper for sending Telegram messages"""
    if not SEND:
        return False
    try:
        if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE" and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID_HERE":
            return asyncio.run(send_telegram_message(msg))
        else:
            return False
    except Exception as e:
        print(f"Telegram mesaj gönderme hatası: {e}", flush=True)
        return False

def edit_telegram_message_sync(message_id, new_text):
    """Synchronous wrapper for editing Telegram messages"""
    if not SEND:
        return False
    try:
        if TELEGRAM_BOT_TOKEN != "YOUR_BOT_TOKEN_HERE" and TELEGRAM_CHAT_ID != "YOUR_CHAT_ID_HERE":
            return asyncio.run(edit_telegram_message(message_id, new_text))
        else:
            return False
    except Exception as e:
        print(f"Telegram mesaj gönderme hatası: {e}", flush=True)
        return False

def analyze_with_llm(scan_results):
    """Send analysis results to LLM for summarization and buy-sell recommendations"""
    if not USE_LLM or not scan_results:
        return None
    
    try:
        # Configure Gemini API
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Prepare data for LLM analysis
        analysis_data = []
        for result in scan_results:
            ticker_name = result["ticker"].replace('.IS', '')
            analysis_data.append({
                "stock": ticker_name,
                "probability": f"{result['probability']*100:.1f}%",
                "current_price": f"{result['current_price']:.2f} TL",
                "lstm_prediction": f"{result['lstm_prob']*100:.1f}%",
                "xgboost_prediction": f"{result['xgb_prob']*100:.1f}%",
                "screening_score": result.get('screening_score', 'N/A'),
                "volume_ratio": f"{result.get('volume_ratio', 0):.1f}x",
                "momentum_5d": f"{result.get('momentum_5d', 0):+.1f}%"
            })
        
        # Create prompt for LLM
        prompt = f"""
Sen bir finansal analiz uzmanısın. Teknik analiz verilerini yorumlayarak yatırımcılara rehberlik ediyorsun. Yanıtların objektif, veri odaklı ve risk uyarıları içermelidir.

Aşağıda Türk borsasından {len(analysis_data)} hissenin teknik analiz sonuçları bulunmaktadır. 
Her hisse için LSTM ve XGBoost modelleri kullanılarak %{target*100:.1f} artış olasılığı hesaplanmıştır.

Analiz Sonuçları:
{json.dumps(analysis_data, indent=2, ensure_ascii=False)}

Lütfen bu verileri analiz ederek:
1. Genel piyasa durumu hakkında kısa bir özet
2. En yüksek potansiyelli 3-5 hisse için BUY önerisi ve nedenleri
3. Riskli görünen hisseler için SELL/HOLD önerisi
4. Genel yatırım stratejisi önerisi
5. Risk yönetimi tavsiyeleri

Paragraf aralarında --- kullan, başka hiçbir yerde kullanma, paragraflar çok uzun olursa yazdığın mesaj gönderilmeyecek

ÖNEMLI: Yanıtında formatlamak için HTML etiketleri kullan:
- Kalın yazı için her zaman: <b>metin</b>
- İtalik için: <i>metin</i>
- Hisse kodları ve önemli bilgiler için <b> kullan örn: THYAO yada BUY
- Ne olursa olsun Markdown (**bold**) kullanma, sadece HTML kullan

Art arda aşırı fazla yeni satırdan kaçın en fazla arka arkaya iki tane!
Yanıtını Türkçe olarak, yatırımcılar için anlaşılır bir dilde ver. Daha çok resmi değil samimi bir dil kullan, sanki olar aile üyelerinmiş gibi. Finansal tavsiye değil, sadece teknik analiz yorumu olduğunu belirt.
"""

        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"❌ LLM analizi hatası: {str(e)}", flush=True)
        return None


def analyze_ticker(ticker, term_use=0):
    """Analyze a single ticker and return prediction probability"""
    try:
        print(f"\n{ticker.replace('.IS', '')} analiz ediliyor...", flush=True)
        
        # -------------------------
        # 2. Fiyat Verisi ve Teknikler
        # -------------------------
        data = download_with_cache(ticker)
        if data is None:
            return None
        
        if data.empty:
            print(f"❌ {ticker.replace('.IS', '')}: Veri bulunamadı!", flush=True)
            return None
            
        if len(data) < 100:
            print(f"❌ {ticker.replace('.IS', '')}: Yetersiz veri ({len(data)} gün)", flush=True)
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
            print(f"❌ {ticker.replace('.IS', '')}: Teknik gösterge hesaplama hatası - {str(e)}", flush=True)
            return None

        # Hedef değişken: ertesi gün tavan (%5 artış)
        try:
            data["target"] = (data["Close"].pct_change().shift(-1) >= target).astype(int)
        except Exception as e:
            print(f"❌ {ticker}: Hedef değişken hesaplama hatası - {str(e)}", flush=True)
            return None

        data = data.dropna()
        
        if len(data) < 50:  # Yeterli veri yoksa
            print(f"❌ {ticker.replace('.IS', '')}: Temizleme sonrası yetersiz veri ({len(data)} satır)", flush=True)
            return None

        # -------------------------
        # 3. LSTM için Zaman Serisi
        # -------------------------
        try:
            seq_len = 20
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data[["Close", "Volume"]])

            X_lstm, y_lstm = [], []
            for i in range(len(scaled_data) - seq_len):
                X_lstm.append(scaled_data[i:i+seq_len])
                y_lstm.append(data["target"].iloc[i+seq_len])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

            if len(X_lstm) < 20:  # Yeterli veri yoksa
                print(f"❌ {ticker}: LSTM eğitimi için yetersiz veri ({len(X_lstm)} örnek)", flush=True)
                return None

            split_idx = int(len(X_lstm) * 0.8)
            X_train_lstm, X_test_lstm = X_lstm[:split_idx], X_lstm[split_idx:]
            y_train_lstm, y_test_lstm = y_lstm[:split_idx], y_lstm[split_idx:]

            # Check if we have positive examples in training data
            positive_examples = np.sum(y_train_lstm)
            total_examples = len(y_train_lstm)
            print(f"  📊 {ticker.replace('.IS', '')}: {positive_examples}/{total_examples} pozitif örnek (%{positive_examples/total_examples*100:.1f})", flush=True)
            
            if positive_examples == 0:
                print(f"  ⚠️ {ticker.replace('.IS', '')}: Hiç pozitif örnek yok, LSTM baseline kullanılıyor!", flush=True)
                lstm_prob = 0.01  # Very low but not zero
            else:
                lstm_model = Sequential()
                lstm_model.add(LSTM(32, return_sequences=True, input_shape=(seq_len, 2)))  # Reduced units
                lstm_model.add(Dropout(0.3))
                lstm_model.add(LSTM(32))  # Reduced units
                lstm_model.add(Dropout(0.3))
                lstm_model.add(Dense(1, activation="sigmoid"))
                lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
                
                # Adaptive epochs based on data size for efficiency
                epochs = min(15, max(5, len(X_train_lstm) // 10))
                lstm_model.fit(X_train_lstm, y_train_lstm, epochs=epochs, batch_size=16, verbose=0)
                # Get LSTM prediction
                last_lstm_input = np.array([scaled_data[-seq_len:]])
                lstm_prob = lstm_model.predict(last_lstm_input, verbose=0)[0][0]
                
        except Exception as e:
            print(f"❌ {ticker}: LSTM model hatası - {str(e)}", flush=True)
            return None

        # -------------------------
        # 4. XGBoost ile Teknik + Sentiment
        # -------------------------
        try:
            features = ["rsi", "ema20", "ema50", "macd", "boll_high", "boll_low", "Volume"]
            X_xgb = data[features]
            y_xgb = data["target"]

            X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(X_xgb, y_xgb, test_size=0.2, shuffle=False)

            # Optimized XGBoost parameters for speed vs accuracy balance
            n_estimators = min(80, max(30, len(X_train_xgb) // 5))  # Adaptive estimators
            xgb_model = xgb.XGBClassifier(
                n_estimators=n_estimators, 
                learning_rate=0.1,  # Slightly higher for faster convergence
                max_depth=4,  # Reduced depth for speed
                random_state=42, 
                base_score=0.5,
                n_jobs=1,  # Single thread to avoid conflicts in parallel processing
                verbosity=0
            )
            xgb_model.fit(X_train_xgb, y_train_xgb)

            # -------------------------
            # 5. Canlı Tahmin
            # -------------------------
            last_xgb_input = np.array([X_xgb.iloc[-1]])
            xgb_prob = xgb_model.predict_proba(last_xgb_input)[0][1]
            
        except Exception as e:
            print(f"❌ {ticker}: XGBoost model hatası - {str(e)}", flush=True)
            return None

        final_prob = (lstm_prob + xgb_prob) / 2
        
        current_price = data["Close"].iloc[-1].item()
        
        # Clean up memory
        del data, X_lstm, y_lstm, X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm
        del X_xgb, y_xgb, X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb
        del lstm_model, xgb_model, scaler, scaled_data
        gc.collect()
        
        if term_use != 1:
            return {
                "ticker": ticker,
                "probability": final_prob,
                "current_price": current_price,
                "lstm_prob": lstm_prob,
                "xgb_prob": xgb_prob
            }
        else:
            print(f"{ticker} Analizi Sonucu")
            print("Tavan yüzdesi: ", final_prob * 100)
            print("Güncel fiyat: ", current_price)
            print("LSTM tahmini: ", lstm_prob * 100)
            print("XGBoost tahmini: ", xgb_prob * 100)
        
    except Exception as e:
        print(f"{ticker} analiz hatası: {e}", flush=True)
        gc.collect()  # Clean up on error too
        return None

def quick_screen_ticker(ticker):
    """Quick screening using basic technical indicators"""
    try:
        # Download only recent data for speed with caching
        data = download_with_cache(ticker, period="3mo", max_age_hours=2)
        if data is None or data.empty or len(data) < 20:
            return None
            
        # Quick technical indicators - handle multi-dimensional data
        try:
            # Handle potential MultiIndex columns from yfinance
            if hasattr(data.columns, 'levels'):  # MultiIndex columns
                close = data.iloc[:, data.columns.get_level_values(0) == 'Close'].iloc[:, 0]
                volume = data.iloc[:, data.columns.get_level_values(0) == 'Volume'].iloc[:, 0]
            else:
                close = data["Close"]
                volume = data["Volume"]
            
            # Ensure 1-dimensional Series
            if hasattr(close, 'squeeze'):
                close = close.squeeze()
            if hasattr(volume, 'squeeze'):
                volume = volume.squeeze()
                
        except Exception as e:
            print(f"⚠️ {ticker}: Veri yapısı hatası - {str(e)}", flush=True)
            return None
        
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
        print(f"⚠️ {ticker}: Hızlı tarama hatası - {str(e)}", flush=True)
        return None

def analyze_multiple_tickers(tickers):
    """Two-stage analysis: quick screening then detailed analysis"""
    print("🔍 Hızlı tarama başlatılıyor...", flush=True)
    send_telegram_message_sync("🔍 Hızlı tarama başlatılıyor...")
    
    # Stage 1: Quick screening with parallel processing
    screening_results = []
    max_workers = min(8, len(tickers))  # Limit concurrent connections
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all screening tasks
        future_to_ticker = {executor.submit(quick_screen_ticker, ticker): ticker for ticker in tickers}
        
        # Collect results as they complete
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result:
                    screening_results.append(result)
            except Exception as e:
                print(f"⚠️ {ticker}: Tarama hatası - {str(e)}", flush=True)
    
    # Sort by score and select top candidates
    screening_results.sort(key=lambda x: x["score"], reverse=True)
    
    # Select top candidates - optimized selection
    # First try score >= 3, then top candidates if not enough
    promising_tickers = [r for r in screening_results if r["score"] >= 3]
    if len(promising_tickers) < 5:
        promising_tickers = [r for r in screening_results if r["score"] >= 2][:12]
    if len(promising_tickers) > 10:  # Cap at 10 for optimal speed/accuracy balance
        promising_tickers = promising_tickers[:10]
    
    print(f"📊 Tarama tamamlandı: {len(screening_results)} hisse tarandı", flush=True)
    print(f"🎯 Detaylı analiz için seçilen: {len(promising_tickers)} hisse", flush=True)
    send_telegram_message_sync(f"📊 Tarama tamamlandı: {len(screening_results)} hisse tarandı")
    send_telegram_message_sync(f"🎯 Detaylı analiz için seçilen: {len(promising_tickers)} hisse")
    
    # Show screening results
    print("\n📋 Hızlı Tarama Sonuçları(İlk 10):", flush=True)
    msg = "📋 Hızlı Tarama Sonuçları(İlk 10):\n"
    for result in screening_results[:10]:  # Show top 10
        ticker = result["ticker"]
        score = result["score"]
        rsi = result["rsi"]
        momentum = result["momentum_5d"]
        volume_ratio = result["volume_ratio"]
        
        status = "🎯" if result in promising_tickers else "📊"
        line = f"{status} {ticker.replace('.IS', '')}: Skor={score}/4, RSI={rsi:.1f}, Momentum={momentum:+.1f}%, Hacim={volume_ratio:.1f}x"
        print(line, flush=True)
        msg += f"{line}\n"
    if SEND_ADVANCED:
        send_telegram_message_sync(msg)
    
    # Stage 2: Detailed analysis on promising tickers with live progress
    print(f"\n🔬 Detaylı analiz başlatılıyor...", flush=True)
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
            gc.collect()
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
    
    msg = f"📊 <b>Borsa Tahmin Raporu</b>\n"
    msg += f"🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"
    
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
            
        msg += f"{emoji} <b>{ticker.replace('.IS', '')}</b>\n"
        msg += f"💰 Fiyat: {price:.2f} TL\n"
        msg += f"📊 Tavan Olasılığı: %{prob:.1f}\n"
        msg += f"🤖 LSTM: %{result['lstm_prob'] * 100:.1f} | XGB: %{result['xgb_prob'] * 100:.1f}\n"
        
        # Add screening info if available
        if 'screening_score' in result:
            score = result['screening_score']
            volume_ratio = result.get('volume_ratio', 0)
            momentum = result.get('momentum_5d', 0)
            msg += f"🔍 Tarama: {score}/4 | Hacim: {volume_ratio:.1f}x | Momentum: {momentum:+.1f}%\n"
        
        msg += "\n"
    
    return msg

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
    
    print("🚀 Çoklu hisse analizi başlatılıyor...", flush=True)
    print(f"🚀 Hedef: %{target * 100:.1f} artış", flush=True)
    print(f"📋 Analiz edilecek hisseler: {', '.join([t.replace('.IS', '') for t in TICKERS])}", flush=True)
    send_telegram_message_sync("🚀 Çoklu hisse analizi başlatılıyor...")
    send_telegram_message_sync(f"🚀 Hedef: %{target * 100:.1f} artış")
    send_telegram_message_sync(f"📋 {len(TICKERS)} hisse analiz edilecek")
    # Analiz yap
    results = analyze_multiple_tickers(TICKERS)
    
    # Sonuçları göster
    if results:
        print(f"\n✅ {len(results)} hisse başarıyla analiz edildi!", flush=True)
        
        # LLM ile analiz et ve öneriler al
        print("\n🤖 LLM ile analiz ve öneriler hazırlanıyor...", flush=True)
        send_telegram_message_sync("🤖 LLM ile analiz ve öneriler hazırlanıyor...")
        
        llm_analysis = analyze_with_llm(results)
        
        if llm_analysis:
            print("\n" + "="*50, flush=True)
            print("LLM ANALİZ VE ÖNERİLER:", flush=True)
            print("="*50, flush=True)
            print(llm_analysis, flush=True)
            
            # LLM analizini parçalara bölerek Telegram'a gönder
            header = f"🤖 <b>AI Analiz ve Yatırım Önerileri</b>\n🕐 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n"
            footer = f"\n📊 <i>Bu analiz {len(results)} hissenin teknik verilerine dayanmaktadır.</i>\n⚠️ <i>Bu bir yatırım tavsiyesi değil, sadece teknik analiz yorumudur.</i>"
            
            paragraphs = llm_analysis.split('---')
            
            # Send all messages
            for i, message in enumerate(paragraphs):
                print(f"📤 LLM mesajı {i+1}/{len(paragraphs)} gönderiliyor...", flush=True)
                send_telegram_message_sync(message + footer if i == len(paragraphs) - 1 else message)
            
            if SEND_ADVANCED:
                # İsteğe bağlı: Ham verileri de gönder
                print("\n📊 Ham analiz verileri de gönderiliyor...", flush=True)
                raw_data_message = format_telegram_message(results)
                raw_data_message = f"📊 <b>Ham Teknik Analiz Verileri</b>\n\n{raw_data_message}"
                send_telegram_message_sync(raw_data_message)
            
        else:
            if SEND_ADVANCED:
                print("❌ LLM analizi başarısız, ham veriler gönderiliyor...", flush=True)
                # Fallback to original telegram message
                telegram_message = format_telegram_message(results)
                send_telegram_message_sync(telegram_message)
            print("❌ LLM analizi başarısız...", flush=True)
            
            send_telegram_message_sync("❌ LLM analizi başarısız!")
            send_telegram_message_sync("❌ Ayarlardan dolayı ham veriler gönderilmiyor...")
        requests.get("https://uptime.betterstack.com/api/v1/heartbeat/B6JPnEGKx41uRTrWCfRZoJ5i")
    else:
        print("❌ Hiçbir hisse analiz edilemedi!", flush=True)
        send_telegram_message_sync("❌ Hiçbir hisse analiz edilemedi!")
    
    # End time measurement and display total process time
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    time_message = f"⏱️ Toplam işlem süresi: {minutes:02d}:{seconds:02d}"
    print(f"\n{time_message}", flush=True)
    send_telegram_message_sync(time_message)
