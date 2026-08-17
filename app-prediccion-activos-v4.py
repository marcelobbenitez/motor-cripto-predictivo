import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Crypto & Equity Predictive Assets Engine - v6",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado CSS para modo oscuro y pulido profesional
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        color: #8a9bb4;
    }
    .stTabs [aria-selected="true"] {
        color: #00ffcc !important;
        border-bottom-color: #00ffcc !important;
    }
    .metric-card {
        background-color: #1a1f2c;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #2d3748;
        margin-bottom: 10px;
    }
    .metric-title {
        color: #a0aec0;
        font-size: 14px;
        font-weight: 500;
    }
    .metric-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
        margin-top: 5px;
    }
    .metric-delta {
        font-size: 14px;
        font-weight: 600;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# CONFIGURACIÓN DE ACTIVOS Y PERFILES DE VOLATILIDAD (Grounded 2026)
# -------------------------------------------------------------------
base_prices = {
    # Criptomonedas
    "BTC": 60000.0, "ETH": 3000.0, "SOL": 140.0, "ADA": 0.45, "XRP": 0.55, "DOGE": 0.15,
    # Acciones de EE.UU.
    "AAPL": 175.0, "TSLA": 180.0, "MSFT": 420.0, "NVDA": 850.0, "AMZN": 180.0, "SPY": 500.0, "QQQ": 440.0,
    # ADRs de Argentina (en USD)
    "GGAL": 28.0, "YPF": 20.0, "BMA": 50.0, "PAM": 45.0, "CEPU": 9.0, "TGS": 15.0,
    # Materias Primas / Otros
    "GOLD": 2000.0, "OIL": 75.0
}

# Perfil de Volatilidad Anualizada (Bitcoin madurando a ~38% en 2026)
vols = {
    "BTC": 0.38, "ETH": 0.45, "SOL": 0.55, "ADA": 0.60, "XRP": 0.50, "DOGE": 0.85,
    "AAPL": 0.20, "TSLA": 0.40, "MSFT": 0.18, "NVDA": 0.45, "AMZN": 0.22, "SPY": 0.15, "QQQ": 0.18,
    "GGAL": 0.48, "YPF": 0.50, "BMA": 0.46, "PAM": 0.42, "CEPU": 0.44, "TGS": 0.40,
    "GOLD": 0.12, "OIL": 0.30
}

# Capitalización bursátil simulada en Millones USD para cálculo de Turnover Ratio
mcaps = {
    "BTC": 1500000, "ETH": 400000, "SOL": 65000, "ADA": 15000, "XRP": 30000, "DOGE": 25000,
    "AAPL": 2800000, "TSLA": 600000, "MSFT": 3100000, "NVDA": 2200000, "AMZN": 1800000, "SPY": 500000, "QQQ": 350000,
    "GGAL": 4200, "YPF": 8000, "BMA": 3200, "PAM": 3500, "CEPU": 1400, "TGS": 1800,
    "GOLD": 15000000, "OIL": 2500000
}

asset_names = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "ADA": "Cardano", "XRP": "Ripple", "DOGE": "Dogecoin",
    "AAPL": "Apple Inc.", "TSLA": "Tesla Inc.", "MSFT": "Microsoft Corp.", "NVDA": "NVIDIA Corp.", "AMZN": "Amazon.com Inc.", "SPY": "S&P 500 ETF (SPY)", "QQQ": "Nasdaq 100 ETF (QQQ)",
    "GGAL": "Grupo Fin. Galicia S.A.", "YPF": "YPF Sociedad Anónima", "BMA": "Banco Macro S.A.", "PAM": "Pampa Energía S.A.", "CEPU": "Central Puerto S.A.", "TGS": "Transportadora Gas del Sur S.A.",
    "GOLD": "Oro de Refugio", "OIL": "Petróleo Crudo WTI"
}

# -------------------------------------------------------------------
# OBTENCIÓN DE DATOS EN TIEMPO REAL (APIs de Binance y Yahoo Finance)
# -------------------------------------------------------------------
def get_binance_live_prices():
    """Obtiene precios en tiempo real para criptomonedas desde Binance"""
    pairs = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "DOGE": "DOGEUSDT"}
    prices = {}
    try:
        res = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=2)
        res.raise_for_status()
        ticker_data = res.json()
        ticker_map = {item['symbol']: float(item['price']) for item in ticker_data if item['symbol'] in pairs.values()}
        
        for name, pair in pairs.items():
            prices[name] = ticker_map.get(pair, None)
        prices["Status"] = "Online (API de Binance)"
    except Exception:
        # Fallback offline
        prices = {
            "BTC": 63450.25, "ETH": 3125.80, "SOL": 142.15, "DOGE": 0.1425,
            "Status": "Simulado (Fallo de conexión Binance)"
        }
    return prices

def get_yahoo_live_prices():
    """Obtiene precios en tiempo real para acciones de EE.UU. y ADRs Argentinos desde Yahoo Finance"""
    tickers = ["AAPL", "TSLA", "NVDA", "SPY", "GGAL", "YPF", "BMA", "PAM"]
    prices = {t: {"price": None, "pct": 0.0} for t in tickers}
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={','.join(tickers)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=2.5)
        res.raise_for_status()
        data = res.json()
        
        quote_list = data.get('quoteResponse', {}).get('result', [])
        for quote in quote_list:
            t = quote.get('symbol')
            if t in prices:
                prices[t]["price"] = quote.get('regularMarketPrice')
                prices[t]["pct"] = quote.get('regularMarketChangePercent', 0.0)
        prices["Status"] = "Online (API de Yahoo Finance)"
    except Exception:
        # Fallback offline
        simulated_vals = {
            "AAPL": {"price": 178.45, "pct": 1.25},
            "TSLA": {"price": 182.10, "pct": -2.40},
            "NVDA": {"price": 862.30, "pct": 4.15},
            "SPY": {"price": 508.40, "pct": 0.45},
            "GGAL": {"price": 29.50, "pct": 3.80},
            "YPF": {"price": 21.20, "pct": -0.85},
            "BMA": {"price": 52.40, "pct": 1.50},
            "PAM": {"price": 46.80, "pct": 2.10}
        }
        prices.update(simulated_vals)
        prices["Status"] = "Simulado (Fallo de conexión Yahoo Finance)"
    return prices

# -------------------------------------------------------------------
# GENERADOR DE DATOS HISTÓRICOS DE VELAS (SINTÉTICO COMPLETO)
# -------------------------------------------------------------------
@st.cache_data
def generate_historical_data(asset, days=180, shock_type="Ninguno (Mercado Normal)"):
    np.random.seed(42)
    base_date = datetime.now() - timedelta(days=days)
    dates = [base_date + timedelta(days=i) for i in range(days)]
    
    # 1. Liquidez de la Fed (TGA y RRP en Miles de Millones $)
    rrp = []
    tga = []
    curr_rrp = 500.0
    curr_tga = 750.0
    for i in range(days):
        if shock_type == "Drenaje de Liquidez Fed (Hawkish Shift)" and i >= days - 45:
            curr_rrp += np.random.normal(6.0, 9.0)
            curr_tga += np.random.normal(5.0, 8.0)
        else:
            curr_rrp += np.random.normal(-0.5, 5.0)
            curr_tga += np.random.normal(-0.3, 6.0)
        rrp.append(max(10.0, curr_rrp))
        tga.append(max(10.0, curr_tga))
    
    df_macro = pd.DataFrame({
        "Date": dates,
        "Fed_RRP": rrp,
        "Fed_TGA": tga
    })
    df_macro["Net_Liquidity_Index"] = 1500.0 - (df_macro["Fed_RRP"] + df_macro["Fed_TGA"])
    
    # 2. Generación de precios correlacionados específicamente para el activo
    base_p = base_prices.get(asset, 100.0)
    vol_p = vols.get(asset, 0.25)
    
    if shock_type != "Ninguno (Mercado Normal)":
        vol_p *= 1.45 # Aumento de volatilidad por estrés
        
    price_series = []
    opens = []
    highs = []
    lows = []
    volumes = []
    
    curr_price = base_p
    
    for i in range(days):
        liq_factor = (df_macro["Net_Liquidity_Index"].iloc[i] - df_macro["Net_Liquidity_Index"].mean()) / df_macro["Net_Liquidity_Index"].std()
        market_shock = np.random.normal(0, 1) + 0.1 * liq_factor
        
        # Rendimiento diario
        asset_ret = vol_p / np.sqrt(365) * market_shock + np.random.normal(0.0003, 0.005)
        
        # Shocks
        if i >= days - 45:
            if shock_type == "Drenaje de Liquidez Fed (Hawkish Shift)":
                asset_ret -= 0.005 if asset in ["BTC", "ETH", "SOL", "DOGE", "QQQ", "TSLA", "GGAL", "YPF"] else 0.001
            elif shock_type == "Pánico Social Extremo (FUD Masivo)":
                if asset in ["DOGE", "SOL", "ADA", "XRP", "ETH"]:
                    asset_ret -= 0.015
                elif asset in ["BTC", "GGAL", "YPF"]:
                    asset_ret -= 0.005
            elif shock_type == "Choque Geopolítico (Petróleo y Oro)":
                if asset == "OIL":
                    asset_ret += 0.02
                elif asset == "GOLD":
                    asset_ret += 0.012
                else:
                    asset_ret -= 0.004 # Caída generalizada de la renta variable
                    
        if i >= days - 15:
            if shock_type == "Capitulación Global ETF (Crypto Flash Crash)":
                asset_ret -= 0.03 if asset in ["BTC", "ETH", "SOL", "DOGE", "ADA", "XRP", "GGAL", "YPF", "TSLA"] else 0.008
                
        curr_price *= (1 + asset_ret)
        price_series.append(curr_price)
        
        # Construir vela diaria
        ret_std = vol_p / np.sqrt(365)
        o = curr_price * (1 + np.random.normal(0, ret_std * 0.3))
        h = max(o, curr_price) * (1 + abs(np.random.normal(0, ret_std * 0.2)))
        l = min(o, curr_price) * (1 - abs(np.random.normal(0, ret_std * 0.2)))
        
        opens.append(o)
        highs.append(h)
        lows.append(l)
        
        # Volúmenes en millones
        base_v = mcaps.get(asset, 1000) * 0.01 # volumen estimado
        vol_multiplier = 1.0
        if shock_type == "Capitulación Global ETF (Crypto Flash Crash)" and i >= days - 15:
            vol_multiplier = 3.5
        elif shock_type == "Pánico Social Extremo (FUD Masivo)" and i >= days - 45:
            vol_multiplier = 2.0
            
        v = base_v * (1 + np.random.normal(0.2, 0.4)) * (1 + abs(liq_factor) * 0.5) * vol_multiplier
        volumes.append(max(0.1, v))
        
    df_all = pd.DataFrame({"Date": dates})
    df_all["Fed_RRP"] = df_macro["Fed_RRP"]
    df_all["Fed_TGA"] = df_macro["Fed_TGA"]
    df_all["Net_Liquidity_Index"] = df_macro["Net_Liquidity_Index"]
    
    # Sentimiento de redes alineado
    twitter_sent = []
    tiktok_sent = []
    for i in range(days):
        market_trend = (price_series[i] - price_series[max(0, i-5)]) / price_series[max(0, i-5)] if price_series[max(0, i-5)] != 0 else 0
        tw = 5.0 + market_trend * 25.0 + np.random.normal(0, 1.2)
        tk = 5.0 + market_trend * 45.0 + np.random.normal(0, 2.5)
        
        if shock_type == "Pánico Social Extremo (FUD Masivo)" and i >= days - 45:
            tw = np.random.normal(1.8, 0.7)
            tk = np.random.normal(0.8, 0.4)
        elif shock_type == "Capitulación Global ETF (Crypto Flash Crash)" and i >= days - 15:
            tw = np.random.normal(1.2, 0.5)
            tk = np.random.normal(0.5, 0.3)
            
        twitter_sent.append(clip(tw, 0.0, 10.0))
        tiktok_sent.append(clip(tk, 0.0, 10.0))
        
    df_all["Twitter_Sentiment"] = twitter_sent
    df_all["TikTok_Sentiment"] = tiktok_sent
    df_all[f"{asset}_Open"] = opens
    df_all[f"{asset}_High"] = highs
    df_all[f"{asset}_Low"] = lows
    df_all[f"{asset}_Close"] = price_series
    df_all[f"{asset}_Volume_M"] = volumes
    
    return df_all

# -------------------------------------------------------------------
# OBTENCIÓN HISTÓRICA COMPLETA DE VELAS DESDE APIS REALES
# -------------------------------------------------------------------
@st.cache_data(ttl=120)
def fetch_binance_klines(asset, days=180, shock_type="Ninguno (Mercado Normal)"):
    """Descarga velas diarias reales de Binance"""
    pair = f"{asset}USDT"
    url = f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1d&limit={days}"
    try:
        res = requests.get(url, timeout=3)
        res.raise_for_status()
        klines = res.json()
        
        dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []
        for k in klines:
            dates.append(datetime.fromtimestamp(k[0] / 1000.0))
            opens.append(float(k[1]))
            highs.append(float(k[2]))
            lows.append(float(k[3]))
            closes.append(float(k[4]))
            volumes.append(float(k[5]) / 1e3) # Normalizar volumen
            
        df_res = pd.DataFrame({
            "Date": dates,
            f"{asset}_Open": opens,
            f"{asset}_High": highs,
            f"{asset}_Low": lows,
            f"{asset}_Close": closes,
            f"{asset}_Volume_M": volumes
        })
        
        # Inyectar factores macro y sentimientos
        df_sim = generate_historical_data(asset, days=len(dates), shock_type=shock_type)
        df_res["Fed_RRP"] = df_sim["Fed_RRP"]
        df_res["Fed_TGA"] = df_sim["Fed_TGA"]
        df_res["Net_Liquidity_Index"] = df_sim["Net_Liquidity_Index"]
        df_res["Twitter_Sentiment"] = df_sim["Twitter_Sentiment"]
        df_res["TikTok_Sentiment"] = df_sim["TikTok_Sentiment"]
        
        # Aplicar distorsión de shock si es necesario
        if shock_type != "Ninguno (Mercado Normal)":
            for i in range(len(df_res)):
                mult = 1.0
                if i >= len(df_res) - 45:
                    if shock_type == "Drenaje de Liquidez Fed (Hawkish Shift)":
                        mult = 0.85
                    elif shock_type == "Pánico Social Extremo (FUD Masivo)":
                        mult = 0.70 if asset == "DOGE" else 0.88
                    elif shock_type == "Choque Geopolítico (Petróleo y Oro)":
                        mult = 0.95
                if i >= len(df_res) - 15:
                    if shock_type == "Capitulación Global ETF (Crypto Flash Crash)":
                        mult = 0.68
                df_res.loc[i, f"{asset}_Open"] *= mult
                df_res.loc[i, f"{asset}_High"] *= mult
                df_res.loc[i, f"{asset}_Low"] *= mult
                df_res.loc[i, f"{asset}_Close"] *= mult
                df_res.loc[i, f"{asset}_Volume_M"] *= (3.5 if shock_type == "Capitulación Global ETF (Crypto Flash Crash)" and i >= len(df_res) - 15 else 1.0)
                
        return df_res, "Binance API (Histórico)"
    except Exception as e:
        df_fallback = generate_historical_data(asset, days=days, shock_type=shock_type)
        return df_fallback, f"Simulado (Fallback conexión: {str(e)[:40]}...)"

@st.cache_data(ttl=120)
def fetch_yahoo_klines(asset, days=180, shock_type="Ninguno (Mercado Normal)"):
    """Descarga velas diarias reales de Yahoo Finance"""
    # Aproximar rango
    range_str = "6mo" if days <= 180 else "1y"
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{asset}?range={range_str}&interval=1d"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        res = requests.get(url, headers=headers, timeout=4)
        res.raise_for_status()
        data = res.json()
        
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        indicators = result['indicators']['quote'][0]
        
        dates, opens, highs, lows, closes, volumes = [], [], [], [], [], []
        for idx, ts in enumerate(timestamps):
            o = indicators['open'][idx]
            h = indicators['high'][idx]
            l = indicators['low'][idx]
            c = indicators['close'][idx]
            v = indicators['volume'][idx]
            
            if o is not None and h is not None and l is not None and c is not None:
                dates.append(datetime.fromtimestamp(ts))
                opens.append(float(o))
                highs.append(float(h))
                lows.append(float(l))
                closes.append(float(c))
                volumes.append(float(v) / 1e6 if v is not None else 0.0)
                
        # Cortar rango
        dates = dates[-days:]
        opens = opens[-days:]
        highs = highs[-days:]
        lows = lows[-days:]
        closes = closes[-days:]
        volumes = volumes[-days:]
        
        df_res = pd.DataFrame({
            "Date": dates,
            f"{asset}_Open": opens,
            f"{asset}_High": highs,
            f"{asset}_Low": lows,
            f"{asset}_Close": closes,
            f"{asset}_Volume_M": volumes
        })
        
        # Inyectar factores macro y sentimientos
        df_sim = generate_historical_data(asset, days=len(dates), shock_type=shock_type)
        df_res["Fed_RRP"] = df_sim["Fed_RRP"]
        df_res["Fed_TGA"] = df_sim["Fed_TGA"]
        df_res["Net_Liquidity_Index"] = df_sim["Net_Liquidity_Index"]
        df_res["Twitter_Sentiment"] = df_sim["Twitter_Sentiment"]
        df_res["TikTok_Sentiment"] = df_sim["TikTok_Sentiment"]
        
        # Aplicar distorsión de shock
        if shock_type != "Ninguno (Mercado Normal)":
            for i in range(len(df_res)):
                mult = 1.0
                if i >= len(df_res) - 45:
                    if shock_type == "Drenaje de Liquidez Fed (Hawkish Shift)":
                        mult = 0.88
                    elif shock_type == "Pánico Social Extremo (FUD Masivo)":
                        mult = 0.92
                    elif shock_type == "Choque Geopolítico (Petróleo y Oro)":
                        if asset == "GOLD":
                            mult = 1.08
                        elif asset == "OIL":
                            mult = 1.15
                        else:
                            mult = 0.92
                if i >= len(df_res) - 15:
                    if shock_type == "Capitulación Global ETF (Crypto Flash Crash)":
                        mult = 0.82
                df_res.loc[i, f"{asset}_Open"] *= mult
                df_res.loc[i, f"{asset}_High"] *= mult
                df_res.loc[i, f"{asset}_Low"] *= mult
                df_res.loc[i, f"{asset}_Close"] *= mult
                df_res.loc[i, f"{asset}_Volume_M"] *= (1.8 if shock_type == "Capitulación Global ETF (Crypto Flash Crash)" and i >= len(df_res) - 15 else 1.0)
                
        return df_res, "Yahoo Finance (Histórico)"
    except Exception as e:
        df_fallback = generate_historical_data(asset, days=days, shock_type=shock_type)
        return df_fallback, f"Simulado (Fallback conexión: {str(e)[:40]}...)"

def clip(val, min_val, max_val):
    return max(min_val, min(val, max_val))

# -------------------------------------------------------------------
# PANEL LATERAL Y CONFIGURACIÓN DEL ENTORNO
# -------------------------------------------------------------------
st.sidebar.title("🛠️ Configuración de Motor")

# Selector de Origen de Datos
st.sidebar.subheader("🔌 Origen de Datos")
data_mode = st.sidebar.radio("Modo de Origen de Datos:", ["Simulado (Offline)", "Conexión Real-Time APIs (Online)"])

# Selector de Shock de Mercado (Stress Testing)
st.sidebar.subheader("🚨 Simulación de Shocks (Stress Test)")
shock_type = st.sidebar.selectbox("Selecciona un Escenario de Shock:", [
    "Ninguno (Mercado Normal)",
    "Drenaje de Liquidez Fed (Hawkish Shift)",
    "Pánico Social Extremo (FUD Masivo)",
    "Choque Geopolítico (Petróleo y Oro)",
    "Capitulación Global ETF (Crypto Flash Crash)"
])

# Selector Tipo de Activo
st.sidebar.subheader("📁 Categoría de Mercado")
asset_class = st.sidebar.selectbox("Selecciona la Categoría:", [
    "Criptomonedas (Binance)",
    "Renta Variable EE.UU. (Yahoo Finance)",
    "Acciones Argentinas (Yahoo Finance)",
    "Materias Primas & Refugios (Yahoo Finance)"
])

# Mapeo de categoría a opciones disponibles
if asset_class == "Criptomonedas (Binance)":
    assets_avail = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE"]
elif asset_class == "Renta Variable EE.UU. (Yahoo Finance)":
    assets_avail = ["AAPL", "TSLA", "MSFT", "NVDA", "AMZN", "SPY", "QQQ"]
elif asset_class == "Acciones Argentinas (Yahoo Finance)":
    assets_avail = ["GGAL", "YPF", "BMA", "PAM", "CEPU", "TGS"]
else:
    assets_avail = ["GOLD", "OIL"]

selected_asset = st.sidebar.selectbox("Activo a Analizar:", assets_avail, format_func=lambda x: f"{x} - {asset_names.get(x, '')}")

# Carga Inteligente de datos según Origen y Categoría
data_source_label = "Simulado (Offline)"
if data_mode == "Conexión Real-Time APIs (Online)":
    if asset_class == "Criptomonedas (Binance)":
        df, data_source_label = fetch_binance_klines(selected_asset, days=180, shock_type=shock_type)
    else:
        df, data_source_label = fetch_yahoo_klines(selected_asset, days=180, shock_type=shock_type)
else:
    df = generate_historical_data(selected_asset, days=180, shock_type=shock_type)
    data_source_label = "Simulado (Offline)"

# Configuración del backtester
st.sidebar.subheader("📈 Backtesting de Portafolio")
initial_capital = st.sidebar.number_input("Capital Inicial ($)", value=10000, step=1000)
slippage_fee = st.sidebar.slider("Comisiones & Slippage por Trade (%)", 0.0, 1.0, 0.1, step=0.05) / 100.0

st.sidebar.markdown("""
---
**Acerca del Motor de Predicción:**
Esta aplicación implementa un sistema híbrido que integra análisis técnico de confluencia con modelos de procesamiento de lenguaje natural y factores de liquidez macro. Basado en el reporte de 2026.
""")

# -------------------------------------------------------------------
# MOTOR DE CONFLUENCIA TÉCNICA (Velas, Soporte/Resistencia, Volumen)
# -------------------------------------------------------------------
def calculate_support_resistance(df, asset):
    close_series = df[f"{asset}_Close"]
    high_series = df[f"{asset}_High"]
    low_series = df[f"{asset}_Low"]
    support = low_series.rolling(window=20).min().iloc[-1]
    resistance = high_series.rolling(window=20).max().iloc[-1]
    return support, resistance

def detect_candlestick_patterns(df, asset):
    last_idx = -1
    o = df[f"{asset}_Open"].iloc[last_idx]
    h = df[f"{asset}_High"].iloc[last_idx]
    l = df[f"{asset}_Low"].iloc[last_idx]
    c = df[f"{asset}_Close"].iloc[last_idx]
    
    body_size = abs(c - o)
    candle_range = h - l if h - l != 0 else 0.001
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    is_bullish = c > o
    is_bearish = c < o
    
    pattern = "Neutral"
    bias = 0.0
    
    # Hammer
    if (lower_wick > 2 * body_size) and (upper_wick < 0.1 * candle_range):
        pattern = "Bullish Hammer / Pin Bar"
        bias = 0.6
    # Hanging Man / Shooting Star
    elif (upper_wick > 2 * body_size) and (lower_wick < 0.1 * candle_range):
        pattern = "Bearish Shooting Star / Hanging Man"
        bias = -0.6
        
    prev_o, prev_c = df[f"{asset}_Open"].iloc[last_idx-1], df[f"{asset}_Close"].iloc[last_idx-1]
    
    # Engulfing
    if (is_bullish) and (prev_c < prev_o) and (c > prev_o) and (o < prev_c):
        pattern = "Bullish Engulfing"
        bias = 0.8
    elif (is_bearish) and (prev_c > prev_o) and (c < prev_o) and (o > prev_c):
        pattern = "Bearish Engulfing"
        bias = -0.8
        
    return pattern, bias

def check_technical_confluence(df, asset):
    pattern, bias = detect_candlestick_patterns(df, asset)
    sup, res = calculate_support_resistance(df, asset)
    last_close = df[f"{asset}_Close"].iloc[-1]
    last_vol = df[f"{asset}_Volume_M"].iloc[-1]
    avg_vol = df[f"{asset}_Volume_M"].rolling(window=20).mean().iloc[-1]
    
    confluence_triggered = False
    confluence_score = 0.0
    
    near_support = abs(last_close - sup) / last_close < 0.02
    near_resistance = abs(last_close - res) / last_close < 0.02
    vol_expansion = last_vol > 1.2 * avg_vol
    
    if bias > 0 and near_support and vol_expansion:
        confluence_triggered = True
        confluence_score = bias * 1.2
    elif bias < 0 and near_resistance and vol_expansion:
        confluence_triggered = True
        confluence_score = bias * 1.2
    else:
        confluence_score = bias * 0.4
        
    confluence_score = clip(confluence_score, -1.0, 1.0)
    return pattern, confluence_triggered, confluence_score

# -------------------------------------------------------------------
# METRICAS DE CAPITALIZACIÓN Y GIRO (TURNOVER)
# -------------------------------------------------------------------
def get_turnover_ratio(df, asset):
    last_vol = df[f"{asset}_Volume_M"].iloc[-1]
    mcap = mcaps.get(asset, 1000)
    ratio = (last_vol / mcap) * 100.0
    
    if ratio < 0.5:
        desc = "Giro Bajo (Low Turnover) - Posible falta de confirmación de tendencia o spreads amplios."
        status = "Warning"
    elif ratio <= 3.0:
        desc = "Giro Normal - Interés institucional estándar y flujos de liquidez saludables."
        status = "Healthy"
    elif ratio <= 12.0:
        desc = "Giro Activo (Active Repricing) - Fuerte volumen que respalda y confirma rotación de precio."
        status = "Strong"
    else:
        desc = "Mercado Muy Caliente (Speculative Hot Spot) - Cuidado con sobrecompra, pánico minorista o manipulación."
        status = "Hot"
        
    return ratio, desc, status

# -------------------------------------------------------------------
# MOTOR DE SENTIMIENTO MULTIMODAL Y ML (ESTILO STANFORD)
# -------------------------------------------------------------------
def calculate_multimodal_sentiment(df, asset):
    last_tw = df["Twitter_Sentiment"].iloc[-1]
    last_tk = df["TikTok_Sentiment"].iloc[-1]
    
    if asset in ["DOGE", "SOL", "ADA", "XRP"]:
        # TikTok domina en altcoins de alta especulación minorista
        combined_score = (0.35 * last_tw + 0.65 * last_tk)
    elif asset in ["BTC", "GOLD", "SPY", "QQQ", "MSFT", "AAPL"]:
        # Twitter y dinámica institucional de mediano-largo plazo domina
        combined_score = (0.75 * last_tw + 0.25 * last_tk)
    else:
        combined_score = (0.55 * last_tw + 0.45 * last_tk)
        
    return combined_score / 10.0

def run_stanford_predictive_model(df, asset):
    df_ml = df.copy()
    df_ml["Target"] = (df_ml[f"{asset}_Close"].shift(-1) > df_ml[f"{asset}_Close"]).astype(int)
    
    features = []
    for lag in range(3):
        df_ml[f"Twitter_Lag_{lag}"] = df_ml["Twitter_Sentiment"].shift(lag)
        df_ml[f"TikTok_Lag_{lag}"] = df_ml["TikTok_Sentiment"].shift(lag)
        df_ml[f"Net_Liq_Lag_{lag}"] = df_ml["Net_Liquidity_Index"].shift(lag)
        features.extend([f"Twitter_Lag_{lag}", f"TikTok_Lag_{lag}", f"Net_Liq_Lag_{lag}"])
        
    df_ml = df_ml.dropna()
    
    if len(df_ml) < 20:
        return 0.5, 0.0
        
    X = df_ml[features]
    y = df_ml["Target"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(C=0.9, penalty='l2', solver='liblinear')
    model.fit(X_scaled[:-1], y.iloc[:-1])
    
    last_features = X_scaled[-1].reshape(1, -1)
    prob_up = model.predict_proba(last_features)[0][1]
    
    return prob_up, model.coef_[0][0]

# -------------------------------------------------------------------
# MOTOR DE TRADING COMBINADO (MACD, ETS & BACKTESTING)
# -------------------------------------------------------------------
def calculate_signals_and_backtest(df, asset, init_cap, fee):
    df_bt = df.copy()
    
    k_short = df_bt[f"{asset}_Close"].ewm(span=12, adjust=False).mean()
    d_short = df_bt[f"{asset}_Close"].ewm(span=26, adjust=False).mean()
    macd_short = k_short - d_short
    signal_short = macd_short.ewm(span=9, adjust=False).mean()
    
    k_long = df_bt[f"{asset}_Close"].ewm(span=19, adjust=False).mean()
    d_long = df_bt[f"{asset}_Close"].ewm(span=39, adjust=False).mean()
    macd_long = k_long - d_long
    signal_long = macd_long.ewm(span=9, adjust=False).mean()
    
    df_bt["MACD_Short"] = macd_short
    df_bt["MACD_Signal_Short"] = signal_short
    df_bt["MACD_Long"] = macd_long
    df_bt["MACD_Signal_Long"] = signal_long
    
    signals = []
    positions = []
    cash = init_cap
    holdings = 0.0
    portfolio_value = []
    
    for i in range(len(df_bt)):
        if i < 30:
            signals.append(0.0)
            portfolio_value.append(init_cap)
            positions.append(0)
            continue
            
        sent_combined = calculate_multimodal_sentiment(df_bt.iloc[:i+1], asset) - 0.5
        _, _, tech_score = check_technical_confluence(df_bt.iloc[:i+1], asset)
        tech_score = tech_score * 0.5
        
        liq_index = df_bt["Net_Liquidity_Index"].iloc[i]
        prev_liq = df_bt["Net_Liquidity_Index"].iloc[max(0, i-5)]
        liq_signal = 0.5 if liq_index > prev_liq else -0.5
        
        raw_it = sent_combined * 0.4 + tech_score * 0.4 + liq_signal * 0.2
        it_signal = clip(raw_it * 2.0, -1.0, 1.0)
        signals.append(it_signal)
        
        bullish_macd = (macd_short.iloc[i] > signal_short.iloc[i]) and (macd_long.iloc[i] > signal_long.iloc[i])
        bearish_macd = (macd_short.iloc[i] < signal_short.iloc[i]) and (macd_long.iloc[i] < signal_long.iloc[i])
        
        price = df_bt[f"{asset}_Close"].iloc[i]
        current_pos = positions[-1] if len(positions) > 0 else 0
        
        if it_signal > 0.25 and bullish_macd and current_pos == 0:
            holdings = (cash * (1.0 - fee)) / price
            cash = 0.0
            positions.append(1)
        elif (it_signal < -0.25 or bearish_macd) and current_pos == 1:
            cash = holdings * price * (1.0 - fee)
            holdings = 0.0
            positions.append(0)
        else:
            positions.append(current_pos)
            
        current_val = cash + (holdings * price)
        portfolio_value.append(current_val)
        
    df_bt["Signal_It"] = signals
    df_bt["Position"] = positions
    df_bt["Portfolio_Value"] = portfolio_value
    df_bt["BH_Value"] = (df_bt[f"{asset}_Close"] / df_bt[f"{asset}_Close"].iloc[30]) * init_cap
    df_bt.loc[:30, "BH_Value"] = init_cap
    
    return df_bt

# Ejecutar el Backtester
df_results = calculate_signals_and_backtest(df, selected_asset, initial_capital, slippage_fee)

# -------------------------------------------------------------------
# RENDERIZADO DE LA INTERFAZ DE USUARIO (UI)
# -------------------------------------------------------------------
st.title("⚡ Crypto & Equity Assets Predictive Engine - v6")
st.markdown(f"### Modelador Predictivo y Plataforma de Stress Testing Multiactivos | Origen: <span style='color:#00ffcc;'>{data_source_label}</span>", unsafe_allow_html=True)

# Alerta de Stress Testing
if shock_type != "Ninguno (Mercado Normal)":
    st.markdown(f"""
    <div style="background-color: #3b0d11; border: 2px solid #ff3333; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="color: #ff3333; margin: 0; font-weight: 700;">🚨 ENTORNO DE STRESS TESTING ACTIVO: {shock_type.upper()}</h4>
        <p style="color: #f7a8a8; margin: 5px 0 0 0; font-size: 14px;">
            Simulando perturbaciones graves en el sistema. Las volatilidades aumentaron un 45% y las series de datos reflejan correcciones severas.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Métricas de Encabezado
col1, col2, col3, col4 = st.columns(4)

with col1:
    last_price = df_results[f"{selected_asset}_Close"].iloc[-1]
    prev_price = df_results[f"{selected_asset}_Close"].iloc[-2]
    pct_change = ((last_price - prev_price) / prev_price) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Precio de Cierre ({selected_asset})</div>
        <div class="metric-value">${last_price:,.2f}</div>
        <div class="metric-delta" style="color: {'#00ffcc' if pct_change >= 0 else '#ff4d4d'};">
            {'▲' if pct_change >= 0 else '▼'} {pct_change:.2f}% (24h)
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    turnover, turn_desc, turn_status = get_turnover_ratio(df_results, selected_asset)
    color_status = {"Warning": "#ff9900", "Healthy": "#00ffcc", "Strong": "#00bbff", "Hot": "#ff3333"}[turn_status]
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Turnover Ratio (Vol/Mcap)</div>
        <div class="metric-value">{turnover:.2f}%</div>
        <div class="metric-delta" style="color: {color_status};">
            ● {turn_status}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    pattern_name, _, conf_score = check_technical_confluence(df_results, selected_asset)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Patrón de Velas Detectado</div>
        <div class="metric-value" style="font-size: 18px; padding-top: 5px;">{pattern_name}</div>
        <div class="metric-delta" style="color: {'#00ffcc' if conf_score > 0 else ('#ff4d4d' if conf_score < 0 else '#8a9bb4')};">
            Score Confluencia: {conf_score:.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    prob_up, feature_imp = run_stanford_predictive_model(df_results, selected_asset)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Probabilidad Stanford (+1D)</div>
        <div class="metric-value">{prob_up*100:.1f}% de Subida</div>
        <div class="metric-delta" style="color: #00ffcc;">
            Modelo Regularizado L2 (C=0.9)
        </div>
    </div>
    """, unsafe_allow_html=True)

# TABS PRINCIPALES DE LA APP
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Panel de Control de Activos", 
    "🔮 Centro Predictivo (Holt-Winters & ML)", 
    "🗣️ Sentimiento Multimodal & Liquidez Fed", 
    "🛡️ Backtesting & Gestión de Riesgo"
])

# -------------------------------------------------------------------
# TAB 1: PANEL DE CONTROL DE ACTIVOS
# -------------------------------------------------------------------
with tab1:
    st.subheader("Cuadro de Precios Globales en Tiempo Real (APIs Conectadas)")
    
    # Obtener precios en tiempo real para crypto y acciones
    binance_live = get_binance_live_prices()
    yahoo_live = get_yahoo_live_prices()
    
    col_live1, col_live2, col_live3, col_live4 = st.columns(4)
    with col_live1:
        st.markdown(f"""
        <div style='background-color:#141722; padding:15px; border-radius:8px; border: 1px solid #2d3748;'>
            <span style='color:#a0aec0; font-size:12px;'>🔥 CRIPTOS (Binance)</span>
            <div style='margin-top: 5px;'>
                <b style='color:#00ffcc;'>BTC:</b> ${binance_live['BTC']:,.2f}<br>
                <b style='color:#00ffcc;'>ETH:</b> ${binance_live['ETH']:,.2f}<br>
                <b style='color:#00ffcc;'>SOL:</b> ${binance_live['SOL']:,.2f}<br>
                <b style='color:#00ffcc;'>DOGE:</b> ${binance_live['DOGE']:,.4f}
            </div>
            <span style='color:#8a9bb4; font-size:10px; display:block; margin-top:5px;'>{binance_live['Status']}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_live2:
        st.markdown(f"""
        <div style='background-color:#141722; padding:15px; border-radius:8px; border: 1px solid #2d3748;'>
            <span style='color:#a0aec0; font-size:12px;'>🇺🇸 ACCIONES EE.UU. (YFinance)</span>
            <div style='margin-top: 5px;'>
                <b style='color:#00bbff;'>AAPL:</b> ${yahoo_live['AAPL']['price']:,.2f} ({yahoo_live['AAPL']['pct']:+.2f}%)<br>
                <b style='color:#00bbff;'>TSLA:</b> ${yahoo_live['TSLA']['price']:,.2f} ({yahoo_live['TSLA']['pct']:+.2f}%)<br>
                <b style='color:#00bbff;'>NVDA:</b> ${yahoo_live['NVDA']['price']:,.2f} ({yahoo_live['NVDA']['pct']:+.2f}%)<br>
                <b style='color:#00bbff;'>SPY:</b> ${yahoo_live['SPY']['price']:,.2f} ({yahoo_live['SPY']['pct']:+.2f}%)
            </div>
            <span style='color:#8a9bb4; font-size:10px; display:block; margin-top:5px;'>{yahoo_live['Status']}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_live3:
        st.markdown(f"""
        <div style='background-color:#141722; padding:15px; border-radius:8px; border: 1px solid #2d3748;'>
            <span style='color:#a0aec0; font-size:12px;'>🇦🇷 ADRs ARGENTINA (YFinance)</span>
            <div style='margin-top: 5px;'>
                <b style='color:#ff9900;'>GGAL:</b> ${yahoo_live['GGAL']['price']:,.2f} ({yahoo_live['GGAL']['pct']:+.2f}%)<br>
                <b style='color:#ff9900;'>YPF:</b> ${yahoo_live['YPF']['price']:,.2f} ({yahoo_live['YPF']['pct']:+.2f}%)<br>
                <b style='color:#ff9900;'>BMA:</b> ${yahoo_live['BMA']['price']:,.2f} ({yahoo_live['BMA']['pct']:+.2f}%)<br>
                <b style='color:#ff9900;'>PAM:</b> ${yahoo_live['PAM']['price']:,.2f} ({yahoo_live['PAM']['pct']:+.2f}%)
            </div>
            <span style='color:#8a9bb4; font-size:10px; display:block; margin-top:5px;'>{yahoo_live['Status']}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with col_live4:
        st.markdown(f"""
        <div style='background-color:#141722; padding:15px; border-radius:8px; border: 1px solid #2d3748;'>
            <span style='color:#a0aec0; font-size:12px;'>⚡ ACTIVO SELECCIONADO</span>
            <h3 style='color:#ffffff; margin:5px 0;'>{selected_asset}</h3>
            <span style='color:#00ffcc; font-size:13px; font-weight:700;'>{asset_names.get(selected_asset, '')}</span>
            <span style='color:#8a9bb4; font-size:11px; display:block; margin-top:5px;'>Categoría: {asset_class}</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Análisis de Tendencias y Velas Japonesas")
    
    # Gráfico de Velas de Plotly
    fig_candles = go.Figure(data=[go.Candlestick(
        x=df_results["Date"],
        open=df_results[f"{selected_asset}_Open"],
        high=df_results[f"{selected_asset}_High"],
        low=df_results[f"{selected_asset}_Low"],
        close=df_results[f"{selected_asset}_Close"],
        name=selected_asset
    )])
    
    sup, res = calculate_support_resistance(df_results, selected_asset)
    fig_candles.add_hline(y=sup, line_dash="dash", line_color="#00ffcc", annotation_text=f"Soporte Local: ${sup:,.2f}")
    fig_candles.add_hline(y=res, line_dash="dash", line_color="#ff4d4d", annotation_text=f"Resistencia Local: ${res:,.2f}")
    
    fig_candles.update_layout(
        title=f"Evolución y Velas Japonesas de {selected_asset} | Modo: {data_source_label}",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=450
    )
    st.plotly_chart(fig_candles, use_container_width=True)
    
    st.info(f"💡 **Filtro de Giro Técnico**: {turn_desc}")

# -------------------------------------------------------------------
# TAB 2: CENTRO PREDICTIVO
# -------------------------------------------------------------------
with tab2:
    st.subheader("Modelado Cuantitativo y Proyecciones Predictivas")
    
    # Suavizado Exponencial Holt-Winters
    close_data = df_results[f"{selected_asset}_Close"].values
    ets_model = ExponentialSmoothing(close_data, seasonal="add", seasonal_periods=7, trend="add").fit()
    forecast_days = 15
    ets_forecast = ets_model.forecast(forecast_days)
    
    future_dates = [df_results["Date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    fig_pred = go.Figure()
    fig_pred.add_scatter(x=df_results["Date"].iloc[-30:], y=df_results[f"{selected_asset}_Close"].iloc[-30:], name="Histórico (30 D)", line_color="#8a9bb4")
    fig_pred.add_scatter(x=future_dates, y=ets_forecast, name="Pronóstico Holt-Winters (15D)", line=dict(color="#00ffcc", width=3, dash="dash"))
    
    fig_pred.update_layout(
        title=f"Ajuste y Pronóstico Holt-Winters para {selected_asset}",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        ### 🧠 Explicación del Modelo Stanford ML
        Este clasificador predictivo está modelado de acuerdo con la metodología de **Stanford University (CS229)**. En lugar de limitarse a calcular la polaridad del sentimiento de las redes, el modelo se entrena asociando directamente las características del sentimiento y el índice macro con el movimiento de precio real futuro de 24 horas (+1D de subida o bajada).
        
        *   **Probabilidad de Mañana:** {prob_up*100:.2f}% de probabilidad de cierre alcista.
        *   **Indicador de Dirección:** {"**COMPRA / ALCISTA**" if prob_up > 0.52 else "**VENTA / BAJISTA**" if prob_up < 0.48 else "**NEUTRAL / CONSOLIDACIÓN**"}
        """)
    with col_b:
        st.markdown(f"""
        ### 📉 Gestión del Perfil de Volatilidad
        *   **Volatilidad Histórica Simulada/Real de {selected_asset}:** {(vols.get(selected_asset, 0.25)*100):.1f}% anualizada.
        *   **Nota de Mercado (2026):** Los perfiles de volatilidad de Bitcoin (~38%) y Ethereum (~45%) se han estabilizado significativamente respecto de la década pasada, asemejándose cada vez más a las acciones de crecimiento del Nasdaq (como QQQ o NVDA) debido al flujo masivo de capital institucional de los ETFs al contado.
        """)

# -------------------------------------------------------------------
# TAB 3: SENTIMIENTO MULTIMODAL & MACRO LIQUIDEZ
# -------------------------------------------------------------------
with tab3:
    st.subheader("Análisis de Sentimiento de Redes y Liquidez de la Fed")
    
    fig_sent = go.Figure()
    fig_sent.add_scatter(x=df_results["Date"], y=df_results["Twitter_Sentiment"], name="Twitter/X (Señal Institucional - Texto)", line_color="#1DA1F2")
    fig_sent.add_scatter(x=df_results["Date"], y=df_results["TikTok_Sentiment"], name="TikTok (Señal Especulativa Retail - Vídeo)", line_color="#EE1D52")
    
    fig_sent.update_layout(
        title="Fusión del Sentimiento Multimodal Cross-Platform",
        template="plotly_dark",
        height=350,
        yaxis_title="Escala de Sentimiento (0-10)"
    )
    st.plotly_chart(fig_sent, use_container_width=True)
    
    fig_liq = go.Figure()
    fig_liq.add_scatter(x=df_results["Date"], y=df_results["Net_Liquidity_Index"], name="Índice de Liquidez Neta (Fed)", line_color="#00ffcc", fill="tozeroy")
    
    fig_liq.update_layout(
        title="Índice de Liquidez Neta (Inversa de RRP + Cuenta General TGA)",
        template="plotly_dark",
        height=300,
        yaxis_title="Billones de $"
    )
    st.plotly_chart(fig_liq, use_container_width=True)

# -------------------------------------------------------------------
# TAB 4: BACKTESTING & GESTIÓN DE RIESGO
# -------------------------------------------------------------------
with tab4:
    st.subheader("Backtesting: Algoritmo de Confluencia Híbrido vs. Buy & Hold")
    
    fig_bt = go.Figure()
    fig_bt.add_scatter(x=df_results["Date"], y=df_results["Portfolio_Value"], name="Estrategia Algorítmica Híbrida (Velas + Sentimiento + Macro)", line_color="#00ffcc", line_width=2.5)
    fig_bt.add_scatter(x=df_results["Date"], y=df_results["BH_Value"], name="Estrategia Pasiva (Buy & Hold)", line_color="#ff9900", line_dash="dash")
    
    fig_bt.update_layout(
        title="Crecimiento de Capital ($10K Inicial)",
        template="plotly_dark",
        height=400,
        yaxis_title="Valor de Cuenta ($)"
    )
    st.plotly_chart(fig_bt, use_container_width=True)
    
    # Cálculos de Riesgo
    final_algo = df_results["Portfolio_Value"].iloc[-1]
    final_bh = df_results["BH_Value"].iloc[-1]
    ret_algo = ((final_algo - initial_capital) / initial_capital) * 100
    ret_bh = ((final_bh - initial_capital) / initial_capital) * 100
    
    returns_algo = df_results["Portfolio_Value"].pct_change().dropna()
    returns_bh = df_results["BH_Value"].pct_change().dropna()
    
    sharpe_algo = (returns_algo.mean() / returns_algo.std()) * np.sqrt(365) if returns_algo.std() != 0 else 0
    sharpe_bh = (returns_bh.mean() / returns_bh.std()) * np.sqrt(365) if returns_bh.std() != 0 else 0
    
    vol_algo = (returns_algo.std() * np.sqrt(365)) * 100 if len(returns_algo) > 0 else 0.0
    vol_bh = (returns_bh.std() * np.sqrt(365)) * 100 if len(returns_bh) > 0 else 0.0
    
    def calc_max_dd(val_series):
        cum_max = val_series.cummax()
        dd = (val_series - cum_max) / cum_max
        return dd.min() * 100
    max_dd_algo = calc_max_dd(df_results["Portfolio_Value"])
    max_dd_bh = calc_max_dd(df_results["BH_Value"])
    
    var_95_algo = np.percentile(returns_algo, 5) * 100 if len(returns_algo) > 0 else 0.0
    var_95_bh = np.percentile(returns_bh, 5) * 100 if len(returns_bh) > 0 else 0.0
    
    def calc_sortino(ret_series):
        if len(ret_series) == 0: return 0.0
        mean_ret = ret_series.mean()
        downside = ret_series[ret_series < 0]
        if len(downside) == 0 or downside.std() == 0: return 0.0
        return (mean_ret / downside.std()) * np.sqrt(365)
    sortino_algo = calc_sortino(returns_algo)
    sortino_bh = calc_sortino(returns_bh)
    
    cum_max_algo = df_results["Portfolio_Value"].cummax()
    dd_algo_curve = (df_results["Portfolio_Value"] - cum_max_algo) / cum_max_algo * 100
    cum_max_bh = df_results["BH_Value"].cummax()
    dd_bh_curve = (df_results["BH_Value"] - cum_max_bh) / cum_max_bh * 100
    
    df_results["Drawdown_Algo"] = dd_algo_curve
    df_results["Drawdown_BH"] = dd_bh_curve
    
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Retorno Algoritmo Híbrido", f"{ret_algo:.2f}%", f"${final_algo:,.2f}")
    with col_s2:
        st.metric("Retorno Buy & Hold", f"{ret_bh:.2f}%", f"${final_bh:,.2f}")
    with col_s3:
        st.metric("Sharpe Ratio (Híbrido vs B&H)", f"{sharpe_algo:.2f}", f"B&H Sharpe: {sharpe_bh:.2f}")
        
    st.markdown("---")
    st.subheader(f"🛡️ Reporte Analítico de Riesgo | Activo: {selected_asset}")
    
    diff_ret = ret_algo - ret_bh
    diff_vol = vol_algo - vol_bh
    diff_sharpe = sharpe_algo - sharpe_bh
    diff_sortino = sortino_algo - sortino_bh
    diff_max_dd = max_dd_algo - max_dd_bh
    diff_var = var_95_algo - var_95_bh
    
    st.markdown(f"""
    <table style="width:100%; border-collapse: collapse; border: 1px solid #2d3748; background-color: #161a23; color: white;">
      <thead>
        <tr style="background-color: #2d3748; border-bottom: 2px solid #00ffcc;">
          <th style="padding: 12px; text-align: left; font-weight: 600;">Métrica de Riesgo</th>
          <th style="padding: 12px; text-align: right; font-weight: 600; color: #00ffcc;">Algoritmo Híbrido</th>
          <th style="padding: 12px; text-align: right; font-weight: 600; color: #ff9900;">Pasivo (Buy & Hold)</th>
          <th style="padding: 12px; text-align: center; font-weight: 600;">Alfa / Diferencia de Riesgo</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom: 1px solid #2d3748;">
          <td style="padding: 12px; font-weight: 500;">Retorno Total</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #00ffcc;">{ret_algo:.2f}%</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff9900;">{ret_bh:.2f}%</td>
          <td style="padding: 12px; text-align: center; font-weight: 700; color: {'#00ffcc' if diff_ret >= 0 else '#ff4d4d'};">{diff_ret:+.2f}%</td>
        </tr>
        <tr style="border-bottom: 1px solid #2d3748;">
          <td style="padding: 12px; font-weight: 500;">Volatilidad Anualizada</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #00ffcc;">{vol_algo:.2f}%</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff9900;">{vol_bh:.2f}%</td>
          <td style="padding: 12px; text-align: center; font-weight: 700; color: {'#00ffcc' if diff_vol <= 0 else '#ff4d4d'};">{diff_vol:+.2f}% {"(Menor volatilidad)" if diff_vol <= 0 else "(Mayor volatilidad)"}</td>
        </tr>
        <tr style="border-bottom: 1px solid #2d3748;">
          <td style="padding: 12px; font-weight: 500;">Sharpe Ratio</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #00ffcc;">{sharpe_algo:.2f}</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff9900;">{sharpe_bh:.2f}</td>
          <td style="padding: 12px; text-align: center; font-weight: 700; color: {'#00ffcc' if diff_sharpe >= 0 else '#ff4d4d'};">{diff_sharpe:+.2f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #2d3748;">
          <td style="padding: 12px; font-weight: 500;">Sortino Ratio</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #00ffcc;">{sortino_algo:.2f}</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff9900;">{sortino_bh:.2f}</td>
          <td style="padding: 12px; text-align: center; font-weight: 700; color: {'#00ffcc' if diff_sortino >= 0 else '#ff4d4d'};">{diff_sortino:+.2f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #2d3748;">
          <td style="padding: 12px; font-weight: 500;">Máximo Drawdown</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff4d4d;">{max_dd_algo:.2f}%</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff4d4d;">{max_dd_bh:.2f}%</td>
          <td style="padding: 12px; text-align: center; font-weight: 700; color: {'#00ffcc' if max_dd_algo > max_dd_bh else '#ff4d4d'};">{diff_max_dd:+.2f}% {"(Menor caída)" if max_dd_algo > max_dd_bh else "(Mayor caída)"}</td>
        </tr>
        <tr>
          <td style="padding: 12px; font-weight: 500;">Value at Risk (VaR Histórico 95% 1-Día)</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff4d4d;">{var_95_algo:.2f}%</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff9900;">{var_95_bh:.2f}%</td>
          <td style="padding: 12px; text-align: center; font-weight: 700; color: {'#00ffcc' if var_95_algo > var_95_bh else '#ff4d4d'};">{diff_var:+.2f}% {"(Menor pérdida potencial diaria)" if var_95_algo > var_95_bh else "(Mayor pérdida potencial diaria)"}</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráfico de Drawdown de área
    fig_dd = go.Figure()
    fig_dd.add_scatter(x=df_results["Date"], y=df_results["Drawdown_Algo"], name="Drawdown Algoritmo Híbrido", line_color="#ff4d4d", fill="tozeroy")
    fig_dd.add_scatter(x=df_results["Date"], y=df_results["Drawdown_BH"], name="Drawdown Buy & Hold", line_color="#ff9900", line_dash="dash")
    
    fig_dd.update_layout(
        title="Curva de Drawdown Histórico (%)",
        template="plotly_dark",
        height=320,
        yaxis_title="Drawdown (%)"
    )
    st.plotly_chart(fig_dd, use_container_width=True)
