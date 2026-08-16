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
    page_title="Crypto & Equity Predictive Assets Engine - v4",
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
# GENERADOR DE DATOS HISTÓRICOS Y SIMULADOS CON SHOCKS DE MERCADO (2026)
# -------------------------------------------------------------------
@st.cache_data
def generate_historical_data(days=180, shock_type="Ninguno (Mercado Normal)"):
    np.random.seed(42)
    base_date = datetime.now() - timedelta(days=days)
    dates = [base_date + timedelta(days=i) for i in range(days)]
    
    # 1. Liquidez de la Fed (TGA y RRP en Miles de Millones $)
    # Decrementos en TGA y RRP significan inyección de liquidez neta en el sistema.
    # En un "Fed Liquidity Drain", simulamos que la Fed drena liquidez aumentando RRP/TGA.
    rrp = []
    tga = []
    curr_rrp = 500.0
    curr_tga = 750.0
    for i in range(days):
        if shock_type == "Drenaje de Liquidez Fed (Hawkish Shift)" and i >= days - 45:
            # Subida repentina y violenta que absorbe liquidez del sistema
            curr_rrp += np.random.normal(6.0, 9.0)
            curr_tga += np.random.normal(5.0, 8.0)
        else:
            curr_rrp += np.random.normal(-0.5, 5.0)  # Tendencia general a la baja
            curr_tga += np.random.normal(-0.3, 6.0)
        rrp.append(max(10.0, curr_rrp))
        tga.append(max(10.0, curr_tga))
    
    df_macro = pd.DataFrame({
        "Date": dates,
        "Fed_RRP": rrp,
        "Fed_TGA": tga
    })
    # Índice de Liquidez Neta de Arthur Hayes (Inversa de RRP + TGA)
    df_macro["Net_Liquidity_Index"] = 1500.0 - (df_macro["Fed_RRP"] + df_macro["Fed_TGA"])
    
    # 2. Generación de precios correlacionados
    sp500_price = 5000.0
    btc_price = 60000.0
    eth_price = 3000.0
    doge_price = 0.15
    gold_price = 2000.0
    oil_price = 75.0
    
    prices = {
        "SPY": [], "QQQ": [], "BTC": [], "ETH": [], "DOGE": [], "GOLD": [], "OIL": []
    }
    volumes = {k: [] for k in prices.keys()}
    highs, lows, opens = {k: [] for k in prices.keys()}, {k: [] for k in prices.keys()}, {k: [] for k in prices.keys()}
    
    # Perfil de Volatilidad Anualizada en 2026 (BTC madurando a ~38%)
    vols = {
        "SPY": 0.15, "QQQ": 0.18, "BTC": 0.38, "ETH": 0.45, "DOGE": 0.85, "GOLD": 0.12, "OIL": 0.30
    }
    
    # Incrementar la volatilidad bajo condiciones de shock generalizado
    if shock_type != "Ninguno (Mercado Normal)":
        for k in vols.keys():
            vols[k] *= 1.45 # Aumento del 45% en la volatilidad por stress test
    
    for i in range(days):
        # Factor común de liquidez
        liq_factor = (df_macro["Net_Liquidity_Index"].iloc[i] - df_macro["Net_Liquidity_Index"].mean()) / df_macro["Net_Liquidity_Index"].std()
        market_shock = np.random.normal(0, 1) + 0.1 * liq_factor
        
        # Rendimientos base por activos
        sp_ret = vols["SPY"] / np.sqrt(365) * market_shock + np.random.normal(0.0003, 0.005)
        qqq_ret = vols["QQQ"] / np.sqrt(365) * (market_shock * 1.2 + np.random.normal(0.0004, 0.006))
        btc_ret = vols["BTC"] / np.sqrt(365) * (market_shock * 0.85 + np.random.normal(0.0005, 0.012))
        eth_ret = vols["ETH"] / np.sqrt(365) * (market_shock * 0.90 + np.random.normal(0.0006, 0.015))
        doge_ret = vols["DOGE"] / np.sqrt(365) * (market_shock * 1.5 + np.random.normal(-0.0002, 0.035))
        
        gold_ret = vols["GOLD"] / np.sqrt(365) * (-market_shock * 0.2 + np.random.normal(0.0002, 0.004))
        oil_ret = vols["OIL"] / np.sqrt(365) * (market_shock * 0.4 + np.random.normal(0.0001, 0.010))
        
        # Aplicación de escenarios de Shock Técnico y Macroeconómico
        if i >= days - 45:
            if shock_type == "Drenaje de Liquidez Fed (Hawkish Shift)":
                # Liquidez cayendo arrastra activos de riesgo, oro sirve de cobertura débil
                sp_ret -= 0.0025
                qqq_ret -= 0.0035
                btc_ret -= 0.0070
                eth_ret -= 0.0090
                doge_ret -= 0.0180
                gold_ret += 0.0006
            elif shock_type == "Pánico Social Extremo (FUD Masivo)":
                # Criptomonedas de alta especulación sufren retail run
                btc_ret -= 0.0040
                eth_ret -= 0.0120
                doge_ret -= 0.0280
            elif shock_type == "Choque Geopolítico (Petróleo y Oro)":
                # Oro y petróleo escalan por tensiones geopolíticas; renta variable sufre por inflación
                oil_ret += 0.0180
                gold_ret += 0.0090
                sp_ret -= 0.0030
                qqq_ret -= 0.0040
                btc_ret -= 0.0015
                
        if i >= days - 15:
            if shock_type == "Capitulación Global ETF (Crypto Flash Crash)":
                # Flash crash masivo en los últimos 15 días simulados
                btc_ret -= 0.0280
                eth_ret -= 0.0380
                doge_ret -= 0.0550
                sp_ret -= 0.0060
                qqq_ret -= 0.0080
                gold_ret += 0.0025 # Oro sube por fuerte demanda de refugio físico
                
        sp500_price *= (1 + sp_ret)
        btc_price *= (1 + btc_ret)
        eth_price *= (1 + eth_ret)
        doge_price *= (1 + doge_ret)
        gold_price *= (1 + gold_ret)
        oil_price *= (1 + oil_ret)
        
        # Guardar cierres
        prices["SPY"].append(sp500_price / 10.0) # Normalizado como ETF SPY
        prices["QQQ"].append(sp500_price / 12.0) # Normalizado como ETF QQQ
        prices["BTC"].append(btc_price)
        prices["ETH"].append(eth_price)
        prices["DOGE"].append(doge_price)
        prices["GOLD"].append(gold_price)
        prices["OIL"].append(oil_price)
        
        # Construir Velas (OHLC) y Volúmenes
        for k in prices.keys():
            close = prices[k][-1]
            ret_std = vols[k] / np.sqrt(365)
            o = close * (1 + np.random.normal(0, ret_std * 0.3))
            h = max(o, close) * (1 + abs(np.random.normal(0, ret_std * 0.2)))
            l = min(o, close) * (1 - abs(np.random.normal(0, ret_std * 0.2)))
            
            opens[k].append(o)
            highs[k].append(h)
            lows[k].append(l)
            
            # Volúmenes base en Millones
            base_vol = {"SPY": 5000, "QQQ": 4000, "BTC": 25000, "ETH": 12000, "DOGE": 1500, "GOLD": 800, "OIL": 1200}
            
            # Multiplicadores de volumen por pánico o liquidaciones
            vol_multiplier = 1.0
            if shock_type == "Capitulación Global ETF (Crypto Flash Crash)" and i >= days - 15:
                vol_multiplier = 3.8 # Volumen salvaje de capitulación
            elif shock_type == "Pánico Social Extremo (FUD Masivo)" and i >= days - 45:
                vol_multiplier = 2.2
            elif shock_type == "Drenaje de Liquidez Fed (Hawkish Shift)" and i >= days - 45:
                vol_multiplier = 1.6
                
            vol = base_vol[k] * (1 + np.random.normal(0.2, 0.4)) * (1 + abs(liq_factor) * 0.5) * vol_multiplier
            volumes[k].append(max(base_vol[k] * 0.1, vol))
            
    # Sentimiento Social (Twitter vs TikTok)
    twitter_sent = []
    tiktok_sent = []
    for i in range(days):
        market_trend = (prices["BTC"][i] - prices["BTC"][max(0, i-5)]) / prices["BTC"][max(0, i-5)]
        
        tw = 5.0 + market_trend * 25.0 + np.random.normal(0, 1.2)
        tk = 5.0 + market_trend * 45.0 + np.random.normal(0, 2.5)
        
        # Alterar el sentimiento de las redes si hay shock social
        if shock_type == "Pánico Social Extremo (FUD Masivo)" and i >= days - 45:
            tw = np.random.normal(1.8, 0.7) # FUD total institucional
            tk = np.random.normal(0.8, 0.4) # Capitulación absoluta en vídeo minorista
        elif shock_type == "Capitulación Global ETF (Crypto Flash Crash)" and i >= days - 15:
            tw = np.random.normal(1.2, 0.5)
            tk = np.random.normal(0.5, 0.3)
            
        twitter_sent.append(clip(tw, 0.0, 10.0))
        tiktok_sent.append(clip(tk, 0.0, 10.0))
        
    df_all = pd.DataFrame({"Date": dates})
    df_all["Fed_RRP"] = df_macro["Fed_RRP"]
    df_all["Fed_TGA"] = df_macro["Fed_TGA"]
    df_all["Net_Liquidity_Index"] = df_macro["Net_Liquidity_Index"]
    df_all["Twitter_Sentiment"] = twitter_sent
    df_all["TikTok_Sentiment"] = tiktok_sent
    
    for k in prices.keys():
        df_all[f"{k}_Close"] = prices[k]
        df_all[f"{k}_Open"] = opens[k]
        df_all[f"{k}_High"] = highs[k]
        df_all[f"{k}_Low"] = lows[k]
        df_all[f"{k}_Volume_M"] = volumes[k]
        
    return df_all

def clip(val, min_val, max_val):
    return max(min_val, min(val, max_val))

# -------------------------------------------------------------------
# PANEL LATERAL Y CONFIGURACIÓN DEL ENTORNO
# -------------------------------------------------------------------
st.sidebar.title("🛠️ Configuración de Motor")

# Selector de Shock de Mercado (Stress Testing)
st.sidebar.subheader("🚨 Simulación de Shocks (Stress Test)")
shock_type = st.sidebar.selectbox("Selecciona un Escenario de Shock:", [
    "Ninguno (Mercado Normal)",
    "Drenaje de Liquidez Fed (Hawkish Shift)",
    "Pánico Social Extremo (FUD Masivo)",
    "Choque Geopolítico (Petróleo y Oro)",
    "Capitulación Global ETF (Crypto Flash Crash)"
])

st.sidebar.subheader("Sleeves de Activos")
sleeve = st.sidebar.selectbox("Selecciona un Grupo de Activos:", [
    "Oro 2.0 (Bitcoin)",
    "Altcoins (Ethereum, Dogecoin)",
    "ETFs de Renta Variable (SPY, QQQ)",
    "Materias Primas & Refugios (Gold, Oil)"
])

# Mapear sleeve a los activos específicos correspondientes
if sleeve == "Oro 2.0 (Bitcoin)":
    active_assets = ["BTC"]
elif sleeve == "Altcoins (Ethereum, Dogecoin)":
    active_assets = ["ETH", "DOGE"]
elif sleeve == "ETFs de Renta Variable (SPY, QQQ)":
    active_assets = ["SPY", "QQQ"]
else:
    active_assets = ["GOLD", "OIL"]

selected_asset = st.sidebar.selectbox("Activo Activo de Análisis:", active_assets)

# Cargar datos históricos modificados dinámicamente por el shock_type
df = generate_historical_data(shock_type=shock_type)

# Configuración del backtester en la barra lateral
st.sidebar.subheader("📈 Backtesting de Portafolio")
initial_capital = st.sidebar.number_input("Capital Inicial ($)", value=10000, step=1000)
slippage_fee = st.sidebar.slider("Comisiones & Slippage por Trade (%)", 0.0, 1.0, 0.1, step=0.05) / 100.0

st.sidebar.markdown("""
---
**Acerca del Motor de Predicción:**
Esta aplicación implementa un sistema híbrido que integra análisis técnico con modelos de procesamiento de lenguaje natural y factores macro. Basado en los últimos descubrimientos del reporte de 2026.
""")

# -------------------------------------------------------------------
# MOTOR DE CONFLUENCIA TÉCNICA (Velas, Soporte/Resistencia, Volumen)
# -------------------------------------------------------------------
def calculate_support_resistance(df, asset):
    close_series = df[f"{asset}_Close"]
    high_series = df[f"{asset}_High"]
    low_series = df[f"{asset}_Low"]
    
    # Calcular niveles utilizando máximos/mínimos locales simples
    support = low_series.rolling(window=20).min().iloc[-1]
    resistance = high_series.rolling(window=20).max().iloc[-1]
    return support, resistance

def detect_candlestick_patterns(df, asset):
    # Últimas 3 velas para análisis de patrones
    last_idx = -1
    o, h, l, c = df[f"{asset}_Open"].iloc[last_idx], df[f"{asset}_High"].iloc[last_idx], df[f"{asset}_Low"].iloc[last_idx], df[f"{asset}_Close"].iloc[last_idx]
    
    # Atributos de vela única
    body_size = abs(c - o)
    candle_range = h - l if h - l != 0 else 0.001
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    is_bullish = c > o
    is_bearish = c < o
    
    pattern = "Neutral"
    bias = 0.0 # -1 para Bearish fuerte, 1 para Bullish fuerte, 0 para neutral
    
    # Martillo (Hammer) / Pin Bar Bullish
    if (lower_wick > 2 * body_size) and (upper_wick < 0.1 * candle_range):
        pattern = "Bullish Hammer / Pin Bar"
        bias = 0.6
        
    # Hombre Colgado (Hanging Man) / Pin Bar Bearish
    elif (upper_wick > 2 * body_size) and (lower_wick < 0.1 * candle_range):
        pattern = "Bearish Shooting Star / Hanging Man"
        bias = -0.6
        
    # Engolfamiento (Engulfing)
    prev_o, prev_c = df[f"{asset}_Open"].iloc[last_idx-1], df[f"{asset}_Close"].iloc[last_idx-1]
    
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
    confluence_score = 0.0 # [-1, 1]
    
    # Validamos patrón con Volumen y Niveles Clave
    near_support = abs(last_close - sup) / last_close < 0.02
    near_resistance = abs(last_close - res) / last_close < 0.02
    vol_expansion = last_vol > 1.2 * avg_vol
    
    if bias > 0 and near_support and vol_expansion:
        confluence_triggered = True
        confluence_score = bias * 1.2 # Amplificación por confluencia
    elif bias < 0 and near_resistance and vol_expansion:
        confluence_triggered = True
        confluence_score = bias * 1.2
    else:
        # Si no hay confluencia de volumen o soportes, reducimos el peso del patrón
        confluence_score = bias * 0.4
        
    # Clampar score entre -1.0 y 1.0
    confluence_score = clip(confluence_score, -1.0, 1.0)
    return pattern, confluence_triggered, confluence_score

# -------------------------------------------------------------------
# METRICAS DE CAPITALIZACIÓN Y GIRO (TURNOVER)
# -------------------------------------------------------------------
def get_turnover_ratio(df, asset):
    # Ratio Volumen a Capitalización de Mercado (Turnover)
    mcaps = {
        "BTC": 1500000, "ETH": 400000, "DOGE": 25000, 
        "SPY": 500000, "QQQ": 350000, "GOLD": 15000000, "OIL": 2500000
    } # En Millones $
    
    last_vol = df[f"{asset}_Volume_M"].iloc[-1]
    mcap = mcaps[asset]
    ratio = (last_vol / mcap) * 100.0
    
    if ratio < 0.5:
        desc = "Giro Bajo (Low Turnover) - Posible falta de confirmación de tendencia o spreads amplios."
        status = "Warning"
    elif ratio <= 3.0:
        desc = "Giro Normal - Interés institucional estándar y flujos saludables."
        status = "Healthy"
    elif ratio <= 12.0:
        desc = "Giro Activo (Active Repricing) - Fuerte interés y volumen que confirma rotación de precio."
        status = "Strong"
    else:
        desc = "Mercado Muy Caliente (Speculative Hot Spot) - Cuidado con sobrecompra, alta volatilidad o manipulación."
        status = "Hot"
        
    return ratio, desc, status

# -------------------------------------------------------------------
# MOTOR DE SENTIMIENTO MULTIMODAL Y ML (ESTILO STANFORD)
# -------------------------------------------------------------------
def calculate_multimodal_sentiment(df, asset):
    # En base al estudio de la Universidad de Auckland, combinamos Twitter (largo plazo) y TikTok (especulativo corto plazo)
    last_tw = df["Twitter_Sentiment"].iloc[-1]
    last_tk = df["TikTok_Sentiment"].iloc[-1]
    
    if asset == "DOGE":
        combined_score = (0.35 * last_tw + 0.65 * last_tk)
    elif asset in ["BTC", "GOLD", "SPY"]:
        combined_score = (0.75 * last_tw + 0.25 * last_tk)
    else:
        combined_score = (0.55 * last_tw + 0.45 * last_tk)
        
    return combined_score / 10.0 # Normalizado a escala [0, 1]

def run_stanford_predictive_model(df, asset):
    # Estilo Stanford: En lugar de analizar sentimiento polarizado, entrenamos un clasificador
    # con variables de texto (aquí representadas por series de sentimiento) etiquetadas con el retorno futuro (+1D)
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
# MOTOR DE TRADING COMBINADO (MACD, ETS & FORECASTING)
# -------------------------------------------------------------------
def calculate_signals_and_backtest(df, asset, init_cap, fee):
    df_bt = df.copy()
    
    # 1. Dual MACD (Filtro Técnico)
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
    
    # 2. Generar Señal Combinada (I_t) diaria
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
            
        # Sentimiento normalizado remapeado a [-0.5, 0.5]
        sent_combined = calculate_multimodal_sentiment(df_bt.iloc[:i+1], asset) - 0.5
        
        # Confluencia Técnica remapeada a [-0.5, 0.5]
        _, _, tech_score = check_technical_confluence(df_bt.iloc[:i+1], asset)
        tech_score = tech_score * 0.5
        
        # Filtro de Liquidez Macro: Retorno del Índice de Liquidez en 5 días
        liq_index = df_bt["Net_Liquidity_Index"].iloc[i]
        prev_liq = df_bt["Net_Liquidity_Index"].iloc[max(0, i-5)]
        liq_signal = 0.5 if liq_index > prev_liq else -0.5
        
        # Señal unificada I_t en el rango [-1, 1]
        raw_it = sent_combined * 0.4 + tech_score * 0.4 + liq_signal * 0.2
        it_signal = clip(raw_it * 2.0, -1.0, 1.0)
        signals.append(it_signal)
        
        # Ejecución técnica de Dual MACD
        bullish_macd = (macd_short.iloc[i] > signal_short.iloc[i]) and (macd_long.iloc[i] > signal_long.iloc[i])
        bearish_macd = (macd_short.iloc[i] < signal_short.iloc[i]) and (macd_long.iloc[i] < signal_long.iloc[i])
        
        # Lógica del Trading Algorithm
        price = df_bt[f"{asset}_Close"].iloc[i]
        current_pos = positions[-1] if len(positions) > 0 else 0
        
        # Filtro de confirmación de señales con umbrales
        if it_signal > 0.25 and bullish_macd and current_pos == 0:
            holdings = (cash * (1.0 - fee)) / price
            cash = 0.0
            positions.append(1) # Long
        elif (it_signal < -0.25 or bearish_macd) and current_pos == 1:
            cash = holdings * price * (1.0 - fee)
            holdings = 0.0
            positions.append(0) # Cash
        else:
            positions.append(current_pos)
            
        current_val = cash + (holdings * price)
        portfolio_value.append(current_val)
        
    df_bt["Signal_It"] = signals
    df_bt["Position"] = positions
    df_bt["Portfolio_Value"] = portfolio_value
    
    # Estrategia de Buy & Hold para comparación
    df_bt["BH_Value"] = (df_bt[f"{asset}_Close"] / df_bt[f"{asset}_Close"].iloc[30]) * init_cap
    df_bt.loc[:30, "BH_Value"] = init_cap
    
    return df_bt

# Ejecutar el Backtester
df_results = calculate_signals_and_backtest(df, selected_asset, initial_capital, slippage_fee)

# -------------------------------------------------------------------
# RENDERIZADO DE LA INTERFAZ DE USUARIO (UI)
# -------------------------------------------------------------------
st.title("⚡ Crypto & Equity Assets Predictive Engine - v4")
st.markdown("### Modelador Predictivo y Plataforma de Stress Testing Multimodal")

# Alerta visible si hay un entorno de stress testing activo
if shock_type != "Ninguno (Mercado Normal)":
    st.markdown(f"""
    <div style="background-color: #3b0d11; border: 2px solid #ff3333; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
        <h4 style="color: #ff3333; margin: 0; font-weight: 700;">🚨 ENTORNO DE STRESS TESTING ACTIVO: {shock_type.upper()}</h4>
        <p style="color: #f7a8a8; margin: 5px 0 0 0; font-size: 14px;">
            Se están simulando choques en el sistema de datos. Las volatilidades aumentaron un 45% y las series históricas de liquidez, precios y sentimientos sociales reflejan distorsiones extremas. Compara cómo el algoritmo híbrido busca preservar capital frente a la caída del mercado.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Secciones de Métricas de Encabezado
col1, col2, col3, col4 = st.columns(4)

with col1:
    last_price = df_results[f"{selected_asset}_Close"].iloc[-1]
    prev_price = df_results[f"{selected_asset}_Close"].iloc[-2]
    pct_change = ((last_price - prev_price) / prev_price) * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Precio Actual ({selected_asset})</div>
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
    st.subheader("Análisis de Tendencias y Perfiles de Volatilidad")
    
    # Gráfico de Precios Clásico con Velas
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
        title=f"Evolución y Velas Japonesas de {selected_asset} | Escenario: {shock_type}",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=450
    )
    st.plotly_chart(fig_candles, use_container_width=True)
    
    st.info(f"💡 **Interpretación del Giro Técnico**: {turn_desc}")

# -------------------------------------------------------------------
# TAB 2: CENTRO PREDICTIVO (ML & HOLT-WINTERS)
# -------------------------------------------------------------------
with tab2:
    st.subheader("Proyecciones de Modelos Cuantitativos Avanzados")
    
    # 1. Ajuste de Holt-Winters (ETS)
    close_data = df_results[f"{selected_asset}_Close"].values
    ets_model = ExponentialSmoothing(close_data, seasonal="add", seasonal_periods=7, trend="add").fit()
    forecast_days = 15
    ets_forecast = ets_model.forecast(forecast_days)
    
    # Construcción de fechas futuras para graficar
    future_dates = [df_results["Date"].iloc[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
    
    # Graficar Ajuste y Predicción en Plotly
    fig_pred = go.Figure()
    fig_pred.add_scatter(x=df_results["Date"].iloc[-30:], y=df_results[f"{selected_asset}_Close"].iloc[-30:], name="Histórico (Últimos 30 días)", line_color="#8a9bb4")
    fig_pred.add_scatter(x=future_dates, y=ets_forecast, name="Pronóstico ETS (Holt-Winters) - 15D", line=dict(color="#00ffcc", width=3, dash="dash"))
    
    fig_pred.update_layout(
        title=f"Predicción del Modelo ETS para {selected_asset} (Bajo {shock_type})",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig_pred, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        ### 🧠 Explicación del Modelo Stanford ML
        El clasificador predictivo **Stanford-Style** de esta sección se salta la subjetividad de etiquetar textos como 'positivos' o 'negativos'. 
        
        En su lugar, entrena una **Regresión Logística L2** correlacionando el historial combinado de flujos de sentimiento social y liquidez directamente con la dirección del precio del activo del día de mañana (+1D).
        
        *   **Probabilidad de Mañana:** {prob_up*100:.2f}% de probabilidad de cierre alcista.
        *   **Dirección Predicha:** {"**COMPRA / ALCISTA**" if prob_up > 0.52 else "**VENTA / BAJISTA**" if prob_up < 0.48 else "**NEUTRAL / CONSOLIDACIÓN**"}
        """)
    with col_b:
        st.markdown(f"""
        ### 📉 Gestión del Perfil de Volatilidad (Maturación de 2026)
        Tradicionalmente, las criptomonedas han operado en mercados altamente fragmentados de volatilidad salvaje. Sin embargo, en el año de análisis **2026**, la entrada masiva de capital institucional y los ETFs al contado han provocado una **compresión masiva de volatilidad**.
        
        *   **Volatilidad de BTC en 2026:** Estabilizada originalmente cerca del **38% anualizado** (mínimo histórico de una década).
        *   **Implicación bajo stress test:** En este escenario de *{shock_type}*, las desviaciones se disparan y las correlaciones se estrechan. Observa cómo el modelo de trading gesiona este estrés temporal.
        """)

# -------------------------------------------------------------------
# TAB 3: SENTIMIENTO MULTIMODAL Y LIQUIDEZ FED
# -------------------------------------------------------------------
with tab3:
    st.subheader("Análisis Cualitativo y Flujos de Inyección de Liquidez")
    
    # Gráfico Dual: Sentimiento Multimodal (Twitter vs TikTok)
    fig_sent = go.Figure()
    fig_sent.add_scatter(x=df_results["Date"], y=df_results["Twitter_Sentiment"], name="Twitter/X (Señal Institucional - Texto)", line_color="#1DA1F2")
    fig_sent.add_scatter(x=df_results["Date"], y=df_results["TikTok_Sentiment"], name="TikTok (Señal Especulativa Retail - Vídeo)", line_color="#EE1D52")
    
    fig_sent.update_layout(
        title=f"Fusión del Sentimiento Multimodal Cross-Platform | Escenario: {shock_type}",
        template="plotly_dark",
        height=350,
        yaxis_title="Escala de Sentimiento (0-10)"
    )
    st.plotly_chart(fig_sent, use_container_width=True)
    
    # Gráfico de Liquidez Fed
    fig_liq = go.Figure()
    fig_liq.add_scatter(x=df_results["Date"], y=df_results["Net_Liquidity_Index"], name="Índice de Liquidez Neta (Fed)", line_color="#00ffcc", fill="tozeroy")
    
    fig_liq.update_layout(
        title="Inyección Macro de Liquidez Neta (Inversa de RRP + Cuenta TGA)",
        template="plotly_dark",
        height=300,
        yaxis_title="Monto del Índice (Billones $)"
    )
    st.plotly_chart(fig_liq, use_container_width=True)
    
    st.warning("⚠️ **Dinámica Clave de Liquidez (Efecto IntoTheBlock)**: Un descenso en el Reverse Repo (RRP) o en la cuenta general del Tesoro (TGA) representa una liberación masiva de capital hacia el mercado interbancario, lo cual históricamente cataliza rallies en activos de riesgo (como BTC y QQQ) en cuestión de 48-72 horas.")

# -------------------------------------------------------------------
# TAB 4: BACKTESTING & GESTIÓN DE RIESGO
# -------------------------------------------------------------------
with tab4:
    st.subheader(f"Simulación Histórica: Algoritmo Hibridizado vs. Buy and Hold | Shock: {shock_type}")
    
    # Gráfico del Backtest
    fig_bt = go.Figure()
    fig_bt.add_scatter(x=df_results["Date"], y=df_results["Portfolio_Value"], name="Estrategia Algorítmica Híbrida (Velas + Sentimiento + Macro)", line_color="#00ffcc", line_width=2.5)
    fig_bt.add_scatter(x=df_results["Date"], y=df_results["BH_Value"], name="Estrategia Pasiva (Buy & Hold)", line_color="#ff9900", line_dash="dash")
    
    fig_bt.update_layout(
        title="Curva de Crecimiento de Capital ($10K Inicial)",
        template="plotly_dark",
        height=400,
        yaxis_title="Valor de Cuenta ($)"
    )
    st.plotly_chart(fig_bt, use_container_width=True)
    
    # Estadísticas Clave de Rendimiento
    final_algo = df_results["Portfolio_Value"].iloc[-1]
    final_bh = df_results["BH_Value"].iloc[-1]
    
    ret_algo = ((final_algo - initial_capital) / initial_capital) * 100
    ret_bh = ((final_bh - initial_capital) / initial_capital) * 100
    
    # Sharpe Ratio estimado
    returns_algo = df_results["Portfolio_Value"].pct_change().dropna()
    returns_bh = df_results["BH_Value"].pct_change().dropna()
    
    sharpe_algo = (returns_algo.mean() / returns_algo.std()) * np.sqrt(365) if returns_algo.std() != 0 else 0
    sharpe_bh = (returns_bh.mean() / returns_bh.std()) * np.sqrt(365) if returns_bh.std() != 0 else 0
    
    # MÉTRICAS DE RIESGO AVANZADAS
    # 1. Volatilidad Anualizada del Portafolio
    vol_algo = (returns_algo.std() * np.sqrt(365)) * 100 if len(returns_algo) > 0 else 0.0
    vol_bh = (returns_bh.std() * np.sqrt(365)) * 100 if len(returns_bh) > 0 else 0.0
    
    # 2. Máximo Drawdown (Max DD)
    def calc_max_dd(val_series):
        cum_max = val_series.cummax()
        dd = (val_series - cum_max) / cum_max
        return dd.min() * 100
    
    max_dd_algo = calc_max_dd(df_results["Portfolio_Value"])
    max_dd_bh = calc_max_dd(df_results["BH_Value"])
    
    # 3. Value at Risk (VaR Histórico 95% 1-Día)
    var_95_algo = np.percentile(returns_algo, 5) * 100 if len(returns_algo) > 0 else 0.0
    var_95_bh = np.percentile(returns_bh, 5) * 100 if len(returns_bh) > 0 else 0.0
    
    # 4. Sortino Ratio (Enfoque en riesgo bajista)
    def calc_sortino(ret_series):
        if len(ret_series) == 0:
            return 0.0
        mean_ret = ret_series.mean()
        downside = ret_series[ret_series < 0]
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
        return (mean_ret / downside.std()) * np.sqrt(365)
        
    sortino_algo = calc_sortino(returns_algo)
    sortino_bh = calc_sortino(returns_bh)
    
    # 5. Calcular curvas de drawdown diarias para graficar
    cum_max_algo = df_results["Portfolio_Value"].cummax()
    dd_algo_curve = (df_results["Portfolio_Value"] - cum_max_algo) / cum_max_algo * 100
    cum_max_bh = df_results["BH_Value"].cummax()
    dd_bh_curve = (df_results["BH_Value"] - cum_max_bh) / cum_max_bh * 100
    
    df_results["Drawdown_Algo"] = dd_algo_curve
    df_results["Drawdown_BH"] = dd_bh_curve
    
    # Renderizar Métricas Principales en Columnas
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric("Retorno Algoritmo Híbrido", f"{ret_algo:.2f}%", f"${final_algo:,.2f}")
    with stat_col2:
        st.metric("Retorno Buy & Hold", f"{ret_bh:.2f}%", f"${final_bh:,.2f}")
    with stat_col3:
        st.metric("Sharpe Ratio (Híbrido vs B&H)", f"{sharpe_algo:.2f}", f"B&H Sharpe: {sharpe_bh:.2f}")
        
    st.markdown("---")
    st.subheader(f"🛡️ Reporte Analítico de Control de Riesgo bajo estrés ({shock_type})")
    
    # Tabla de métricas cruzadas
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
          <th style="padding: 12px; text-align: left; font-weight: 600;">Métrica de Rendimiento y Riesgo</th>
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
          <td style="padding: 12px; font-weight: 500;">Sharpe Ratio (Retorno/Riesgo Total)</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #00ffcc;">{sharpe_algo:.2f}</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff9900;">{sharpe_bh:.2f}</td>
          <td style="padding: 12px; text-align: center; font-weight: 700; color: {'#00ffcc' if diff_sharpe >= 0 else '#ff4d4d'};">{diff_sharpe:+.2f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #2d3748;">
          <td style="padding: 12px; font-weight: 500;">Sortino Ratio (Retorno/Volatilidad Bajista)</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #00ffcc;">{sortino_algo:.2f}</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff9900;">{sortino_bh:.2f}</td>
          <td style="padding: 12px; text-align: center; font-weight: 700; color: {'#00ffcc' if diff_sortino >= 0 else '#ff4d4d'};">{diff_sortino:+.2f}</td>
        </tr>
        <tr style="border-bottom: 1px solid #2d3748;">
          <td style="padding: 12px; font-weight: 500;">Máximo Drawdown (Max DD - Caída Histórica)</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff4d4d;">{max_dd_algo:.2f}%</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff4d4d;">{max_dd_bh:.2f}%</td>
          <td style="padding: 12px; text-align: center; font-weight: 700; color: {'#00ffcc' if max_dd_algo > max_dd_bh else '#ff4d4d'};">{diff_max_dd:+.2f}% {"(Menos caída)" if max_dd_algo > max_dd_bh else "(Mayor caída)"}</td>
        </tr>
        <tr>
          <td style="padding: 12px; font-weight: 500;">Value at Risk (VaR Histórico 95% 1-Día)</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff4d4d;">{var_95_algo:.2f}%</td>
          <td style="padding: 12px; text-align: right; font-weight: 700; color: #ff4d4d;">{var_95_bh:.2f}%</td>
          <td style="padding: 12px; text-align: center; font-weight: 700; color: {'#00ffcc' if var_95_algo > var_95_bh else '#ff4d4d'};">{diff_var:+.2f}% {"(Menor pérdida potencial diaria)" if var_95_algo > var_95_bh else "(Mayor pérdida potencial diaria)"}</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráfico interactivo de Drawdowns sobre el tiempo
    fig_dd = go.Figure()
    fig_dd.add_scatter(x=df_results["Date"], y=df_results["Drawdown_Algo"], name="Drawdown Algoritmo Híbrido", line_color="#ff4d4d", fill="tozeroy")
    fig_dd.add_scatter(x=df_results["Date"], y=df_results["Drawdown_BH"], name="Drawdown Buy & Hold", line_color="#ff9900", line_dash="dash")
    
    fig_dd.update_layout(
        title="Curva de Pérdida Máxima Diaria en el Tiempo (Drawdown %)",
        template="plotly_dark",
        height=320,
        yaxis_title="Drawdown (%)"
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    st.info("💡 **Explicación Analítica de Riesgos**: El **Value at Risk (VaR 95%)** indica la pérdida máxima diaria esperada con un 95% de nivel de confianza. El **Sortino Ratio** complementa al Sharpe ratio ignorando las desviaciones positivas, enfocándose únicamente en la volatilidad dañina de las pérdidas.")
