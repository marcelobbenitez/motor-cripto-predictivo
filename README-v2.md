# Multi-Asset Predictive Engine & Portfolio Stress Testing App - v5

Este repositorio contiene el código de producción de una aplicación de Streamlit diseñada para la gestión, predicción y simulación de escenarios de estrés en carteras multiactivos. El motor de análisis hibrida indicadores técnicos clásicos, métricas macroeconómicas de liquidez interbancaria y modelos de procesamiento de lenguaje natural (NLP) multimodales para predecir movimientos de precios a corto y mediano plazo.

En la **Versión v5**, la aplicación se ha conectado de forma nativa a la **API Pública de Binance**, permitiendo la recopilación de datos de precios en tiempo real y el análisis dinámico de velas para decisiones de trading.

---

## 🔌 Novedades de la Versión v5: Integración con Binance API

La aplicación cuenta con una arquitectura de datos dual, seleccionable directamente desde la barra lateral:

1.  **Cuadro de Tickers en Tiempo Real (Binance API)**: Consulta continuamente las cotizaciones más recientes de pares clave (`BTCUSDT`, `ETHUSDT`, `DOGEUSDT`) a través del endpoint público `https://api.binance.com/api/v3/ticker/price`. Muestra un panel interactivo con estados de conexión en vivo.
2.  **Klines Históricos Dinámicos**: Si se selecciona el modo "Binance Real-Time", la aplicación recupera hasta 180 días de velas japonesas diarias reales mediante el endpoint `/api/v3/klines`.
3.  **Mecanismo de Resiliencia (Offline Fallback)**: El motor incorpora gestión de excepciones robusta para entornos offline o con restricciones de red (como en contenedores cerrados de investigación). Si la conexión con Binance falla o se corta, la aplicación se autoprotege redirigiendo el flujo de datos hacia nuestro generador determinista sintético, garantizando un funcionamiento sin caídas del 100%.

---

## 📑 Fundamentación Científica y Teórica

La aplicación no se basa en análisis especulativos simples, sino en la confluencia de metodologías cuantitativas y de comportamiento de mercado documentadas en la literatura financiera reciente de 2026:

### 1. Motor de Confluencia Técnica y Patrones de Velas
*   **El Filtro de Confluencia**: Estudios empíricos demuestran que los patrones de velas japonesas analizados de forma aislada carecen de un poder predictivo consistente en el mercado de criptomonedas [2]. Sin embargo, adquieren una alta fiabilidad de reversión y continuación de tendencia cuando ocurren exactamente en niveles históricos de soporte y resistencia y son validados por una expansión significativa del volumen de transacciones [2, 8].
*   **Anatomía de Velas**: Las velas japonesas (OHLC) representan la batalla psicológica en tiempo real entre compradores y vendedores [8]. Patrones como el *Hammer* (Martillo) o el *Engulfing* (Engolfamiento) requieren validación mediante osciladores de impulso (MACD, RSI) para filtrar el ruido intradía [2, 8].

### 2. Maduración del Mercado y Perfiles de Volatilidad (2026)
*   **Compresión de la Volatilidad**: Para principios de 2026, la volatilidad anualizada histórica de Bitcoin se ha comprimido a su mínimo histórico del **38%** gracias a la adopción institucional masiva y la maduración de los ETFs al contado, que ahora integran las carteras de más de 4,500 entidades institucionales [3]. 
*   **Sincronización de Activos**: Bitcoin ya no opera como un activo aislado; su correlación con el S&P 500 alcanzó un máximo histórico de **0.65** en marzo de 2026 [3], y su coeficiente de correlación con el Nasdaq llega a ser de hasta **0.93** en ciclos de liquidez macro [5]. Esto justifica tratar a las criptomonedas "Blue Chip" como un sleeve tecnológico de alto crecimiento dentro de una cartera de renta variable tradicional.

### 3. Dinámica de Liquidez Macroeconómica (Efecto Fed & IntoTheBlock)
*   **Inyección Neta de Capital**: De acuerdo con los modelos cuantitativos de IntoTheBlock, los movimientos de los activos de riesgo responden directamente a la liquidez de los bancos centrales [5].
*   **El Índice de Liquidez Neta**: La aplicación rastrea el balance de la Reserva Federal descontando los incrementos en la Cuenta General del Tesoro (TGA) y los saldos del Reverse Repo (RRP) [5]. Una reducción en la cuenta TGA o en el RRP representa una inyección masiva de liquidez neta al sistema interbancario, actuando como un catalizador alcista para BTC y QQQ con un desfase de 48 a 72 horas [5].

### 4. Fusión de Sentimiento Multimodal Cross-Platform
*   **Twitter/X vs. TikTok**: Las investigaciones de la Universidad de Auckland (2025) señalan que los canales de comunicación transmiten el sentimiento de inversores de forma asimétrica [7]. 
*   **Efecto Especulativo**: El sentimiento en videos de formato corto (TikTok) actúa como un motor de volatilidad especulativa de muy corto plazo (especialmente en altcoins minoristas como Dogecoin) [7], mientras que el sentimiento basado en texto (Twitter/X, Reddit) muestra una correlación de mediano y largo plazo con activos principales como Bitcoin [7]. Integrar de forma combinada ambas curvas de sentimiento mejora la precisión predictiva general en un **20%** en BTC, y hasta en un **35%** en Dogecoin en el corto plazo [7].

### 5. Clasificador de Stanford ML (+1D return)
*   **Etiquetado Predictivo Directo**: Rompiendo con los esquemas de NLP tradicionales que clasifican textos de forma genérica como "positivos" o "negativos", este motor implementa el enfoque de Stanford (CS229) [10].
*   **Regresión Regularizada L2**: Entrena una Regresión Logística regularizada con una constante de regularización inversa de $C=0.9$ donde las variables explicativas (sentimiento acumulado y flujos de liquidez) se etiquetan directamente contra la dirección real del precio un día en el futuro ($+1D \in \{0, 1\}$) [10]. Esto optimiza la capacidad del modelo para predecir y anticipar los días con las fluctuaciones porcentuales más extremas del mercado [10].

### 6. Sistema de Trading Cuantitativo e Indicadores Combinados
*   **Trading Strategy**: Toma de decisiones incorpora señales combinadas filtradas por volumen [5, 6].
*   **Filtro VW MACD & Dual MACD**: Se implementa el MACD Ponderado por Volumen (VW MACD) y el Dual MACD (que exige la sincronización del impulso de corto plazo de 12-26-9 días con el institucional de largo plazo de 19-39-9 días) para anular las trampas de consolidación lateral. Las decisiones se complementan con pronósticos basados en series temporales con Suavizado Exponencial Holt-Winters (ETS).

---

## 🚀 Características de la Aplicación

La interfaz de Streamlit está dividida en cuatro módulos de control:

1.  **Tab 1: Panel de Control de Activos**: Visualización de gráficos de velas japonesas interactivos (Plotly) integrando el motor de detección de soportes, resistencias y patrones de confluencia técnica, precedido por el **glowing board de precios de Binance en tiempo real**.
2.  **Tab 2: Centro Predictivo (ML & Holt-Winters)**: Proyecciones de precios a 15 días con modelos estadísticos de suavizado exponencial (ETS) y el porcentaje probabilístico de cierre alcista calculado por el clasificador de Stanford.
3.  **Tab 3: Sentimiento Multimodal & Liquidez Fed**: Comparativa gráfica temporal de la curva institucional de Twitter frente a la curva especulativa de TikTok y su correlación visual con el índice de inyección de liquidez de la Fed.
4.  **Tab 4: Backtesting & Gestión de Riesgo (Stress Testing)**: Simulación de rendimiento histórico comparado contra una estrategia pasiva de *Buy & Hold*. Incluye métricas de riesgo de grado institucional:
    *   **Volatilidad Anualizada del Portafolio** [3].
    *   **Máximo Drawdown (Max DD)**: Rastro y gráfico del peor escenario de caída de capital acumulada.
    *   **Sortino Ratio**: Ajuste de retorno sobre la volatilidad de caídas (desviación a la baja).
    *   **Value at Risk (VaR Histórico 95% 1-Día)**: Máxima pérdida diaria esperada bajo condiciones normales.

### 🚨 Simulador de Shocks de Mercado
Desde la barra lateral, puedes estresar el modelo activando 4 condiciones de crisis reales:
*   **Drenaje de Liquidez Fed (Hawkish Shift)** [5].
*   **Pánico Social Extremo (FUD Masivo en Redes)** [7].
*   **Choque Geopolítico (Petróleo y Oro)**.
*   **Capitulación Global ETF (Crypto Flash Crash)**.

---

## 🛠️ Instalación y Uso Local

Sigue estos pasos para ejecutar la aplicación en tu computadora local:

### 1. Clonar el repositorio y navegar a la carpeta:
```bash
git clone https://github.com/tu-usuario/nombre-repositorio.git
cd nombre-repositorio
```

### 2. Instalar dependencias obligatorias:
Asegúrate de tener Python 3.9 o superior instalado. Ejecuta en la terminal:
```bash
python -m pip install streamlit pandas numpy plotly scikit-learn statsmodels requests
```

### 3. Ejecutar la aplicación:
```bash
python -m streamlit run app.py
```

---

## 📌 Estructura del Repositorio
```text
├── app.py                 # Código fuente principal de la aplicación Streamlit
├── requirements.txt       # Archivo de dependencias del entorno de internet (incluye 'requests')
└── README.md              # Documentación técnica y fundamentación científica
```

---

## 🏛️ Referencias Científicas (Grounded Sources)
*   **[Kraken Learn, 2024]** *Candlestick chart patterns: Empower your crypto trading.* [2]
*   **[KuCoin Blog, 2026]** *The Ultimate Guide to Candlestick Patterns: How to Trade Crypto Market Sentiments.* [8]
*   **[KuCoin Blog, 2026]** *Crypto vs stocks: 5 years of historical volatility data in 2026.* [3]
*   **[Bitcoin Foundation, 2026]** *Market Cap vs Market Volume. How to Read and How to Use.* [6]
*   **[IntoTheBlock / CoinMarketCap]** *Analysis of Market Liquidity and its Effect on Crypto Asset's Price and Stablecoin Supply.* [1, 5]
*   **[Universidad de Auckland, arXiv 2025]** *Enhancing Cryptocurrency Sentiment Analysis with Multimodal Features.* [7]
*   **[Universidad de Stanford, CS229]** *Cryptocurrency Price Prediction Using News and Social Media Sentiment.* [10]
*   **[UCLA, arXiv 2025]** *Enhancing Trading Performance Through Sentiment Analysis with Large Language Models: Evidence from the S&P 500.*
*   **[arXiv, 2022]** *Social Media Sentiment Analysis for Cryptocurrency Market Prediction.*
