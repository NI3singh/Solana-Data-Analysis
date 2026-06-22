import os

import json

import time

import random

import atexit

import logging

import threading

import traceback

import warnings

from datetime import datetime, timedelta


import numpy as np

import pandas as pd

import plotly.graph_objects as go

import streamlit as st

import websocket

from binance.client import Client

from binance.exceptions import BinanceAPIException


# ──────────────────────────── logging & warnings ────────────────────────────

warnings.filterwarnings("ignore")

logging.getLogger("websocket").setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

logger = logging.getLogger(__name__)



# ──────────────────────────── shared (thread-safe) state ────────────────────

# The WebSocket thread mutates the globals below; the Streamlit main thread

# reads them through `sync_state()`. Two separate locks keep concerns clean:

_state_lock = threading.Lock()        # protects OHLCV + price globals

_ws_lock = threading.Lock()           # protects WebSocket lifecycle globals


chart_data_global: pd.DataFrame = pd.DataFrame()

live_price_global: float = 0.0

prev_price_global: float = 0.0

last_tick_global = None               # datetime of most recent WS tick


ws_connected: bool = False            # mirrors WebSocketApp open/close state

ws_app = None                         # current WebSocketApp instance

current_pair: str = "solusdt"         # always lowercase; matches Binance URL

current_interval: str = "1d"



# ──────────────────────────── page config & theme ───────────────────────────

st.set_page_config(

    page_title="Solana Live Candlestick Chart",

    page_icon="📈",

    layout="wide",

    initial_sidebar_state="expanded",

)


st.markdown(

    """

<style>

    .main { background-color: #0e1117; padding-top: 0; }

    .block-container { padding-top: 1rem; padding-bottom: 0; }

    .stPlotlyChart { height: 70vh !important; }

    .metric-card {

        background-color: #151a28; color: white; padding: 15px;

        border-radius: 5px; text-align: center;

        border: 1px solid #2c3246;

        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);

        margin-bottom: 10px;

    }

    .metric-card.up {

        background-color: rgba(38, 166, 154, 0.1);

        border-left: 4px solid #26a69a;

    }

    .metric-card.down {

        background-color: rgba(239, 83, 80, 0.1);

        border-left: 4px solid #ef5350;

    }

    .metric-title { font-size: 0.9rem; color: #b0b0b0; margin-bottom: 5px; }

    .metric-value { font-size: 1.5rem; font-weight: 700; margin-bottom: 5px; }

    .change-value.positive { color: #26a69a; }

    .change-value.negative { color: #ef5350; }

    .stButton>button {

        background-color: #26a69a; color: white; font-weight: bold;

        border: none; padding: 10px 24px; border-radius: 4px;

        display: block; margin: 0 auto; width: auto;

        transition: background-color 0.2s;

    }

    .stButton>button:hover { background-color: #2bbbad; }

    div[data-testid="stSidebar"] { background-color: #0b0e14; padding: 2rem 1rem; }

    div[data-testid="stSidebar"] h1,

    div[data-testid="stSidebar"] h2,

    div[data-testid="stSidebar"] h3 { color: #e0e0e0; }

    div[data-testid="stSidebar"] .stSelectbox>div>div,

    div[data-testid="stSidebar"] .stMultiselect>div>div {

        background-color: #1e2130; color: white; border: 1px solid #2c3246;

    }

    div[data-testid="stExpander"] { border: 1px solid #2c3246; border-radius: 5px; }

    h1, h2, h3 { color: white; }

    .stDataFrame table, .stTable table {

        background-color: #1e2130 !important; color: white !important;

    }

    .stDataFrame th, .stTable th {

        background-color: #1e2130 !important; color: white !important;

    }

    .stDataFrame td, .stTable td { color: white !important; }

    .button-container { display: flex; justify-content: center; margin: 20px 0; }

</style>

""",

    unsafe_allow_html=True,

)



# ──────────────────────────── session state defaults ────────────────────────

_DEFAULTS = dict(

    chart_data=pd.DataFrame(),

    last_update=datetime.now(),

    live_price=0.0,

    prev_price=0.0,

    interval="1d",

    days_to_fetch=360,

    pair="SOLUSDT",

    ema_indicators=[20, 50, 100, 200],

    price_change_24h=0.0,

    price_change_24h_pct=0.0,

    status_message="Starting…",

    data_loaded_key=None,

    refresh_count=0,

    update_frequency=2,

)

for _k, _v in _DEFAULTS.items():

    if _k not in st.session_state:

        st.session_state[_k] = _v



# ──────────────────────────── Binance client (no auth needed) ───────────────

@st.cache_resource

def get_binance_client():

    """Public market data (klines, websocket) doesn't require API keys."""

    try:

        api_key = os.environ.get("API_KEY") or None

        api_secret = os.environ.get("API_SECRET") or None

        return Client(api_key, api_secret, {"timeout": 20})

    except Exception as e:

        logger.error("Binance client init failed: %s", e)

        return None



# ──────────────────────────── cached kline fetch (one REST call) ────────────

@st.cache_data(ttl=3600, show_spinner=False)

def fetch_klines_cached(symbol: str, interval: str, days: int) -> pd.DataFrame:

    """

    Fetch historical klines. Cached for 1h per (symbol, interval, days).

    This is THE ONLY place that hits Binance REST, so it must be cheap.

    Retries with exponential backoff on rate-limit errors.

    """

    client = get_binance_client()

    if client is None:

        raise RuntimeError("Binance client could not be initialized")


    last_err = None

    for attempt in range(5):

        try:

            klines = client.get_historical_klines(

                symbol, interval, f"{days} days ago UTC"

            )

            df = pd.DataFrame(

                klines,

                columns=[

                    "timestamp", "open", "high", "low", "close", "volume",

                    "close_time", "quote_asset_volume", "number_of_trades",

                    "taker_buy_base_asset_volume",

                    "taker_buy_quote_asset_volume", "ignore",

                ],

            )

            df = df[["timestamp", "open", "high", "low", "close", "volume"]]

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

            df.set_index("timestamp", inplace=True)

            for col in ("open", "high", "low", "close", "volume"):

                df[col] = df[col].astype(float)

            return df

        except BinanceAPIException as e:

            last_err = e

            if e.code == -1003:  # IP banned / rate limited

                wait = (2 ** attempt) + random.uniform(0.1, 1.0)

                logger.warning(

                    "Binance rate-limited (attempt %d/5). Sleeping %.1fs.",

                    attempt + 1, wait,

                )

                time.sleep(wait)

                continue

            raise

        except Exception as e:

            last_err = e

            logger.error("Kline fetch error: %s", e)

            time.sleep(1)


    raise RuntimeError(f"Failed to fetch klines after retries: {last_err}")



# ──────────────────────────── EMA helper ────────────────────────────────────

def add_ema(df: pd.DataFrame, periods) -> pd.DataFrame:

    if df.empty:

        return df

    for p in periods:

        df[f"EMA_{p}"] = df["close"].ewm(span=p, adjust=False).mean()

    return df



# ──────────────────────────── WebSocket callbacks ───────────────────────────

def _floor(ts: pd.Timestamp, interval: str) -> pd.Timestamp:

    """Floor a timestamp to the start of its candle bucket."""

    if interval.endswith("m"):

        return ts.floor(f"{int(interval[:-1])}min")

    if interval.endswith("h"):

        return ts.floor(f"{int(interval[:-1])}h")

    if interval.endswith("d"):

        return ts.floor("D")

    return ts.floor("min")



def on_ws_message(ws, message):

    """Runs on the WebSocket thread. Mutates globals under `_state_lock`."""

    global live_price_global, prev_price_global, chart_data_global, last_tick_global

    try:

        data = json.loads(message)

        if "k" not in data:

            return


        k = data["k"]

        ts = pd.Timestamp(datetime.fromtimestamp(k["t"] / 1000))

        price = float(k["c"])

        o, h, l, v = float(k["o"]), float(k["h"]), float(k["l"]), float(k["v"])

        candle_closed = bool(k["x"])


        with _state_lock:

            prev_price_global = live_price_global

            live_price_global = price

            last_tick_global = datetime.now()


            if chart_data_global.empty:

                return


            last_idx = chart_data_global.index[-1]

            ema_periods = st.session_state.get("ema_indicators", [])


            same_bucket = (

                _floor(ts, current_interval)

                == _floor(pd.Timestamp(last_idx), current_interval)

            )


            if candle_closed or same_bucket:

                # Update the last row in place (live tick or finalized candle).

                chart_data_global.loc[

                    last_idx, ["open", "high", "low", "close", "volume"]

                ] = [o, h, l, price, v]

            else:

                # New candle interval started → append a new row + refresh EMAs.

                new_row = pd.DataFrame(

                    {

                        "open": [o], "high": [h], "low": [l],

                        "close": [price], "volume": [v],

                    },

                    index=[ts],

                )

                chart_data_global = pd.concat([chart_data_global, new_row])

                for p in ema_periods:

                    chart_data_global[f"EMA_{p}"] = (

                        chart_data_global["close"].ewm(span=p, adjust=False).mean()

                    )

    except Exception as e:

        logger.error("WS message error: %s\n%s", e, traceback.format_exc())



def on_ws_error(ws, error):

    logger.error("WS error: %s", error)



def on_ws_close(ws, code, msg):

    """Schedule a reconnect — but do not touch Streamlit session_state here."""

    global ws_connected

    ws_connected = False

    logger.info("WS closed (%s): %s. Scheduling reconnect in 5s.", code, msg)


    def _reconnect():

        time.sleep(5)

        with _ws_lock:

            if ws_connected:

                return

        try:

            _start_websocket_internal(st.session_state.pair, st.session_state.interval)

        except Exception as e:

            logger.error("Reconnect failed: %s", e)


    threading.Thread(target=_reconnect, daemon=True).start()



def on_ws_open(ws):

    global ws_connected

    ws_connected = True

    logger.info("WS opened: %s @kline_%s", current_pair, current_interval)



def _start_websocket_internal(pair: str, interval: str):

    """Open (or replace) the kline WebSocket. Safe to call from any thread."""

    global ws_app, current_pair, current_interval

    with _ws_lock:

        current_pair = pair.lower()

        current_interval = interval

        if ws_app is not None:

            try:

                ws_app.close()

            except Exception:

                pass

        url = f"wss://stream.binance.com:9443/ws/{current_pair}@kline_{current_interval}"

        logger.info("Connecting WS: %s", url)

        ws_app = websocket.WebSocketApp(

            url,

            on_message=on_ws_message,

            on_error=on_ws_error,

            on_close=on_ws_close,

            on_open=on_ws_open,

        )

        threading.Thread(target=ws_app.run_forever, daemon=True).start()



def stop_websocket():

    global ws_app, ws_connected

    with _ws_lock:

        if ws_app is not None:

            try:

                ws_app.close()

            except Exception:

                pass

        ws_app = None

    ws_connected = False



# ──────────────────────────── data loader (idempotent) ──────────────────────

def load_initial_data(force: bool = False):

    """

    Fetch + cache historical klines.


    Idempotent: if the same (pair, interval, days) was already loaded,

    this is a no-op. That's the single change that fixes the IP ban.

    """

    global chart_data_global, live_price_global, prev_price_global


    key = (

        st.session_state.pair,

        st.session_state.interval,

        st.session_state.days_to_fetch,

    )

    if not force and st.session_state.data_loaded_key == key and not chart_data_global.empty:

        return  # ← THE critical early-return that stops the rate-limit storm


    try:

        df = fetch_klines_cached(*key)

        df = add_ema(df, st.session_state.ema_indicators)


        with _state_lock:

            chart_data_global = df

            live_price_global = float(df["close"].iloc[-1])

            prev_price_global = (

                float(df["close"].iloc[-2]) if len(df) > 1 else live_price_global

            )


        # First-time copy into session_state so the first render isn't empty.

        st.session_state.chart_data = df

        st.session_state.live_price = live_price_global

        st.session_state.prev_price = prev_price_global

        st.session_state.data_loaded_key = key

        st.session_state.last_update = datetime.now()

        st.session_state.refresh_count += 1

        st.session_state.status_message = (

            f"✅ Loaded {len(df)} candles for {key[0]} {key[1]} ({key[2]} days)"

        )

        logger.info(st.session_state.status_message)

    except BinanceAPIException as e:

        if e.code == -1003:

            st.session_state.status_message = (

                "⚠️ Binance has temporarily throttled this server's IP "

                "(code -1003). The ban usually clears in 5–15 minutes. "

                "The live WebSocket tick stream is independent and will keep "

                "updating the chart — please avoid manual page refreshes."

            )

        else:

            st.session_state.status_message = f"⚠️ Binance error: {e}"

        logger.error("BinanceAPIException in load_initial_data: %s", e)

    except Exception as e:

        st.session_state.status_message = f"⚠️ Failed to load data: {e}"

        logger.error("load_initial_data failed: %s\n%s", e, traceback.format_exc())



# ──────────────────────────── thread → UI sync ─────────────────────────────

def sync_state():

    """Copy thread-safe globals into session_state for the UI."""

    with _state_lock:

        if not chart_data_global.empty:

            # `.copy()` so render-thread sees a stable snapshot even if WS

            # continues mutating the original between placeholders.

            st.session_state.chart_data = chart_data_global

        st.session_state.live_price = live_price_global

        st.session_state.prev_price = prev_price_global



def calculate_24h_change():

    df = st.session_state.chart_data

    if df.empty:

        return

    cur = st.session_state.live_price or float(df["close"].iloc[-1])

    periods = {

        "1m": 1440, "5m": 288, "15m": 96, "30m": 48,

        "1h": 24, "4h": 6, "1d": 1,

    }

    n = periods.get(st.session_state.interval, 1)

    if len(df) > n:

        prev = (

            float(df["close"].iloc[-(n + 1)])

            if len(df) > n + 1

            else float(df["close"].iloc[0])

        )

        st.session_state.price_change_24h = cur - prev

        st.session_state.price_change_24h_pct = (

            (st.session_state.price_change_24h / prev) * 100 if prev else 0

        )



# ──────────────────────────── chart & metrics ──────────────────────────────

def create_candlestick_chart(df: pd.DataFrame) -> go.Figure:

    if df.empty:

        return go.Figure().update_layout(

            template="plotly_dark",

            paper_bgcolor="#1e2130",

            plot_bgcolor="#1e2130",

            font=dict(color="white"),

        )


    fig = go.Figure()

    fig.add_trace(

        go.Candlestick(

            x=df.index,

            open=df["open"], high=df["high"],

            low=df["low"], close=df["close"],

            name="Price",

            increasing_line_color="#26a69a",

            decreasing_line_color="#ef5350",

        )

    )


    ema_colors = ["#ff9800", "#2196f3", "#9c27b0", "#e91e63"]

    for i, p in enumerate(st.session_state.ema_indicators):

        col = f"EMA_{p}"

        if col in df.columns:

            fig.add_trace(

                go.Scatter(

                    x=df.index, y=df[col], mode="lines",

                    line=dict(width=1.5, color=ema_colors[i % len(ema_colors)]),

                    name=f"EMA {p}",

                )

            )


    fig.add_trace(

        go.Bar(

            x=df.index, y=df["volume"], name="Volume",

            marker_color="rgba(0, 0, 255, 0.3)",

            opacity=0.3, yaxis="y2",

        )

    )


    fig.update_layout(

        title=f"{st.session_state.pair} {st.session_state.interval} Candlestick Chart",

        xaxis_title="Date",

        yaxis_title="Price (USDT)",

        xaxis_rangeslider_visible=False,

        margin=dict(l=50, r=50, t=85, b=50),

        height=600,

        template="plotly_dark",

        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),

        yaxis=dict(domain=[0.2, 1], gridcolor="rgba(255, 255, 255, 0.1)"),

        yaxis2=dict(title="Volume", domain=[0, 0.2]),

        font=dict(color="white"),

        plot_bgcolor="#1e2130",

        paper_bgcolor="#1e2130",

    )

    return fig



def display_metrics(df: pd.DataFrame):

    if df.empty:

        return

    cur = float(st.session_state.live_price or 0)

    prev = float(st.session_state.prev_price or cur)

    direction = "up" if cur >= prev else "down"

    change = cur - prev

    change_pct = (change / prev) * 100 if prev else 0

    cls = "positive" if change >= 0 else "negative"

    sym = "+" if change >= 0 else ""


    cols = st.columns([1] + [1] * len(st.session_state.ema_indicators))

    with cols[0]:

        st.markdown(

            f"""

<div class="metric-card {direction}">

    <div class="metric-title">Current Price</div>

    <div class="metric-value">${cur:.6f}</div>

    <div class="change-value {cls}">{sym}{change_pct:.2f}%</div>

</div>""",

            unsafe_allow_html=True,

        )


    for i, p in enumerate(st.session_state.ema_indicators):

        col_name = f"EMA_{p}"

        if col_name not in df.columns:

            continue

        ema_val = float(df[col_name].iloc[-1])

        diff = cur - ema_val

        pct = (diff / ema_val) * 100 if ema_val else 0

        c = "positive" if diff >= 0 else "negative"

        s = "+" if diff >= 0 else ""

        with cols[i + 1]:

            st.markdown(

                f"""

<div class="metric-card">

    <div class="metric-title">Price vs EMA-{p}</div>

    <div class="metric-value">${ema_val:.6f}</div>

    <div class="change-value {c}">{s}{pct:.2f}%</div>

</div>""",

                unsafe_allow_html=True,

            )



# ──────────────────────────── live panel body ──────────────────────────────

def live_panel_body():

    sync_state()

    calculate_24h_change()

    with metrics_placeholder.container():

        display_metrics(st.session_state.chart_data)

    with chart_placeholder.container():

        fig = create_candlestick_chart(st.session_state.chart_data)

        st.plotly_chart(fig, use_container_width=True)

    tick_str = (

        last_tick_global.strftime("%H:%M:%S") if last_tick_global else "—"

    )

    ws_status = "🟢 connected" if ws_connected else "🔴 reconnecting…"

    status_placeholder.markdown(

        f"<div style='text-align:center;font-size:.8rem;color:#b0b0b0;'>"

        f"{st.session_state.status_message} &nbsp;|&nbsp; "

        f"Last tick: {tick_str} &nbsp;|&nbsp; WS: {ws_status}</div>",

        unsafe_allow_html=True,

    )



# ──────────────────────────── Streamlit version detection ───────────────────

try:

    _ST_VER = tuple(int(x) for x in st.__version__.split(".")[:2])

except Exception:

    _ST_VER = (0, 0)

_HAS_FRAGMENT_RUN_EVERY = _ST_VER >= (1, 34)



# ──────────────────────────── UI: placeholders ──────────────────────────────

header_placeholder = st.empty()

metrics_placeholder = st.empty()

chart_placeholder = st.empty()

status_placeholder = st.empty()



# ──────────────────────────── sidebar ──────────────────────────────────────

st.sidebar.title("Solana Chart Settings")


PAIR_OPTIONS = ["SOLUSDT", "SOLBUSD", "SOLBTC", "SOLETH"]

try:

    _default_pair_idx = PAIR_OPTIONS.index(st.session_state.pair)

except ValueError:

    _default_pair_idx = 0

trading_pair = st.sidebar.selectbox(

    "Trading Pair", PAIR_OPTIONS, index=_default_pair_idx

)

if trading_pair != st.session_state.pair:

    st.session_state.pair = trading_pair

    st.session_state.data_loaded_key = None  # force reload on next pass


TF_LABELS = {

    "1 minute": "1m", "5 minutes": "5m", "15 minutes": "15m",

    "30 minutes": "30m", "1 hour": "1h", "4 hours": "4h", "1 day": "1d",

}

_INV_TF = {v: k for k, v in TF_LABELS.items()}

try:

    _default_tf_idx = list(TF_LABELS.keys()).index(

        _INV_TF.get(st.session_state.interval, "1 day")

    )

except ValueError:

    _default_tf_idx = 6

selected_tf = st.sidebar.selectbox(

    "Timeframe", list(TF_LABELS.keys()), index=_default_tf_idx

)

new_interval = TF_LABELS[selected_tf]

if new_interval != st.session_state.interval:

    st.session_state.interval = new_interval

    st.session_state.data_loaded_key = None


ema_indicators = st.sidebar.multiselect(

    "EMA Indicators",

    options=[20, 50, 100, 200],

    default=st.session_state.ema_indicators,

)

if ema_indicators != st.session_state.ema_indicators:

    st.session_state.ema_indicators = ema_indicators

    # Recompute EMAs locally — no REST call needed.

    df = st.session_state.chart_data

    if not df.empty:

        df = add_ema(df.copy(), ema_indicators)

        with _state_lock:

            chart_data_global = df


with st.sidebar.expander("Advanced Options"):

    update_frequency = st.slider(

        "Update Frequency (seconds)",

        min_value=1, max_value=60,

        value=int(st.session_state.update_frequency),

    )

    st.session_state.update_frequency = update_frequency


    days_to_fetch = st.slider(

        "Historical Data (days)",

        min_value=7, max_value=360,

        value=int(st.session_state.days_to_fetch),

    )

    if days_to_fetch != st.session_state.days_to_fetch:

        st.session_state.days_to_fetch = days_to_fetch

        st.session_state.data_loaded_key = None


st.sidebar.markdown('<div class="button-container">', unsafe_allow_html=True)

if st.sidebar.button("Fetch Data"):

    load_initial_data(force=True)

    _start_websocket_internal(st.session_state.pair, st.session_state.interval)

    st.rerun()

st.sidebar.markdown("</div>", unsafe_allow_html=True)



# ──────────────────────────── header ───────────────────────────────────────

header_placeholder.markdown(

    "<h1 style='text-align: center;'>Solana Live Candlestick Chart</h1>",

    unsafe_allow_html=True,

)



# ──────────────────────────── one-time work (guarded by data_loaded_key) ───

load_initial_data()


# Open / refresh the WebSocket whenever the pair or interval changes.

if (

    ws_app is None

    or current_pair != st.session_state.pair.lower()

    or current_interval != st.session_state.interval

):

    _start_websocket_internal(st.session_state.pair, st.session_state.interval)



# ──────────────────────────── live panel: re-renders without REST ───────────

# This is the second key fix: the panel re-renders every `update_frequency`

# seconds WITHOUT calling Binance REST. The data is already in `chart_data`

# (synced from the WebSocket-updated global).

if _HAS_FRAGMENT_RUN_EVERY:

    @st.fragment(run_every=timedelta(seconds=update_frequency))

    def _live_fragment():

        live_panel_body()

    _live_fragment()

else:

    try:

        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(

            interval=update_frequency * 1000,

            key="binance_autorefresh",

        )

    except ImportError:

        st.warning(

            "Live auto-refresh needs Streamlit ≥ 1.34, "

            "or `pip install streamlit-autorefresh`."

        )

    live_panel_body()



# ──────────────────────────── static data table ────────────────────────────

st.subheader("Historical Data Table (OHLCV only)")

ohlcv_cols = ["open", "high", "low", "close", "volume"]

if not st.session_state.chart_data.empty:

    st.dataframe(

        st.session_state.chart_data[ohlcv_cols].iloc[::-1],

        use_container_width=True,

    )

else:

    st.info(

        "No data loaded yet. If Binance throttled this server's IP (code -1003), "

        "the ban usually clears within 5–15 minutes — the live WebSocket tick "

        "stream will keep updating the chart in the meantime."

    )



# ──────────────────────────── shutdown cleanup ─────────────────────────────

atexit.register(stop_websocket)
