import streamlit as st
import pandas as pd
import numpy as np
import time
import threading
import json
from datetime import datetime, timedelta
from binance.client import Client
import websocket
import plotly.graph_objects as go
import logging
import traceback
import warnings
import os

live_price_global = 0.0
prev_price_global = 0.0
chart_data_global = pd.DataFrame()

# 1. Suppress all Python warnings
warnings.filterwarnings("ignore")
# 2. Silence the websocket‐client logger
logging.getLogger("websocket").setLevel(logging.ERROR)
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Set page config
st.set_page_config(
    page_title="Solana Live Candlestick Chart",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Apply custom CSS for dark theme and UI improvements
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        padding-top: 0;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0;
    }
    .stPlotlyChart {
        height: 70vh !important;
    }
    .metric-card {
        background-color: #151a28;
        color: white;
        padding: 15px;
        border-radius: 5px;
        text-align: center;
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
    .metric-title {
        font-size: 0.9rem;
        font-weight: 400;
        margin-bottom: 5px;
        color: #b0b0b0;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    .change-value {
        font-size: 0.9rem;
    }
    .change-value.positive {
        color: #26a69a;
    }
    .change-value.negative {
        color: #ef5350;
    }
    .stButton>button {
        background-color: #26a69a;
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        border-radius: 4px;
        cursor: pointer;
        display: block;
        margin: 0 auto;
        width: auto;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #2bbbad;
    }
    .stSlider>div>div>div {
        background-color: #26a69a;
    }
    div[data-testid="stSidebar"] {
        background-color: #0b0e14;
        padding: 2rem 1rem;
    }
    div[data-testid="stSidebar"] .stSelectbox label, 
    div[data-testid="stSidebar"] .stMultiselect label,
    div[data-testid="stSidebar"] .stSlider label {
        color: #e0e0e0;
    }
    div[data-testid="stSidebar"] .stSelectbox>div>div {
        background-color: #1e2130;
        color: white;
        border: 1px solid #2c3246;
    }
    div[data-testid="stSidebar"] .stMultiselect>div>div {
        background-color: #1e2130;
        color: white;
        border: 1px solid #2c3246;
    }
    div[data-testid="stSidebar"] h1, 
    div[data-testid="stSidebar"] h2, 
    div[data-testid="stSidebar"] h3 {
        color: #e0e0e0;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #2c3246;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        color: white;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        border: 1px solid #2c3246;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #26a69a !important;
        color: white !important;
    }
    /* Dark mode for Streamlit tables */
    .stDataFrame table, .stTable table {
        background-color: #1e2130 !important;
        color: white !important;
    }
    .stDataFrame th, .stTable th {
        background-color: #1e2130 !important;
        color: white !important;
    }
    .stDataFrame td, .stTable td {
        color: white !important;
    }
    /* Button container styling */
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
        margin-bottom: 20px;
    }
    /* Metrics container styling */
    .metrics-container {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)
# Initialize session state variables if they don't exist
if 'chart_data' not in st.session_state:
    st.session_state.chart_data = pd.DataFrame()
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'live_price' not in st.session_state:
    st.session_state.live_price = 0
if 'prev_price' not in st.session_state:
    st.session_state.prev_price = 0
if 'ws_client' not in st.session_state:
    st.session_state.ws_client = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'interval' not in st.session_state:
    st.session_state.interval = '1d'
if 'days_to_fetch' not in st.session_state:
    st.session_state.days_to_fetch = 360
if 'ws_thread' not in st.session_state:
    st.session_state.ws_thread = None
if 'price_change_24h' not in st.session_state:
    st.session_state.price_change_24h = 0
if 'price_change_24h_pct' not in st.session_state:
    st.session_state.price_change_24h_pct = 0
if 'refresh_count' not in st.session_state:
    st.session_state.refresh_count = 0
# Create placeholders for dynamic content
header_placeholder = st.empty()
metrics_placeholder = st.container()
chart_placeholder = st.empty()
status_placeholder = st.empty()
button_placeholder = st.empty()
# Sidebar configuration
st.sidebar.title("Solana Chart Settings")
# Trading pair selection
trading_pair = st.sidebar.selectbox(
    "Trading Pair",
    options=["SOLUSDT", "SOLBUSD", "SOLBTC", "SOLETH"],
    index=0
)
# Timeframe selection
timeframe_options = {
    "1 minute": "1m",
    "5 minutes": "5m",
    "15 minutes": "15m",
    "30 minutes": "30m",
    "1 hour": "1h",
    "4 hours": "4h",
    "1 day": "1d"
}
selected_timeframe = st.sidebar.selectbox(
    "Timeframe",
    options=list(timeframe_options.keys()),
    index=6  # Default to 1d
)
interval = timeframe_options[selected_timeframe]
# Update interval if changed
if interval != st.session_state.interval:
    st.session_state.interval = interval
    # Reset data when interval changes
    st.session_state.chart_data = pd.DataFrame()
# EMA settings
ema_indicators = st.sidebar.multiselect(
    "EMA Indicators",
    options=[20, 50, 100, 200],
    default=[20, 50, 100, 200]
)
# Advanced options
with st.sidebar.expander("Advanced Options"):
    update_frequency = st.slider(
        "Update Frequency (seconds)",
        min_value=1,
        max_value=60,
        value=5
    )
    
    days_to_fetch = st.slider(
        "Historical Data (days)",
        min_value=7,
        max_value=360,
        value=360
    )
    # Update days to fetch if changed
    if days_to_fetch != st.session_state.days_to_fetch:
        st.session_state.days_to_fetch = days_to_fetch
        # Reset data when days changes
        st.session_state.chart_data = pd.DataFrame()
# Binance API credentials
# API_KEY = st.secrets.get("API_KEY", "")
# API_SECRET = st.secrets.get("API_SECRET", "")
API_KEY = os.environ.get("API_KEY", "")
API_SECRET = os.environ.get("API_SECRET", "")
# Initialize Binance client with or without credentials
@st.cache_resource
def get_binance_client():
    try:
        # Check if keys were loaded from secrets
        if API_KEY and API_SECRET:
            logger.info("Initializing Binance client with API keys from secrets.")
            return Client(API_KEY, API_SECRET)
        else:
            logger.info("Initializing Binance client without API keys (credentials not found in secrets).")
            return Client("", "")
    except Exception as e:
        logger.error(f"Failed to initialize Binance client: {e}")
        st.sidebar.error(f"Failed to initialize Binance client: {e}")
        # Return a client with no keys as fallback
        return Client("", "")

def get_historical_klines(symbol, interval, days):
    """
    Fetch historical klines (candlestick) data from Binance.
    
    :param symbol: Trading pair symbol (e.g., 'SOLUSDT')
    :param interval: Timeframe for candlesticks (e.g., '1h', '1d')
    :param days: Number of days to look back
    :return: Pandas DataFrame with OHLCV data
    """
    try:
        logger.info(f"Fetching {days} days of historical data for {symbol} at {interval} interval")
        lookback = f"{days} days ago UTC"
        klines = client.get_historical_klines(symbol, interval, lookback)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Keep only necessary columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        logger.info(f"Successfully fetched {len(df)} data points")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        logger.error(traceback.format_exc())
        raise Exception(f"Error fetching historical data: {e}")
def add_ema(df, periods):
    """
    Add Exponential Moving Averages (EMAs) to the DataFrame.
    
    :param df: DataFrame with price data
    :param periods: List of periods for EMAs
    :return: DataFrame with added EMA columns
    """
    if df.empty:
        return df
        
    for period in periods:
        df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    return df
def on_ws_message(ws, message):
    """
    Handle WebSocket message for real-time price updates.
    """
    global prev_price_global, live_price_global, chart_data_global
    if not st.session_state.running:
        return
        
    try:
        data = json.loads(message)
        
        if 'k' in data:
            kline = data['k']
            
            # Get the current candle timestamp
            timestamp = datetime.fromtimestamp(kline['t'] / 1000)
            
            # Update session state with the latest price
            # st.session_state.prev_price = st.session_state.live_price
            # st.session_state.live_price = float(kline['c'])
            prev_price_global = live_price_global
            live_price_global = float(kline['c'])
            # Optionally: if you want to update the session_state in main thread (only if safe)
            # st.session_state['live_price'] = live_price_global
            # st.session_state['prev_price'] = prev_price_global
            print(f"Received new price: {live_price_global}")
            print(f"Previous price: {prev_price_global}")
            # If the candle is closed, update our historical data
            if kline['x']:
                # We need to fetch the latest data again when a candle closes
                update_chart_data()
            else:
                # Update the last candle in our existing data
                if not chart_data_global.empty:
                    last_timestamp = chart_data_global.index[-1]
                    
                    # Only update if we have this timestamp in our data
                    if timestamp.strftime('%Y-%m-%d %H:%M:%S') == last_timestamp.strftime('%Y-%m-%d %H:%M:%S'):
                        chart_data_global.chart_data.at[last_timestamp, 'open'] = float(kline['o'])
                        chart_data_global.chart_data.at[last_timestamp, 'high'] = float(kline['h'])
                        chart_data_global.chart_data.at[last_timestamp, 'low'] = float(kline['l'])
                        chart_data_global.chart_data.at[last_timestamp, 'close'] = float(kline['c'])
                        chart_data_global.chart_data.at[last_timestamp, 'volume'] = float(kline['v'])
                        
                        # Recalculate EMAs for the updated data
                        chart_data_global = add_ema(chart_data_global, ema_indicators)
            
            chart_data_global.last_update = datetime.now()
            
            # Calculate 24h change
            calculate_24h_change()
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        logger.error(traceback.format_exc())
def on_ws_error(ws, error):
    logger.error(f"WebSocket error: {error}")
    
def on_ws_close(ws, close_status_code, close_msg):
    logger.info(f"WebSocket connection closed: {close_status_code} - {close_msg}")
    st.session_state.running = False
    
def on_ws_open(ws):
    logger.info(f"WebSocket connection opened for {trading_pair.lower()}@kline_{interval}")
    st.session_state.running = True
def start_websocket():
    """
    Start WebSocket connection for real-time data.
    """
    try:
        # Close any existing connection
        if st.session_state.ws_client is not None:
            try:
                st.session_state.ws_client.close()
                logger.info("Closed existing WebSocket connection")
            except Exception as e:
                logger.error(f"Error closing existing WebSocket: {e}")
                
        # Create a new WebSocket connection
        socket_url = f"wss://stream.binance.com:9443/ws/{trading_pair.lower()}@kline_{interval}"
        logger.info(f"Starting WebSocket connection to {socket_url}")
        
        # Create and configure WebSocket
        ws = websocket.WebSocketApp(
            socket_url,
            on_message=on_ws_message,
            on_error=on_ws_error,
            on_close=on_ws_close,
            on_open=on_ws_open
        )
        
        # Start WebSocket connection in a separate thread
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        st.session_state.ws_client = ws
        st.session_state.ws_thread = ws_thread
        logger.info("WebSocket thread started")
        
    except Exception as e:
        logger.error(f"Failed to start WebSocket: {e}")
        logger.error(traceback.format_exc())
        status_placeholder.error(f"Failed to start WebSocket: {e}")
def stop_websocket():
    """
    Stop WebSocket connection.
    """
    if st.session_state.ws_client is not None:
        try:
            st.session_state.ws_client.close()
            st.session_state.ws_client = None
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
    st.session_state.running = False
def calculate_24h_change():
    """
    Calculate the 24-hour price change and percentage.
    """
    df = chart_data_global
    if df.empty:
        return
    
    current_price = st.session_state.live_price
    
    # Find data from 24 hours ago (using the interval)
    if interval == '1m':
        periods_in_day = 24 * 60
    elif interval == '5m':
        periods_in_day = 24 * 12
    elif interval == '15m':
        periods_in_day = 24 * 4
    elif interval == '30m':
        periods_in_day = 24 * 2
    elif interval == '1h':
        periods_in_day = 24
    elif interval == '4h':
        periods_in_day = 6
    else:  # 1d
        periods_in_day = 1
    
    if len(df) > periods_in_day:
        price_24h_ago = df['close'].iloc[-periods_in_day-1] if len(df) > periods_in_day + 1 else df['close'].iloc[0]
        st.session_state.price_change_24h = current_price - price_24h_ago
        st.session_state.price_change_24h_pct = (st.session_state.price_change_24h / price_24h_ago) * 100 if price_24h_ago != 0 else 0
def update_chart_data():
    """
    Update the chart data from Binance API.
    """
    try:
        # Fetch historical data
        df = get_historical_klines(trading_pair, interval, st.session_state.days_to_fetch)
        
        # Add EMAs to the data
        df = add_ema(df, ema_indicators)
        
        # Update session state
        st.session_state.chart_data = df
        st.session_state.live_price = df['close'].iloc[-1]
        st.session_state.prev_price = df['close'].iloc[-2] if len(df) > 1 else st.session_state.live_price
        
        # Calculate 24h change
        calculate_24h_change()
        
        # Update last update time
        st.session_state.last_update = datetime.now()
        st.session_state.refresh_count += 1
        
        logger.info(f"Chart data updated successfully. Current price: {st.session_state.live_price}")
        
    except Exception as e:
        logger.error(f"Error updating chart data: {e}")
        logger.error(traceback.format_exc())
        status_placeholder.error(f"Error updating chart data: {e}")
def create_candlestick_chart(df):
    """
    Create an interactive Plotly candlestick chart with EMAs.
    
    :param df: DataFrame with price and EMA data
    :return: Plotly figure
    """
    if df.empty:
        return go.Figure()
    
    # Create figure
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Candlesticks',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    
    # Add EMAs
    colors = ['#ff9800', '#2196f3', '#9c27b0', '#e91e63']  # Different colors for different EMAs
    for i, period in enumerate(ema_indicators):
        ema_col = f'EMA_{period}'
        if ema_col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[ema_col],
                mode='lines',
                line=dict(width=1.5, color=colors[i % len(colors)]),
                name=f'EMA {period}'
            ))
    
    # Add volume as a bar chart
    colors_volume = np.where(df['close'] >= df['open'], '#26a69a', '#ef5350')
    
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volume',
        marker_color='rgba(0, 0, 255, 0.3)',
        opacity=0.3,
        yaxis='y2'
    ))
    
    # Customize layout
    fig.update_layout(
        title=f"{trading_pair} {selected_timeframe} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price (USDT)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=85, b=50),
        height=600,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        yaxis=dict(
            domain=[0.2, 1],
            side="left",
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        yaxis2=dict(
            title="Volume",
            domain=[0, 0.2],
            side="right",
            showgrid=False
        ),
        font=dict(
            color="white"
        ),
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
    )
    
    # Add trading tools
    fig.update_layout(
        modebar_add=[
            'drawline',
            'drawopenpath',
            'drawrect',
            'eraseshape'
        ]
    )
    
    return fig
def display_metrics(df):
    """
    Display key metrics in a dashboard style.
    
    :param df: DataFrame with price data
    """
    if df.empty:
        return
    
    # Get current values
    current_price = st.session_state.live_price
    previous_close = st.session_state.prev_price
    
    # Get all EMA values
    ema_values = {}
    for period in ema_indicators:
        ema_col = f'EMA_{period}'
        if ema_col in df.columns:
            ema_values[period] = df[ema_col].iloc[-1]
    
    # Price direction (up or down)
    price_direction = "up" if current_price >= previous_close else "down"
    
    # Display current price in the first card
    col1, *ema_cols = st.columns([1] + [1 for _ in range(len(ema_indicators))])
    
    with col1:
        price_change = current_price - previous_close
        price_change_pct = (price_change / previous_close) * 100 if previous_close != 0 else 0
        change_class = "positive" if price_change >= 0 else "negative"
        change_symbol = "+" if price_change >= 0 else ""
        
        st.markdown(f"""
        <div class="metric-card {price_direction}">
            <div class="metric-title">Current Price</div>
            <div class="metric-value">${current_price:.6f}</div>
            <div class="change-value {change_class}">{change_symbol}{price_change_pct:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Display EMA comparisons
    for i, period in enumerate(ema_indicators):
        if period in ema_values:
            with ema_cols[i]:
                ema_val = ema_values[period]
                price_vs_ema = current_price - ema_val
                price_vs_ema_pct = (price_vs_ema / ema_val) * 100 if ema_val != 0 else 0
                ema_class = "positive" if price_vs_ema >= 0 else "negative"
                ema_symbol = "+" if price_vs_ema >= 0 else ""
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Price vs EMA-{period}</div>
                    <div class="metric-value">${ema_val:.6f}</div>
                    <div class="change-value {ema_class}">{ema_symbol}{price_vs_ema_pct:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
def main():
    """
    Main function to run the Streamlit app.
    """
    # Display app title
    header_placeholder.markdown(
        "<h1 style='text-align: center;'>Solana Live Candlestick Chart</h1>",
        unsafe_allow_html=True
    )
    
    # Display metrics in the metrics container
    with metrics_placeholder.container():
        display_metrics(st.session_state.chart_data)
    
    # Create and display chart
    with chart_placeholder.container():
        fig = create_candlestick_chart(st.session_state.chart_data)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show the historical data table
    st.subheader("Historical Data Table (OHLCV only)")
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    st.dataframe(
        st.session_state.chart_data[ohlcv_cols],
        use_container_width=True,
    )
    
    # Add a button below the chart to manually fetch data
    # Using a custom container for better button styling
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("Fetch Data"):
        with st.spinner('Fetching latest data...'):
            update_chart_data()
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display last update time
    time_diff = (datetime.now() - st.session_state.last_update).total_seconds()
    status_placeholder.markdown(
        f"<div style='text-align: center; font-size: 0.8rem; color: #b0b0b0;'>"
        f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')} "
        f"({int(time_diff)} seconds ago)</div>",
        unsafe_allow_html=True
    )
    
    # Check if we need to update data
    if st.session_state.running and time.time() % update_frequency < 1:
        logger.info(f"Auto-updating data (refresh count: {st.session_state.refresh_count + 1})")
        update_chart_data()
        st.rerun()
    # 4. Auto‐refresh the entire page every `update_frequency` seconds
    st.markdown(
        f"""
        <script>
            setTimeout(function(){{
                window.location.reload();
            }}, {update_frequency * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )
# Initialize WebSocket and data on startup
def initialize():
    """
    Initialize the application on startup.
    """
    # Initialize session state values if not already set
    
    if st.session_state.chart_data.empty:
        try:
            logger.info("Initializing application and fetching initial data")
            # Initial data fetch
            update_chart_data()
            
            # Start WebSocket connection
            if not st.session_state.running:
                start_websocket()
                
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            logger.error(traceback.format_exc())
            st.error(f"Initialization error: {e}")
# Cleanup on session end
def cleanup():
    """
    Clean up resources when the application stops.
    """
    logger.info("Cleaning up resources")
    stop_websocket()
# Register the cleanup function
import atexit
atexit.register(cleanup)
# Run the application
if __name__ == "__main__":
    try:
        client = get_binance_client()
        initialize()
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Application error: {e}")
