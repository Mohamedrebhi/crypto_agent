import streamlit as st
from langchain_community.llms import HuggingFaceHub
import yfinance as yf
from datetime import datetime, timedelta
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="AI Crypto Analyzer 🚀",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stHeader {
        background-color: rgba(0,0,0,0);
    }
    .css-1d391kg {
        padding: 1rem 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🚀 Advanced Crypto Analysis AI")
st.markdown("""
    <div style='background-color: #1c1c1c; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
    This AI-powered tool provides detailed cryptocurrency analysis and investment recommendations using advanced market metrics.
    </div>
""", unsafe_allow_html=True)

# Popular crypto symbols
POPULAR_CRYPTOS = {
    "Bitcoin": "BTC",
    "Ethereum": "ETH",
    "Binance Coin": "BNB",
    "Cardano": "ADA",
    "Solana": "SOL",
    "Ripple": "XRP",
    "Dogecoin": "DOGE",
    "Polkadot": "DOT",
    "Avalanche": "AVAX",
    "Chainlink": "LINK",
    "Polygon": "MATIC",
    "Uniswap": "UNI",
    "Litecoin": "LTC",
    "Bitcoin Cash": "BCH",
    "Stellar": "XLM"
}

def get_technical_indicators(hist_data):
    df = pd.DataFrame(hist_data)
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate Moving Averages
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    return df

def plot_crypto_data(symbol, hist_data):
    df = get_technical_indicators(hist_data)
    
    fig = make_subplots(rows=3, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       subplot_titles=(f'{symbol} Price & Volume', 'RSI', 'MACD'),
                       row_heights=[0.6, 0.2, 0.2])

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ), row=1, col=1)
    
    # Add Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA20'], name='SMA20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA50'], name='SMA50', line=dict(color='blue')), row=1, col=1)
    
    # Volume
    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], name='Signal', line=dict(color='orange')), row=3, col=1)
    
    fig.update_layout(
        height=800,
        template='plotly_dark',
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def get_crypto_info(symbol):
    try:
        crypto = yf.Ticker(f"{symbol}-USD")
        info = crypto.info
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)  # Extended to 90 days
        hist = crypto.history(start=start_date, end=end_date, interval='1d')
        
        return {
            'symbol': symbol,
            'info': info,
            'history': hist,
            'current_price': info.get('regularMarketPrice', 0),
            'market_cap': info.get('marketCap', 0),
            'volume_24h': info.get('volume24Hr', 0),
            'change_24h': info.get('regularMarketChangePercent', 0),
            'total_supply': info.get('totalSupply', 0),
            'circulating_supply': info.get('circulatingSupply', 0),
            'max_supply': info.get('maxSupply', 0)
        }
    except Exception as e:
        return f"Error getting data for {symbol}: {str(e)}"

# Sidebar
with st.sidebar:
    st.header("🔑 API Configuration")
    huggingface_api_key = st.text_input("Enter your Hugging Face API key:", type="password")
    
    st.header("📈 Cryptocurrency Selection")
    # Use selectbox with names, but store symbols
    crypto1_name = st.selectbox("Select first cryptocurrency:", list(POPULAR_CRYPTOS.keys()), index=0)
    crypto2_name = st.selectbox("Select second cryptocurrency:", list(POPULAR_CRYPTOS.keys()), index=1)
    
    crypto1 = POPULAR_CRYPTOS[crypto1_name]
    crypto2 = POPULAR_CRYPTOS[crypto2_name]

if huggingface_api_key:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
    
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={
            "temperature": 0.2,
            "max_new_tokens": 2048,
            "top_p": 0.9,
        }
    )

    if crypto1 and crypto2:
        with st.spinner(f"Analyzing {crypto1} and {crypto2}..."):
            try:
                # Get data for both cryptocurrencies
                data1 = get_crypto_info(crypto1)
                data2 = get_crypto_info(crypto2)
                
                if isinstance(data1, str) or isinstance(data2, str):
                    st.error("Error fetching cryptocurrency data")
                else:
                    # Create tabs for different analyses
                    tab1, tab2, tab3 = st.tabs(["📊 Market Data", "📈 Technical Analysis", "🤖 AI Analysis"])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        
                        # Display detailed market data for each crypto
                        with col1:
                            st.subheader(f"{crypto1} Market Data")
                            st.metric("Price", f"${data1['current_price']:,.2f}", 
                                    f"{data1['change_24h']:.2f}%")
                            st.metric("Market Cap", f"${data1['market_cap']:,.0f}")
                            st.metric("24h Volume", f"${data1['volume_24h']:,.0f}")
                            st.metric("Circulating Supply", f"{data1['circulating_supply']:,.0f}")
                        
                        with col2:
                            st.subheader(f"{crypto2} Market Data")
                            st.metric("Price", f"${data2['current_price']:,.2f}", 
                                    f"{data2['change_24h']:.2f}%")
                            st.metric("Market Cap", f"${data2['market_cap']:,.0f}")
                            st.metric("24h Volume", f"${data2['volume_24h']:,.0f}")
                            st.metric("Circulating Supply", f"{data2['circulating_supply']:,.0f}")
                    
                    with tab2:
                        st.subheader(f"{crypto1} Technical Analysis")
                        st.plotly_chart(plot_crypto_data(crypto1, data1['history']), use_container_width=True)
                        
                        st.subheader(f"{crypto2} Technical Analysis")
                        st.plotly_chart(plot_crypto_data(crypto2, data2['history']), use_container_width=True)
                    
                    with tab3:
                        st.subheader("AI Analysis and Recommendation")
                        
                        analysis_prompt = f"""
                        Analyze these two cryptocurrencies and provide a detailed investment recommendation:

                        {crypto1}:
                        - Price: ${data1['current_price']:,.2f}
                        - Market Cap: ${data1['market_cap']:,.2f}
                        - 24h Volume: ${data1['volume_24h']:,.2f}
                        - 24h Change: {data1['change_24h']:.2f}%
                        - Circulating Supply: {data1['circulating_supply']:,.0f}

                        {crypto2}:
                        - Price: ${data2['current_price']:,.2f}
                        - Market Cap: ${data2['market_cap']:,.2f}
                        - 24h Volume: ${data2['volume_24h']:,.2f}
                        - 24h Change: {data2['change_24h']:.2f}%
                        - Circulating Supply: {data2['circulating_supply']:,.0f}

                        Please provide:
                        1. Detailed market position analysis
                        2. Volume and liquidity comparison
                        3. Price movement analysis
                        4. Risk assessment
                        5. Clear investment recommendation with reasoning

                        Format in markdown with headers and bullet points.
                        """
                        
                        with st.spinner("Generating AI analysis..."):
                            analysis = llm.invoke(analysis_prompt)
                            st.markdown(analysis)
                
                # Add disclaimer
                st.markdown("""
                    <div style='background-color: #1c1c1c; padding: 1rem; border-radius: 10px; margin-top: 2rem;'>
                    ⚠️ <b>Disclaimer:</b> This analysis is for informational purposes only. Cryptocurrency investments carry high risk. 
                    Always conduct your own research before making investment decisions.
                    </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
                st.info("Please try again or check your API key.")
else:
    st.warning("Please enter your Hugging Face API key to start the analysis.") 