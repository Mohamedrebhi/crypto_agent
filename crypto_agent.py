import streamlit as st
from langchain_community.llms import HuggingFaceHub
import yfinance as yf
from datetime import datetime, timedelta
import os

st.title("AI Crypto Agent 🚀💹")
st.caption("This app allows you to compare cryptocurrencies and generate detailed analysis with price predictions.")

huggingface_api_key = st.text_input("Enter your Hugging Face API key:", type="password")

if huggingface_api_key:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_api_key
    
    # Initialize Hugging Face model
    llm = HuggingFaceHub(
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        model_kwargs={
            "temperature": 0.2,
            "max_new_tokens": 2048,
            "top_p": 0.9,
        }
    )
    
    def get_crypto_data(symbol1, symbol2):
        try:
            # Get data for first crypto
            crypto1 = yf.Ticker(f"{symbol1}-USD")
            info1 = crypto1.info
            
            # Get data for second crypto
            crypto2 = yf.Ticker(f"{symbol2}-USD")
            info2 = crypto2.info
            
            analysis_prompt = f"""
            Analyze these two cryptocurrencies and recommend which is better for investment:

            {symbol1}:
            - Current Price: ${info1.get('regularMarketPrice', 0):,.2f}
            - Market Cap: ${info1.get('marketCap', 0):,.2f}
            - 24h Volume: ${info1.get('volume24Hr', 0):,.2f}
            - 24h Change: {info1.get('regularMarketChangePercent', 0):.2f}%

            {symbol2}:
            - Current Price: ${info2.get('regularMarketPrice', 0):,.2f}
            - Market Cap: ${info2.get('marketCap', 0):,.2f}
            - 24h Volume: ${info2.get('volume24Hr', 0):,.2f}
            - 24h Change: {info2.get('regularMarketChangePercent', 0):.2f}%

            Please provide a detailed analysis including:
            1. Market Position Comparison
            2. Volume Analysis
            3. Price Movement Analysis
            4. Risk Assessment
            5. Clear Investment Recommendation

            Format your response in markdown with headers and bullet points.
            Conclude with a clear recommendation on which cryptocurrency shows better investment potential and why.
            """
            
            # Get AI analysis
            analysis = llm.invoke(analysis_prompt)
            
            return {
                'raw_data': {
                    symbol1: {
                        'price': info1.get('regularMarketPrice', 0),
                        'market_cap': info1.get('marketCap', 0),
                        'volume': info1.get('volume24Hr', 0),
                        'change': info1.get('regularMarketChangePercent', 0)
                    },
                    symbol2: {
                        'price': info2.get('regularMarketPrice', 0),
                        'market_cap': info2.get('marketCap', 0),
                        'volume': info2.get('volume24Hr', 0),
                        'change': info2.get('regularMarketChangePercent', 0)
                    }
                },
                'analysis': analysis
            }
            
        except Exception as e:
            return f"Error analyzing cryptocurrencies: {str(e)}"

    col1, col2 = st.columns(2)
    with col1:
        crypto1 = st.text_input("Enter first crypto symbol (e.g. BTC)")
    with col2:
        crypto2 = st.text_input("Enter second crypto symbol (e.g. ETH)")

    if crypto1 and crypto2:
        with st.spinner(f"Analyzing {crypto1} and {crypto2}..."):
            try:
                # Get data and AI analysis
                result = get_crypto_data(crypto1.upper(), crypto2.upper())
                
                if isinstance(result, str):  # Error occurred
                    st.error(result)
                else:
                    # Display raw data
                    st.subheader("Current Market Data:")
                    col1, col2 = st.columns(2)
                    
                    # Format and display data for first crypto
                    with col1:
                        data1 = result['raw_data'][crypto1.upper()]
                        st.markdown(f"""
                        **{crypto1.upper()}**
                        - Price: ${data1['price']:,.2f}
                        - Market Cap: ${data1['market_cap']:,.2f}
                        - 24h Volume: ${data1['volume']:,.2f}
                        - 24h Change: {data1['change']:.2f}%
                        """)
                    
                    # Format and display data for second crypto
                    with col2:
                        data2 = result['raw_data'][crypto2.upper()]
                        st.markdown(f"""
                        **{crypto2.upper()}**
                        - Price: ${data2['price']:,.2f}
                        - Market Cap: ${data2['market_cap']:,.2f}
                        - 24h Volume: ${data2['volume']:,.2f}
                        - 24h Change: {data2['change']:.2f}%
                        """)
                    
                    # Display AI analysis
                    st.subheader("AI Analysis and Recommendation:")
                    st.markdown(result['analysis'])
                
                # Add disclaimer
                st.caption("⚠️ Disclaimer: This analysis is for informational purposes only. Cryptocurrency investments carry high risk.")
                
            except Exception as e:
                st.error(f"Analysis Error: {str(e)}")
                st.info("Please try again with different crypto symbols or check your API key.")
                st.caption("Note: Use valid cryptocurrency symbols (e.g., BTC, ETH, USDT)") 