import streamlit as st
import pandas as pd
from bot import TradeHunter

Hunter = TradeHunter()

# Getting decisions from the bot
data = Hunter.run()

# st.write(data[0])

# Convert to DataFrame for table use (with relevant decision data)
df = pd.DataFrame([{
    "Ticker": d[0]["ticker"],
    "Action": d[1]["action"],
    "Confidence": d[1]["confidence"],
    "Stop Loss": f"{d[1]['stop_loss']:.2f}" if d[1]['stop_loss'] else "N/A",
    "Target": f"{d[1]['target']:.2f}" if d[1]['target'] else "N/A",
    "Reasons": '\n'.join(d[1]['reasons']),
    "RSI": d[0]['indicators']['RSI'],
    "MACD": d[0]['indicators']['MACD'],
    "Volatility": d[0]['volatility'],
} for d in data])

# UI
st.set_page_config(page_title="Trade Hunter", layout="wide")
st.title("📈 Trade Hunter")

tab1, tab2 = st.tabs(["📊 Cards View", "📋 Table View"])

with tab1:
    st.subheader("Signals (Card View)")
    cols = st.columns(3)
    for i, (analysis, decision) in enumerate(data):
        signal_color = {
            "BUY": "limegreen",
            "SELL": "red",
            "HOLD": "white"
        }.get(decision['action'], "white")

        with cols[i % 3]:
            st.markdown(f"""
            <div style="background-color:#1C2027;padding:15px;border-radius:10px;margin-bottom:10px;color:white">
                <h4 style="color:{signal_color}">{analysis['ticker']} - {decision['action']} ({decision['confidence']})</h4>
                <b>Price:</b> {analysis['current_price']:.2f}<br>
                <b>RSI:</b> {analysis['indicators']['RSI']:.2f}<br>
                <b>MACD:</b> {analysis['indicators']['MACD']:.2f}<br>
                <b>Volatility:</b> {analysis['volatility']:.4f}<br>
                <b>Stop Loss:</b> {decision['stop_loss'] if decision['stop_loss'] else 'N/A'}<br>
                <b>Target:</b> {decision['target'] if decision['target'] else 'N/A'}<br>
                <b>Reasons:</b><br>{'<br>'.join(f'• {reason}' for reason in decision['reasons'])}
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("Signals (Table View)")
    st.dataframe(df, use_container_width=True)
