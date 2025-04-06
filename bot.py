from scraping import scrapDataHistory, getCurrentData
from DEFAULT import STOCKS
import pandas as pd
import numpy as np
from datetime import datetime, date
import time
from typing import Dict, List, Union

# from rich.progress import Progress

import logging

import os
import shutil

class TradeHunter:
    def __init__(self):
        self.current_time = datetime.now()
        
        # self.telegram_notifier = TelegramNotifier(token=os.getenv("TELEGRAM_BOT_TOKEN"), chat_id=os.getenv("TELEGRAM_CHAT_ID"))
        
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        results = {}
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        results['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        results['MACD'] = exp1 - exp2
        results['Signal_Line'] = results['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma = data['Close'].rolling(window=20).mean()
        std = data['Close'].rolling(window=20).std()
        results['BB_Upper'] = sma + (std * 2)
        results['BB_Lower'] = sma - (std * 2)
        results['BB_Middle'] = sma
        
        # Moving Averages
        results['SMA_50'] = data['Close'].rolling(window=50).mean()
        results['SMA_200'] = data['Close'].rolling(window=200).mean()
        results['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        return results

    def find_support_resistance(self, data: pd.DataFrame, window: int = 20) -> tuple:
        """Find support and resistance levels"""
        highs = data['High'].rolling(window=window, center=True).max()
        lows = data['Low'].rolling(window=window, center=True).min()
        
        support_levels = set()
        resistance_levels = set()
        
        for i in range(window, len(data) - window):
            if lows.iloc[i] == min(lows.iloc[i-window:i+window]):
                support_levels.add(round(lows.iloc[i], 2))
            if highs.iloc[i] == max(highs.iloc[i-window:i+window]):
                resistance_levels.add(round(highs.iloc[i], 2))
        
        return sorted(list(support_levels)), sorted(list(resistance_levels))

    def analyze_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        avg_volume = data['Volume'].mean()
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        return {
            'avg_volume': avg_volume,
            'current_volume': current_volume,
            'volume_ratio': volume_ratio
        }
    
    def analyze_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """Analyze volatility by calculating the rolling standard deviation of daily returns"""
        # Calculate daily returns
        returns = data['Close'].pct_change()
        
        # Calculate the rolling standard deviation (volatility)
        volatility = returns.rolling(window=window).std().iloc[-1]  # Get the latest volatility value
        
        return volatility


    def generate_decision(self, analysis: Dict) -> Dict:
        """Generate trading decision based on technical analysis for PSX"""
        decision = {
            "action": "",
            "confidence": "",
            "reasons": [],
            "stop_loss": None,
            "target": None,
            "timestamp": self.current_time.strftime("%Y-%m-%d %H:%M:%S UTC")
        }
        
        current_price = analysis['current_price']
        
        # Decision Logic for PSX
        if analysis['signal_strength'] == "STRONG":
            decision['confidence'] = "HIGH"
            if analysis['signal_type'] == "SELL":
                decision['action'] = "SELL"
                decision['stop_loss'] = current_price * 0.98  # 2% below current price
                decision['target'] = current_price * 0.95     # 5% below for profit target
            else:
                decision['action'] = "BUY"
                decision['stop_loss'] = current_price * 0.98  # 2% below current price
                decision['target'] = current_price * 1.05     # 5% above for profit target
                
        elif analysis['signal_strength'] == "MODERATE":
            decision['confidence'] = "MEDIUM"
            if analysis['signal_type'] == "SELL":
                decision['action'] = "SELL"
                decision['stop_loss'] = current_price * 0.985  # 1.5% below current price
                decision['target'] = current_price * 0.97      # 3% below for profit target
            else:
                decision['action'] = "BUY"
                decision['stop_loss'] = current_price * 0.985  # 1.5% below current price
                decision['target'] = current_price * 1.03      # 3% above for profit target
                
        else:  # WEAK signal
            decision['confidence'] = "LOW"
            decision['action'] = "HOLD"
        
        # Generate reasons with PSX context
        if analysis['indicators']['RSI'] > 70:
            decision['reasons'].append(f"Overbought RSI: {analysis['indicators']['RSI']:.2f}")
        elif analysis['indicators']['RSI'] < 30:
            decision['reasons'].append(f"Oversold RSI: {analysis['indicators']['RSI']:.2f}")
        
        bb_upper = analysis['indicators']['BB_Upper']
        bb_lower = analysis['indicators']['BB_Lower']
        
        if current_price > bb_upper:
            decision['reasons'].append(f"Price above upper Bollinger Band: {bb_upper:.2f}")
        elif current_price < bb_lower:
            decision['reasons'].append(f"Price below lower Bollinger Band: {bb_lower:.2f}")
        
        # Check resistance and support
        nearest_resistance = min(analysis['resistance_levels']) if analysis['resistance_levels'] else float('inf')
        nearest_support = max(analysis['support_levels']) if analysis['support_levels'] else float('-inf')
        
        if nearest_resistance != float('inf'):
            decision['reasons'].append(f"Near resistance: {nearest_resistance:.2f}")
        if nearest_support != float('-inf'):
            decision['reasons'].append(f"Near support: {nearest_support:.2f}")
        
        return decision, decision['action'], decision['confidence']

    def analyze_stock(self, ticker: str, hist_data: pd.DataFrame, current_data: Dict) -> Dict:
        """Analyze a single stock with volatility included"""
        
        # Calculate technical indicators
        indicators = self.calculate_technical_indicators(hist_data)
        
        # Get current values
        current_price = float(current_data['current'].replace(',', ''))  # Use 'current' for current price
        
        # Find support/resistance levels
        support_levels, resistance_levels = self.find_support_resistance(hist_data)
        
        # Analyze volume
        volume_analysis = self.analyze_volume(hist_data)
        
        # Analyze volatility
        volatility = self.analyze_volatility(hist_data)
        
        # Generate signals
        signals = []
        signal_strength = 0
        
        # 1. RSI Analysis
        current_rsi = indicators['RSI'].iloc[-1]
        if current_rsi < 30:  # Oversold
            signals.append({"type": "RSI", "signal": "BUY", "strength": 2})
            signal_strength += 2
        elif current_rsi > 70:  # Overbought
            signals.append({"type": "RSI", "signal": "SELL", "strength": -2})
            signal_strength -= 2
        
        # 2. MACD Analysis
        if indicators['MACD'].iloc[-1] > indicators['Signal_Line'].iloc[-1]:
            signals.append({"type": "MACD", "signal": "BUY", "strength": 1})
            signal_strength += 1
        else:
            signals.append({"type": "MACD", "signal": "SELL", "strength": -1})
            signal_strength -= 1
        
        # 3. Bollinger Bands Analysis
        if current_price < indicators['BB_Lower'].iloc[-1]:
            signals.append({"type": "BB", "signal": "BUY", "strength": 1.5})
            signal_strength += 1.5
        elif current_price > indicators['BB_Upper'].iloc[-1]:
            signals.append({"type": "BB", "signal": "SELL", "strength": -1.5})
            signal_strength -= 1.5
        
        # 4. Volume Analysis
        if volume_analysis['volume_ratio'] > 1.5:
            signals.append({"type": "VOLUME", "signal": "HIGH", "strength": 1})
            signal_strength += 1
        elif volume_analysis['volume_ratio'] < 0.5:
            signals.append({"type": "VOLUME", "signal": "LOW", "strength": -1})
            signal_strength -= 1
        
        # Determine signal strength
        if abs(signal_strength) >= 4:
            strength_label = "STRONG"
        elif abs(signal_strength) >= 2:
            strength_label = "MODERATE"
        else:
            strength_label = "WEAK"
        
        signal_type = "BUY" if signal_strength > 0 else "SELL" if signal_strength < 0 else "NEUTRAL"
        
        analysis = {
            "ticker": ticker,
            "current_price": current_price,  # Using correct 'current' price
            "signal_type": signal_type,
            "signal_strength": strength_label,
            "total_strength": signal_strength,
            "signals": signals,
            "support_levels": support_levels,
            "resistance_levels": resistance_levels,
            "indicators": {
                "RSI": current_rsi,
                "MACD": indicators['MACD'].iloc[-1],
                "BB_Upper": indicators['BB_Upper'].iloc[-1],
                "BB_Lower": indicators['BB_Lower'].iloc[-1]
            },
            "volume_analysis": volume_analysis,
            "volatility": volatility  # Add volatility to analysis
        }
        
        return analysis


    def format_analysis_message(self, analysis: Dict) -> str:
        """Format analysis results into a readable message with volatility"""
        
        decision, action, confidence = self.generate_decision(analysis)
        
        message = f"""
Analysis Report for {analysis['ticker']}
Timestamp: {decision['timestamp']}
Current Price: {analysis['current_price']:.2f}

BOT DECISION: {decision['action']}
Confidence: {decision['confidence']}
"""
        if decision['stop_loss'] and decision['target']:
            message += f"Stop Loss: {decision['stop_loss']:.2f}\n"
            message += f"Target: {decision['target']:.2f}\n"
        
        message += f"""
Reasons:
{chr(10).join(f'• {reason}' for reason in decision['reasons'])}

Technical Analysis:
• RSI: {analysis['indicators']['RSI']:.2f}
• MACD: {analysis['indicators']['MACD']:.2f}
• BB Upper: {analysis['indicators']['BB_Upper']:.2f}
• BB Lower: {analysis['indicators']['BB_Lower']:.2f}

Volume Analysis:
• Current Volume: {analysis['volume_analysis']['current_volume']:.0f}
• Average Volume: {analysis['volume_analysis']['avg_volume']:.0f}
• Volume Ratio: {analysis['volume_analysis']['volume_ratio']:.2f}x

Volatility (Standard Deviation): {analysis['volatility']:.4f}

Support/Resistance Levels:
• Support: {', '.join(f'{level:.2f}' for level in analysis['support_levels'])}
• Resistance: {', '.join(f'{level:.2f}' for level in analysis['resistance_levels'])}

Signal Details:
"""
        
        for signal in analysis['signals']:
            message += f"• {signal['type']}: {signal['signal']} (Strength: {signal['strength']})\n"
        
        return message, action, confidence


    def run(self):
        """Main execution loop"""
        try:
            DATA = []
            MSGS = []
            # Get historical data
            # print("Fetching historical data...")
            historical_data = scrapDataHistory(STOCKS)
            
            # Get current market data
            # print("Fetching current market data...")
            current_data = getCurrentData(STOCKS)

            if os.path.exists("AnalysisResult/") and os.path.isdir("AnalysisResult/"):
                shutil.rmtree("AnalysisResult/")  # Use rmtree to delete non-empty directories
            
            # Analyze each stock
            # with Progress() as progress:
                # task = progress.add_task(f"[magenta]Generating Results...", total=len(STOCKS))
            for ticker, ticker_data in zip(STOCKS, current_data):
                if ticker_data:
                    # print(f"\nAnalyzing {ticker}...")
                    
                    # Get stock-specific historical data
                    stock_hist_data = historical_data.xs(ticker, level='Ticker')
                    
                    # Perform analysis
                    analysis = self.analyze_stock(ticker, stock_hist_data, ticker_data)

                    # DATA.append(analysis)
                    
                    # Format and print results
                    message, action, confidence = self.format_analysis_message(analysis)
                    decision, x, y = self.generate_decision(analysis)

                    # MSGS.append(message)
                    DATA.append((analysis, decision))
                    
                    
                    # Send the analysis result to Telegram
                    # self.telegram_notifier.send_message(message)
                    
                    
                    # # Make sure the folders exist, if not, create them
                    os.makedirs('Logs/', exist_ok=True)


                    folder_path = f"AnalysisResult/{action}"
                    os.makedirs(folder_path, exist_ok=True)
                    file_path = os.path.join(folder_path, f"{ticker}.txt")
                    with open(file_path, "w") as f:
                        f.write(message)

                        
                    logging.basicConfig(filename=f'Logs/{date.today().strftime("%d-%m-%Y")}.log', level=logging.INFO, format='%(asctime)s - %(message)s')
                    logging.info(message)
                    
                        # progress.update(task, advance=1)
                            
                time.sleep(1)  # Small delay between stocks
                
            # print("\nAnalysis complete.")
            
            if os.path.exists("__pycache__/") and os.path.isdir("__pycache__/"):
                shutil.rmtree("__pycache__/")
        
        except Exception as e:
            # print(f"Error occurred: {str(e)}")
            time.sleep(5)
        
        return DATA

if __name__ == "__main__":
    bot = TradeHunter()
    bot.run()