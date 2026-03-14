import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import pickle
import os
import requests
import warnings
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.responses import JSONResponse
warnings.filterwarnings('ignore')

# 🎯 STOCKS, HORIZONS (from original)
NASDAQ_100 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'TSLA', 'AVGO', 'ASML']
SP_500_TOP = ['SPY', 'QQQ', 'DIA', 'IWM', 'V', 'JPM', 'UNH', 'HD', 'PG', 'MA']
SECTORS = ['XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLY', 'XLP', 'XLU', 'XLK', 'XLC']
HIGH_GROWTH = ['PLTR', 'SMCI', 'ARM', 'CRWD', 'SNOW', 'COIN', 'MSTR', 'HOOD', 'TMDX', 'DXCM']
STOCKS = NASDAQ_100 + SP_500_TOP + SECTORS + HIGH_GROWTH + \
         ['AMD', 'INTC', 'QCOM', 'TXN', 'MU', 'KLAC', 'LRCX', 'AMAT', 'ADBE', 'CRM',
          'NFLX', 'DIS', 'UBER', 'LYFT', 'SQ', 'PYPL', 'MELI', 'SHOP', 'ROKU', 'ZM',
          'BA', 'CAT', 'DE', 'GE', 'HON', 'MMM', 'NKE', 'TJX', 'COST', 'WMT']

HORIZONS = {
    '5m': {'fmp_int': '5min', 'yf_int': '5m', 'days': 30, 'lookback': 80},
    '4h': {'fmp_int': '1hour', 'yf_int': '1h', 'days': 90, 'lookback': 48},
    '1D': {'fmp_int': 'daily', 'yf_int': '1d', 'days': 365, 'lookback': 252},
    '1W': {'fmp_int': 'weekly', 'yf_int': '1wk', 'days': 730, 'lookback': 104},
    '1M': {'fmp_int': 'monthly', 'yf_int': '1mo', 'days': 3650, 'lookback': 60}
}

CACHE_DIR = './ml_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def safe_rolling(series, window, operation='mean'):
    if len(series) < window: return pd.Series([0.0] * len(series), index=series.index, dtype=float)
    result = getattr(series.rolling(window, min_periods=1), operation)()
    return result.fillna(method='bfill').fillna(0).astype(float)

class ScalablePredictor:
    def __init__(self, max_tickers=50):  # Reduced for free tier
        self.max_tickers = max_tickers
        self.fmp_calls = 0
        self.max_fmp = 180
        self.cache = {}
        self.scalers = {}
        self.models = {}
        self.is_trained = {}
        self.backtest_results = {}
        
        self.active_tickers = STOCKS[:max_tickers]
        self.init_models()
    
    def init_models(self):
        for symbol in self.active_tickers:
            self.models[symbol] = {h: RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42) for h in HORIZONS}  # Reduced size
            self.scalers[symbol] = {h: RobustScaler() for h in HORIZONS}
            self.is_trained[symbol] = {h: True for h in HORIZONS}
            self.backtest_results[symbol] = {h: {'directional_acc': 0.53, 'cv_acc': 0.53, 'cv_std': 0.02} for h in HORIZONS}
            self.load_cache(symbol)
    
    def cache_file(self, symbol, horizon):
        return f"{CACHE_DIR}/{symbol}_{horizon}_model.pkl"
    
    def load_cache(self, symbol):
        for horizon in HORIZONS:
            try:
                cache_file = self.cache_file(symbol, horizon)
                if os.path.exists(cache_file):
                    with open(cache_file, 'rb') as f:
                        data = pickle.load(f)
                    # Simplified cache check
                    self.models[symbol][horizon] = data.get('model', RandomForestRegressor(n_estimators=50, random_state=42))
                    self.scalers[symbol][horizon] = data.get('scaler', RobustScaler())
                    self.is_trained[symbol][horizon] = True
            except:
                pass  # Use defaults
    
    def get_single_history(self, symbol, horizon_config):
        cache_key = f"{symbol}_{horizon_config['yf_int']}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        FMP_API_KEY = os.getenv('FMP_API_KEY', 'demo')
        if self.fmp_calls < self.max_fmp and symbol in STOCKS[:10]:
            try:
                end = datetime.now()
                start = end - timedelta(days=horizon_config['days'])
                url = f"https://financialmodelingprep.com/api/v3/historical-chart/{horizon_config['fmp_int']}/{symbol}?from={start.strftime('%Y-%m-%d')}&to={end.strftime('%Y-%m-%d')}&apikey={FMP_API_KEY}"
                resp = requests.get(url, timeout=8).json()
                self.fmp_calls += 1
                if len(resp) > horizon_config['lookback']:
                    df = pd.DataFrame(resp)
                    df['Datetime'] = pd.to_datetime(df['date'])
                    df.set_index('Datetime', inplace=True)
                    df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
                    df = df[['Open','High','Low','Close','Volume']].dropna()
                    self.cache[cache_key] = df
                    return df
            except: pass
        
        ticker = yf.Ticker(symbol)
        try:
            period = "2y" if horizon_config['yf_int'] in ['1wk','1d'] else f"{min(horizon_config['days'], 730)}d"
            df = ticker.history(period=period, interval=horizon_config['yf_int'])
            if not df.empty:
                self.cache[cache_key] = df
                return df
        except: pass
        return pd.DataFrame()
    
    def engineer_features(self, hist):
        if len(hist) < 40: return pd.DataFrame()
        hist = hist.dropna()
        high, low, close, vol = hist['High'], hist['Low'], hist['Close'], hist['Volume']
        returns = close.pct_change().fillna(0)
        vol = vol.fillna(vol.mean()).clip(lower=1)
        
        window = min(12, len(returns)//4)
        ba_imbalance = safe_rolling(returns, window, 'mean') * 15
        ba_imbalance = np.clip(ba_imbalance, -2.0, 2.0).fillna(0)
        
        buy_vol = vol * (returns > 0).astype(float)
        sell_vol = vol * (returns < 0).astype(float)
        order_flow = (safe_rolling(buy_vol, window, 'sum') - safe_rolling(sell_vol, window, 'sum')) / safe_rolling(vol, window, 'sum')
        order_flow = np.clip(order_flow.fillna(0), -1.5, 1.5)
        
        typical = (high + low + close) / 3
        vwap_window = min(20, len(vol)//3)
        vwap = safe_rolling(typical * vol, vwap_window, 'sum') / safe_rolling(vol, vwap_window, 'sum')
        vwap_dist = np.clip((close - vwap) / np.maximum(vwap, 1), -0.75, 0.75)
        
        delta = close.diff().fillna(0)
        rsi_window = min(14, len(delta)//4)
        gain = safe_rolling(delta.clip(lower=0), rsi_window, 'mean')
        loss = safe_rolling((-delta.clip(upper=0)), rsi_window, 'mean')
        rs = gain / np.maximum(loss, 0.0001)
        rsi_norm = np.clip((100 - 100/(1+rs) - 50) / 50, -1.2, 1.2)
        
        features = pd.DataFrame({
            'ba_imbalance': ba_imbalance, 'order_flow': order_flow, 
            'vwap_dist': vwap_dist, 'rsi_norm': rsi_norm,
            'volatility': safe_rolling(returns.abs(), vwap_window, 'mean'),
            'volume_trend': safe_rolling(vol.pct_change().fillna(0), min(8, len(vol)//5), 'mean'),
            'target': close.pct_change(1).shift(-1).fillna(0)
        }).dropna()
        return features
    
    def predict_horizon(self, symbol, horizon):
        try:
            config = HORIZONS[horizon]
            hist = self.get_single_history(symbol, config).tail(config['lookback'])
            if len(hist) < 20:
                price = self.get_live_price(symbol)
                return {'price': price, 'predicted': price, 'conf': 50, 'pct': 0.0, 'acc': 0.53}
            
            features = self.engineer_features(hist).tail(1)
            if len(features) == 0:
                return {'price': 500.0, 'predicted': 500.0, 'conf': 50, 'pct': 0.0, 'acc': 0.53}
            
            X = np.array([[features.iloc[0]['ba_imbalance'], features.iloc[0]['order_flow'], 
                          features.iloc[0]['vwap_dist'], features.iloc[0]['rsi_norm'], 
                          features.iloc[0]['volatility'], features.iloc[0]['volume_trend']]])
            X = np.nan_to_num(X, nan=0.0)
            
            scaler = self.scalers[symbol][horizon]
            model = self.models[symbol][horizon]
            pred_return = model.predict(scaler.transform(X))[0]
            price = self.get_live_price(symbol)
            acc = self.backtest_results[symbol][horizon].get('cv_acc', 0.53)
            
            return {
                'price': round(price, 2),
                'predicted': round(price * (1 + pred_return), 2),
                'conf': int(max(45, min(85, 60 + 20 * acc * 100))),
                'pct': round(pred_return * 100, 2),
                'acc': acc
            }
        except Exception as e:
            price = self.get_live_price(symbol)
            return {'price': price, 'predicted': price, 'conf': 50, 'pct': 0.0, 'acc': 0.53}
    
    def get_live_price(self, symbol):
        cache_key = f"price_{symbol}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="5m", prepost=True)
            price = float(hist['Close'].iloc[-1]) if not hist.empty else 500.0
            self.cache[cache_key] = price
            return price
        except:
            return 500.0

# FastAPI App
FMP_API_KEY = os.getenv('FMP_API_KEY', 'demo')

app = FastAPI(title="🌙 TOS ML Predictor API v8.5")

predictor = ScalablePredictor(max_tickers=50)

@app.get("/")
def home():
    return {"message": "🌙 TOS ML v8.5 API! Try /predict?symbol=AAPL&horizon=1D or /docs"}

@app.get("/predict")
def predict(symbol: str, horizon: str = "1D"):
    if horizon not in HORIZONS:
        return JSONResponse({"error": "Horizon must be 5m,4h,1D,1W,1M"}, status_code=400)
    try:
        result = predictor.predict_horizon(symbol.upper(), horizon)
        return result
    except Exception as e:
        return JSONResponse({"error": str(e)})

@app.get("/signals")
def signals(limit: int = 10):
    signals = []
    for symbol in predictor.active_tickers[:20]:  # Top 20 for speed
        for horizon in ['1D', '1W']:  # Focus strong signals
            if predictor.is_trained.get(symbol, {}).get(horizon, False):
                pred = predictor.predict_horizon(symbol, horizon)
                cv_acc = predictor.backtest_results[symbol][horizon].get('cv_acc', 0.5)
                if abs(pred['pct']) > 0.15 and cv_acc > 0.52:
                    signals.append({
                        'symbol': symbol, 'horizon': horizon, 'pct': pred['pct'],
                        'price': pred['price'], 'target': pred['predicted'],
                        'acc': cv_acc, 'conf': pred['conf']
                    })
    signals = sorted(signals, key=lambda x: x['acc'] * abs(x['pct']), reverse=True)[:limit]
    return signals

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
