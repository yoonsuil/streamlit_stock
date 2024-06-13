import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf
import matplotlib.pyplot as plt
import math
import numpy as np

# 데이터 로드 함수
@st.cache_data
def load_data(tickers, start_date, end_date):
    data_list = []
    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        data["Ticker"] = ticker
        data = data.reset_index()
        data_list.append(data)
    
    df = pd.concat(data_list, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df


# 사용자 입력 설정
st.title("자동매매 시스템 lite version")

# 1. 초기 자본 설정
initial_capital = st.number_input("초기 자본(원)", min_value=100000.0, step=10000.0, value=1000000.0)

# 2. 매매 비율 설정
trade_ratio = st.slider("매매 비율(%)", min_value=1.0, max_value=70.0, value=5.0, step=0.1)

# 3. 매매 전략 선택
strategy = st.selectbox("매매 전략 선택", ["트렌드 추종", "모멘텀", "평균 회귀"])

# 4. 기술 지표 파라미터 설정
if strategy == "트렌드 추종":
    short_window = st.number_input("단기 이동 평균선 기간", min_value=10, max_value=50, value=20, step=1)
    long_window = st.number_input("장기 이동 평균선 기간", min_value=30, max_value=100, value=50, step=1)
elif strategy == "모멘텀":
    rsi_window = st.number_input("RSI 기간", min_value=5, max_value=30, value=14, step=1)
elif strategy == "평균 회귀":
    bb_window = st.number_input("볼린저 밴드 기간", min_value=10, max_value=50, value=20, step=1)
    bb_std = st.number_input("볼린저 밴드 표준편차", min_value=1.0, max_value=3.0, value=2.0, step=0.1)

# 5. 리스크 관리
max_loss = st.number_input("최대 손실 허용(%)", min_value=1.0, max_value=10.0, value=2.0, step=0.1)

# 6. 백테스트 실행
if st.button("백테스트 실행"):
    # 백테스트 기간 선택
    start_date = st.date_input("백테스트 시작 날짜", value=pd.Timestamp.today() - pd.Timedelta(days=365))
    end_date = st.date_input("백테스트 종료 날짜", value=pd.Timestamp.today())
    # 데이터 불러오기
    tickers = ["NVDA"]
    df = load_data(tickers, start_date, end_date)
        
# 전략 적용 함수

def apply_strategy(df, strategy, **kwargs):
    if strategy == "트렌드 추종":
        short_window = kwargs["short_window"]
        long_window = kwargs["long_window"]
        df["short_ma"] = df.groupby(level=0)["Close"].rolling(short_window).mean().reset_index(level=0, drop=True)
        df["long_ma"] = df.groupby(level=0)["Close"].rolling(long_window).mean().reset_index(level=0, drop=True)
        df["signal"] = np.where(df["short_ma"] > df["long_ma"], 1, -1)
    elif strategy == "모멘텀":
        rsi_window = kwargs["rsi_window"]
        df["rsi"] = df.groupby(level=0)["Close"].rolling(rsi_window).apply(lambda x: ta.rsi(x)).reset_index(level=0, drop=True)
        df["signal"] = np.where(df["rsi"] > 30, 1, np.where(df["rsi"] < 70, -1, 0))
    elif strategy == "평균 회귀":
        bb_window = kwargs["bb_window"]
        bb_std = kwargs["bb_std"]
        bb = df.groupby(level=0)["Close"].rolling(bb_window).apply(lambda x: ta.bbands(x, length=bb_window, std=bb_std)).reset_index(level=0, drop=True)
        df["bb_upper"] = bb[:, 0]
        df["bb_middle"] = bb[:, 1]
        df["bb_lower"] = bb[:, 2]
        df["signal"] = np.where(df["Close"] < df["bb_upper"], 1, np.where(df["Close"] > df["bb_lower"], -1, 0))
    
    df["Returns"] = df.groupby("Ticker")["signal"].shift(1) * df["Close"].pct_change()
    
    results = df.groupby("Ticker")["Returns"].apply(lambda x: pd.Series([x.mean() * 252, x.std() * sqrt(252)], index=["Annual Return", "Annual Volatility"])).T
    return results


    # 포트폴리오 관리
    portfolio = initial_capital
    trades = 0
    pnl = []
    max_drawdown = 0
    annual_return = 0
    sharpe_ratio = 0
    for i in range(1, len(df)):
        if df["signal"][i] == 1:
            order_amount = portfolio * (trade_ratio / 100)
            portfolio -= order_amount
            trades += 1
        elif df["signal"][i] == -1:
            order_amount = portfolio
            portfolio = 0
            trades += 1
        pnl.append(portfolio)
        
        # Drawdown 계산
        if max(pnl) != 0:
            drawdown = (portfolio - max(pnl)) / max(pnl) * 100
            max_drawdown = min(max_drawdown, drawdown)
        
        # 연간 수익률 및 샤프 비율 계산
        if len(pnl) > 1:
            annual_return = (portfolio / initial_capital - 1) * 100 / (len(df) / 252)
            if np.std(np.array(pnl)) != 0:
                sharpe_ratio = annual_return / np.std(np.array(pnl))

#백테스트 실행 후
if strategy == "트렌드 추종":
    # 전략 적용 후 결과 데이터프레임 생성
    results = apply_strategy(df, strategy, short_window=20, long_window=50)
    
    # 수익률 및 성과 지표 계산
    annual_return = results["Returns"].mean() * 252
    max_drawdown = results["Returns"].cumsum().groupby(pd.Grouper(freq="Y")).min().min()
    sharpe_ratio = results["Returns"].mean() / results["Returns"].std() * np.sqrt(252)
    
    # 결과 출력
    st.write(f"연간 수익률: {annual_return:.2%}")
    st.write(f"최대 낙폭: {max_drawdown:.2%}")
    st.write(f"샤프 비율: {sharpe_ratio:.2f}")
    
# 수익률 그래프 그리기
fig, ax = plt.subplots(figsize=(12, 6))
results["Returns"].cumsum().plot(ax=ax)
ax.set_title("누적 수익률 그래프")
ax.set_xlabel("날짜")
ax.set_ylabel("수익률")
st.pyplot(fig)

# 주가 및 전략 신호 그래프 그리기
fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
results["Close"].plot(ax=ax[0], label="주가")
results["signal"].plot(ax=ax[1], label="매매 신호")
ax[0].set_title("주가 그래프")
ax[1].set_title("매매 신호 그래프")
ax[0].legend()
ax[1].legend()
st.pyplot(fig)


