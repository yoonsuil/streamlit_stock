import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import yfinance as yf

# 사용자 입력 설정
st.title("자동매매 시스템")

# 1. 초기 자본 설정
initial_capital = st.number_input("초기 자본(원)", min_value=100000.0, step=10000.0, value=1000000.0)

# 2. 매매 비율 설정
trade_ratio = st.slider("매매 비율(%)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)

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
    # 데이터 불러오기
    tickers = ["TSLA", "NVDA"]
    df = yf.tickers_download(tickers, period="5y")

    # 전략 적용
    if strategy == "트렌드 추종":
        df["short_ma"] = df.groupby("Ticker")["Close"].rolling(short_window).mean().reset_index(level=1, drop=True)
        df["long_ma"] = df.groupby("Ticker")["Close"].rolling(long_window).mean().reset_index(level=1, drop=True)
        df["signal"] = np.where(df["short_ma"] > df["long_ma"], 1, -1)
    elif strategy == "모멘텀":
        df["rsi"] = df.groupby("Ticker")["Close"].apply(lambda x: ta.rsi(x, length=rsi_window))
        df["signal"] = np.where(df["rsi"] < 30, 1, np.where(df["rsi"] > 70, -1, 0))
    elif strategy == "평균 회귀":
        df["bb_upper"] = df.groupby("Ticker")["Close"].apply(lambda x: ta.bbands(x, length=bb_window, std=bb_std)[0])
        df["bb_lower"] = df.groupby("Ticker")["Close"].apply(lambda x: ta.bbands(x, length=bb_window, std=bb_std)[2])
        df["signal"] = np.where(df["Close"] > df["bb_upper"], -1, np.where(df["Close"] < df["bb_lower"], 1, 0))

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
        drawdown = (portfolio - max(pnl)) / max(pnl) * 100
        max_drawdown = min(max_drawdown, drawdown)
        annual_return = (portfolio / initial_capital - 1) * 100 / (len(df) / 252)
        sharpe_ratio = annual_return / np.std(np.array(pnl))

    # 결과 출력
    st.write(f"최종 포트폴리오: {portfolio:.2f}원")
    st.write(f"총 수익률: {(portfolio / initial_capital - 1) * 100:.2f}%")
    st.write(f"연간 수익률: {annual_return:.2f}%")
    st.write(f"최대 낙폭: {max_drawdown:.2f}%")
    st.write(f"샤프 비율: {sharpe_ratio:.2f}")
    st.write(f"총 거래 횟수: {trades}")
    st.line_chart(pd.DataFrame(pnl))