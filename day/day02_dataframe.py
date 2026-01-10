import pandas as pd

def main():
    data= {
    'Open': [150, 2800, 700],
    'Close': [155, 2850, 695],
    'Volume': [1000, 200, 5000]
    }
    df=pd.DataFrame(data, index=['AAPL', 'GOOGL', 'TSLA'])
    # 计算收盘价高于开盘价的股票的总交易量
    print(df[df['Close'] > df['Open']]['Volume'].sum())

if __name__ == "__main__":
    main()