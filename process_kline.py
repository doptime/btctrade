import json
import pandas as pd
import ta  # 新增导入


def load_kline_data(file_path):
    """
    加载KLine数据。
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def process_kline_data(kline_data):
    """
    处理KLine数据，提取价格数据和交易量数据，并计算MACD指标。
    """
    processed_data = []
    for entry in kline_data:
        open_time = entry[0]
        open_price = float(entry[1])
        high_price = float(entry[2])
        low_price = float(entry[3])
        close_price = float(entry[4])
        volume = float(entry[5])
        close_time = entry[6]
        turnover = float(entry[7])
        num_trades = int(entry[8])
        buy_volume = float(entry[9])
        buy_turnover = float(entry[10])
        # 忽略第12项

        sell_volume = volume - buy_volume  # 卖出量 = 总成交量 - 买入量

        processed_entry = {
            'open_time': open_time,
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price,
            'close_price': close_price,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'total_volume': volume
        }
        processed_data.append(processed_entry)
    
    df = pd.DataFrame(processed_data)

    # 计算MACD指标
    df['MACD'] = ta.trend.macd(close=df['close_price'])
    df['MACD_Signal'] = ta.trend.macd_signal(close=df['close_price'])
    df['MACD_Hist'] = ta.trend.macd_diff(close=df['close_price'])

    return df


def save_to_csv(df, output_path):
    """
    将处理后的数据保存为CSV文件。
    """
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    input_file = 'path_to_your_kline_data.json'  # 替换为实际的KLine数据文件路径
    output_file = 'processed_kline_data.csv'

    kline_data = load_kline_data(input_file)
    df = process_kline_data(kline_data)
    save_to_csv(df, output_file)

    print(f"Processed data saved to {output_file}")