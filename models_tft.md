# 1. 导入必要的库
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

# 2. 数据配置
data_config = {
    "seq_length": 168,
    "predict_horizon": 24,
    "batch_size": 128,
    "static_features": 0,
    "known_features": {
        "利率": 1,
        "监管新闻": 10,
        "挖矿难度": 1,
        "汇率": 4,
    },
    "unknown_features": {
        "价格数据": 4,
        "交易量数据": 3,
        "技术分析指标": 15,
        "市场情绪数据": 6,
        "链上数据": 12,
        "宏观与监管数据": 15,
        "市场深度数据": 6,
    },
    "output_features": {
        "交易信号": 8,
        "风险评估": 12,
    }
}

# 3. 数据预处理
def preprocess_data(df):
    df = df.sort_values(['symbol', 'time']).reset_index(drop=True)
    df['time_idx'] = df.groupby('symbol')['time'].rank(method="dense").astype(int)
    label_encoder = LabelEncoder()
    df['买入卖出持有信号'] = label_encoder.fit_transform(df['买入卖出持有信号'])
    scaler = StandardScaler()
    unknown_reals = list(data_config["unknown_features"].keys())
    known_reals = list(data_config["known_features"].keys())
    df[unknown_reals + known_reals] = scaler.fit_transform(df[unknown_reals + known_reals])
    return df, label_encoder, scaler

# 加载和预处理数据
# df = pd.read_csv('your_data.csv')
df, label_encoder, scaler = preprocess_data(df)

# 4. 定义 TimeSeriesDataSet
training_cutoff = df['time_idx'].max() - data_config["predict_horizon"]

training = TimeSeriesDataSet(
    df[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target=["交易信号", "风险评估"],
    group_ids=["symbol"],
    min_encoder_length=data_config["seq_length"],
    max_encoder_length=data_config["seq_length"],
    min_prediction_length=data_config["predict_horizon"],
    max_prediction_length=data_config["predict_horizon"],
    static_categoricals=[],  
    static_reals=[],         
    time_varying_known_categoricals=[],  
    time_varying_known_reals=list(data_config["known_features"].keys()),  
    time_varying_unknown_categoricals=[],  
    time_varying_unknown_reals=list(data_config["unknown_features"].keys()),
    target_normalizer=GroupNormalizer(groups=["symbol"], transformation="softplus"),  
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

validation = TimeSeriesDataSet.from_dataset(training, df, min_prediction_idx=training.index.max() + 1)

train_dataloader = training.to_dataloader(train=True, batch_size=data_config["batch_size"], num_workers=0)
val_dataloader = validation.to_dataloader(train=False, batch_size=data_config["batch_size"], num_workers=0)

# 5. 定义 MultiTaskTFT 模型
class MultiTaskTFT(nn.Module):
    def __init__(self, tft_model, output_size_dict):
        super().__init__()
        self.tft = tft_model
        self.output_size_dict = output_size_dict
        hidden_size = tft_model.hparams.hidden_size
        
        # 交易信号输出头
        self.buy_signal = nn.Linear(hidden_size, output_size_dict['交易信号']['买入信号'])
        self.sell_signal = nn.Linear(hidden_size, output_size_dict['交易信号']['卖出信号'])
        self.hold_signal = nn.Linear(hidden_size, output_size_dict['交易信号']['持有信号'])
        self.signal_strength = nn.Linear(hidden_size, output_size_dict['交易信号']['信号强度'])
        self.position_ratio = nn.Linear(hidden_size, output_size_dict['交易信号']['建议持仓比例'])
        self.price_prediction = nn.Linear(hidden_size, output_size_dict['交易信号']['预测价格'])
        
        # 风险评估输出头
        self.volatility_risk = nn.Linear(hidden_size, output_size_dict['风险评估']['波动风险'])
        self.trend_reliability = nn.Linear(hidden_size, output_size_dict['风险评估']['趋势可靠度'])
        self.liquidity_risk = nn.Linear(hidden_size, output_size_dict['风险评估']['流动性风险'])
        self.tech_score = nn.Linear(hidden_size, output_size_dict['风险评估']['技术面评分'])
        self.fundamental_score = nn.Linear(hidden_size, output_size_dict['风险评估']['基本面评分'])
        self.market_timing_score = nn.Linear(hidden_size, output_size_dict['风险评估']['市场时机评分'])
        self.comprehensive_score = nn.Linear(hidden_size, output_size_dict['风险评估']['综合建议得分'])
        
    def forward(self, x):
        tft_output = self.tft(x)
        hidden = tft_output[:, -1, :]
        
        # 交易信号预测
        buy_pred = self.buy_signal(hidden)
        sell_pred = self.sell_signal(hidden)
        hold_pred = self.hold_signal(hidden)
        strength_pred = self.signal_strength(hidden)
        ratio_pred = self.position_ratio(hidden)
        price_pred = self.price_prediction(hidden)
        
        # 风险评估预测
        volatility_pred = self.volatility_risk(hidden)
        trend_pred = self.trend_reliability(hidden)
        liquidity_pred = self.liquidity_risk(hidden)
        tech_score_pred = self.tech_score(hidden)
        fundamental_score_pred = self.fundamental_score(hidden)
        market_timing_pred = self.market_timing_score(hidden)
        comprehensive_pred = self.comprehensive_score(hidden)
        
        return {
            "交易信号": {
                "买入信号": buy_pred,
                "卖出信号": sell_pred,
                "持有信号": hold_pred,
                "信号强度": strength_pred,
                "建议持仓比例": ratio_pred,
                "预测价格": price_pred
            },
            "风险评估": {
                "波动风险": volatility_pred,
                "趋势可靠度": trend_pred,
                "流动性风险": liquidity_pred,
                "技术面评分": tech_score_pred,
                "基本面评分": fundamental_score_pred,
                "市场时机评分": market_timing_pred,
                "综合建议得分": comprehensive_pred
            }
        }

# 6. 定义自定义损失函数
class MultiTaskLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.buy_sell_hold_loss = nn.CrossEntropyLoss()
        self.signal_strength_loss = nn.MSELoss()
        self.position_ratio_loss = nn.MSELoss()
        self.price_prediction_loss = nn.MSELoss()
        self.volatility_risk_loss = nn.MSELoss()
        self.trend_reliability_loss = nn.MSELoss()
        self.liquidity_risk_loss = nn.MSELoss()
        self.tech_score_loss = nn.MSELoss()
        self.fundamental_score_loss = nn.MSELoss()
        self.market_timing_loss = nn.MSELoss()
        self.comprehensive_score_loss = nn.MSELoss()
        
        self.weights = weights or {
            "buy_sell_hold": 1.0,
            "signal_strength": 1.0,
            "position_ratio": 1.0,
            "price_prediction": 1.0,
            "volatility_risk": 1.0,
            "trend_reliability": 1.0,
            "liquidity_risk": 1.0,
            "tech_score": 1.0,
            "fundamental_score": 1.0,
            "market_timing": 1.0,
            "comprehensive_score": 1.0,
        }
    
    def forward(self, y_pred, y_true):
        buy_pred = y_pred["交易信号"]["买入信号"]
        sell_pred = y_pred["交易信号"]["卖出信号"]
        hold_pred = y_pred["交易信号"]["持有信号"]
        signal_strength_pred = y_pred["交易信号"]["信号强度"]
        position_ratio_pred = y_pred["交易信号"]["建议持仓比例"]
        price_pred = y_pred["交易信号"]["预测价格"]
        
        buy_true = y_true["交易信号"]["买入信号"]
        sell_true = y_true["交易信号"]["卖出信号"]
        hold_true = y_true["交易信号"]["持有信号"]
        signal_strength_true = y_true["交易信号"]["信号强度"]
        position_ratio_true = y_true["交易信号"]["建议持仓比例"]
        price_true = y_true["交易信号"]["预测价格"]
        
        loss_buy_sell_hold = self.buy_sell_hold_loss(
            torch.cat([buy_pred, sell_pred, hold_pred], dim=1),
            torch.argmax(torch.cat([buy_true, sell_true, hold_true], dim=1), dim=1)
        )
        loss_signal_strength = self.signal_strength_loss(signal_strength_pred, signal_strength_true)
        loss_position_ratio = self.position_ratio_loss(position_ratio_pred, position_ratio_true)
        loss_price_prediction = self.price_prediction_loss(price_pred, price_true)
        
        risk_pred = y_pred["风险评估"]
        risk_true = y_true["风险评估"]
        
        loss_volatility_risk = self.volatility_risk_loss(risk_pred["波动风险"], risk_true["波动风险"])
        loss_trend_reliability = self.trend_reliability_loss(risk_pred["趋势可靠度"], risk_true["趋势可靠度"])
        loss_liquidity_risk = self.liquidity_risk_loss(risk_pred["流动性风险"], risk_true["流动性风险"])
        loss_tech_score = self.tech_score_loss(risk_pred["技术面评分"], risk_true["技术面评分"])
        loss_fundamental_score = self.fundamental_score_loss(risk_pred["基本面评分"], risk_true["基本面评分"])
        loss_market_timing = self.market_timing_loss(risk_pred["市场时机评分"], risk_true["市场时机评分"])
        loss_comprehensive_score = self.comprehensive_score_loss(risk_pred["综合建议得分"], risk_true["综合建议得分"])
        
        total_loss = (
            self.weights["buy_sell_hold"] * loss_buy_sell_hold +
            self.weights["signal_strength"] * loss_signal_strength +
            self.weights["position_ratio"] * loss_position_ratio +
            self.weights["price_prediction"] * loss_price_prediction +
            self.weights["volatility_risk"] * loss_volatility_risk +
            self.weights["trend_reliability"] * loss_trend_reliability +
            self.weights["liquidity_risk"] * loss_liquidity_risk +
            self.weights["tech_score"] * loss_tech_score +
            self.weights["fundamental_score"] * loss_fundamental_score +
            self.weights["market_timing"] * loss_market_timing +
            self.weights["comprehensive_score"] * loss_comprehensive_score
        )
        
        return total_loss

# 7. 定义训练步骤
def train_model(model, train_dataloader, val_dataloader, loss_fn, max_epochs=30, gpus=0):
    device = torch.device("cuda" if gpus and torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            batch = {k: v.to(device) for k, v in batch.items()}
            y_pred = model(batch)
            y_true = {
                "交易信号": {
                    "买入信号": batch["future"]["买入信号"],
                    "卖出信号": batch["future"]["卖出信号"],
                    "持有信号": batch["future"]["持有信号"],
                    "信号强度": batch["future"]["信号强度"],
                    "建议持仓比例": batch["future"]["建议持仓比例"],
                    "预测价格": batch["future"]["预测价格"],
                },
                "风险评估": {
                    "波动风险": batch["future"]["波动风险"],
                    "趋势可靠度": batch["future"]["趋势可靠度"],
                    "流动性风险": batch["future"]["流动性风险"],
                    "技术面评分": batch["future"]["技术面评分"],
                    "基本面评分": batch["future"]["基本面评分"],
                    "市场时机评分": batch["future"]["市场时机评分"],
                    "综合建议得分": batch["future"]["综合建议得分"],
                }
            }
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                y_pred = model(batch)
                y_true = {
                    "交易信号": {
                        "买入信号": batch["future"]["买入信号"],
                        "卖出信号": batch["future"]["卖出信号"],
                        "持有信号": batch["future"]["持有信号"],
                        "信号强度": batch["future"]["信号强度"],
                        "建议持仓比例": batch["future"]["建议持仓比例"],
                        "预测价格": batch["future"]["预测价格"],
                    },
                    "风险评估": {
                        "波动风险": batch["future"]["波动风险"],
                        "趋势可靠度": batch["future"]["趋势可靠度"],
                        "流动性风险": batch["future"]["流动性风险"],
                        "技术面评分": batch["future"]["技术面评分"],
                        "基本面评分": batch["future"]["基本面评分"],
                        "市场时机评分": batch["future"]["市场时机评分"],
                        "综合建议得分": batch["future"]["综合建议得分"],
                    }
                }
                loss = loss_fn(y_pred, y_true)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{max_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
    
    return model

# 8. 实例化并训练模型
# 创建基础 TFT 模型
base_tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=64,
    output_size=1,  # 不直接使用基础 TFT 输出
    loss=nn.L1Loss(),  # 占位损失
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# 实例化多任务模型
multi_task_tft = MultiTaskTFT(
    tft_model=base_tft,
    output_size_dict=output_size_dict
)

# 实例化损失函数
multi_task_loss = MultiTaskLoss()

# 检查 GPU
gpus = 1 if torch.cuda.is_available() else 0

# 训练模型
trained_model = train_model(
    model=multi_task_tft,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    loss_fn=multi_task_loss,
    max_epochs=30,
    gpus=gpus
)




# 预测函数
def make_predictions(model, dataloader):
    model.eval()
    predictions = {
        "交易信号": {
            "买入信号": [],
            "卖出信号": [],
            "持有信号": [],
            "信号强度": [],
            "建议持仓比例": [],
            "预测价格": []
        },
        "风险评估": {
            "波动风险": [],
            "趋势可靠度": [],
            "流动性风险": [],
            "技术面评分": [],
            "基本面评分": [],
            "市场时机评分": [],
            "综合建议得分": []
        }
    }
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y_pred = model(batch)
            # 交易信号
            for key in predictions["交易信号"]:
                predictions["交易信号"][key].append(y_pred["交易信号"][key].cpu().numpy())
            # 风险评估
            for key in predictions["风险评估"]:
                predictions["风险评估"][key].append(y_pred["风险评估"][key].cpu().numpy())
    
    # 合并批次结果
    for category in predictions:
        for key in predictions[category]:
            predictions[category][key] = np.concatenate(predictions[category][key], axis=0)
    
    return predictions

# 示例预测
predictions = make_predictions(trained_model, val_dataloader)

# 示例输出
for i in range(5):
    print(f"预测价格: {predictions['交易信号']['预测价格'][i]}, "
          f"买入信号: {predictions['交易信号']['买入信号'][i]}, "
          f"卖出信号: {predictions['交易信号']['卖出信号'][i]}, "
          f"持有信号: {predictions['交易信号']['持有信号'][i]}")