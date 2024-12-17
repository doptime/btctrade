当前系统目标是开发一个虚拟币交易模型。
    1. 模型采用TFT,使用python 来训练和部署。目前正在做早期的原型实现。
    2. 训练用到的模型的输入和输出数据采用python 处理后保持到csv作为训练数据。
开发原则：
1. Kepp Lean, 暂时不要管多模态之类的目标。
2. 当前的目标只需要涉及KLines 可以获得的那些TFT模型的输入变量。因为我正在开发最初的原型。
3. 对其它KLines无法生成的 TFT模型的输入变量，我会用0矢量替代，你只需要通过分析，然后告诉我 诸如 “讨论热度无法通过KLines获得”
4. 也不要想着一个模型可以搞定多模态。不要把事情复杂化。情绪等指标我会购买别人的API 来实现。不用操心这块的模型实现。


你需要一次专注高质量地实现一个目标。 把kLinesSlice 变成预期的输入，和输出 比如价格数据，交易量数据等：

请确保对目标的实现符合:
	1.面向实质功能的改进，务必避免和杀死早期优化
	2.这种实现确保你的实现并非对完全脱离现有系统进行重大的重新设计，而是对现有系统做渐进式的实现和调整。
    3.实现的目标需要包括必要的注释，以使得后续的迭代成为可能。
	



以下是将输入输出特征转化为任务清单的Markdown表格。请继续按顺序逐项完成 未开始 或需要完善 条目，以确保各个任务能够依次实现。

## 一、优化后的输入特征 (61维)

### 1. 基础市场数据 (7维)

| 特征       | 维度 | 具体含义                                                                                                                                               | 完成状态   |
|------------|------|--------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| 价格数据   | 4    | - 开盘价：交易时段开始时的价格<br>- 最高价：交易时段内的最高价格<br>- 最低价：交易时段内的最低价格<br>- 收盘价：交易时段结束时的价格                     | 未开始     |
| 交易量数据 | 3    | - 买入量：市场总买入订单量<br>- 卖出量：市场总卖出订单量<br>- 成交量：实际成交的总交易量                                                                 | 未开始     |

### 2. 技术分析指标 (15维)

| 特征组         | 维度 | 具体含义                                                                                               | 完成状态   |
|----------------|------|--------------------------------------------------------------------------------------------------------|------------|
| MACD           | 3    | - MACD线<br>- 信号线<br>- 柱状图                                                                        | 未开始     |
| 移动平均线     | 2    | - 短期MA<br>- 中期MA                                                                                     | 未开始     |
| RSI            | 1    | - 相对强弱指数                                                                                            | 未开始     |
| KD指标         | 2    | - K线<br>- D线                                                                                            | 未开始     |
| 布林带         | 3    | - 上轨<br>- 中轨<br>- 下轨                                                                                 | 未开始     |
| ATR            | 1    | - 平均真实波幅                                                                                            | 未开始     |
| 资金流向       | 1    | - 资金流向指标                                                                                            | 未开始     |
| 成交量趋势     | 2    | - 成交量上升趋势<br>- 成交量下降趋势                                                                      | 未开始     |

### 3. 市场情绪数据 (6维)

| 特征组       | 维度 | 具体含义                                                                                     | 完成状态   |
|--------------|------|----------------------------------------------------------------------------------------------|------------|
| 市场情绪指数 | 1    | - 综合市场情绪指数                                                                            | 未开始     |
| 社交情绪     | 2    | - 社交情绪正面比例<br>- 社交情绪负面比例                                                      | 未开始     |
| 讨论热度     | 1    | - 当前讨论热度                                                                                | 未开始     |
| 社区活跃度   | 2    | - 社区发帖数量<br>- 社区互动频率                                                                | 未开始     |

### 4. 链上数据 (12维)

| 特征组           | 维度 | 具体含义                                                                                     | 完成状态   |
|------------------|------|----------------------------------------------------------------------------------------------|------------|
| 活跃地址         | 2    | - 活跃地址数<br>- 活跃地址增长率                                                              | 未开始     |
| 交易数据         | 2    | - 交易数量<br>- 平均交易金额                                                                    | 未开始     |
| 网络算力与难度   | 2    | - 网络算力<br>- 挖矿难度                                                                        | 未开始     |
| 矿工收益         | 1    | - 矿工收益率                                                                                    | 未开始     |
| UTXO分布        | 2    | - UTXO总数<br>- 大额UTXO比例                                                                    | 未开始     |
| 大额交易         | 1    | - 大额交易占比                                                                                  | 未开始     |
| 网络手续费       | 1    | - 平均网络手续费                                                                                | 未开始     |
| 链上流动性       | 1    | - 链上流动性指标                                                                                | 未开始     |

### 5. 宏观与监管数据 (15维)

| 特征组             | 维度 | 具体含义                                                                                     | 完成状态   |
|--------------------|------|----------------------------------------------------------------------------------------------|------------|
| 宏观经济指标       | 3    | - GDP增长率<br>- 通货膨胀率<br>- 失业率                                                        | 未开始     |
| 股指表现           | 2    | - 主要股指涨跌幅<br>- 股指波动率                                                                | 未开始     |
| 大宗商品价格       | 2    | - 原油价格<br>- 黄金价格                                                                        | 未开始     |
| 货币供应量         | 1    | - 货币供应量M2                                                                                 | 未开始     |
| 监管政策           | 3    | - 主要市场监管政策指数<br>- 新出台政策影响度<br>- 监管环境变化率                                | 未开始     |
| 汇率数据           | 4    | - 美元指数<br>- 欧元汇率<br>- 日元汇率<br>- 人民币汇率                                          | 未开始     |

### 6. 市场深度数据 (6维)

| 特征组     | 维度 | 具体含义                                                                                     | 完成状态   |
|------------|------|----------------------------------------------------------------------------------------------|------------|
| 买单深度   | 3    | - 买单价格层级1<br>- 买单价格层级2<br>- 买单价格层级3                                          | 未开始     |
| 卖单深度   | 3    | - 卖单价格层级1<br>- 卖单价格层级2<br>- 卖单价格层级3                                          | 未开始     |

## 二、优化后的输出特征 (20维)

### 1. 交易信号 (8维)

| 特征组       | 维度 | 具体含义                                                                                     | 完成状态   |
|--------------|------|----------------------------------------------------------------------------------------------|------------|
| 买入信号     | 1    | - 买入建议                                                                                    | 未开始     |
| 卖出信号     | 1    | - 卖出建议                                                                                    | 未开始     |
| 持有信号     | 1    | - 持有建议                                                                                    | 未开始     |
| 信号强度     | 1    | - 信号的强度评分                                                                              | 未开始     |
| 建议持仓比例 | 1    | - 推荐的持仓比例                                                                              | 未开始     |
| 预测价格     | 3    | - 预测收盘价<br>- 预测价格上限<br>- 预测价格下限                                              | 未开始     |

### 2. 风险评估 (12维)

| 特征组           | 维度 | 具体含义                                                                                     | 完成状态   |
|------------------|------|----------------------------------------------------------------------------------------------|------------|
| 波动风险         | 1    | - 市场波动风险评分                                                                            | 未开始     |
| 趋势可靠度       | 1    | - 趋势判断的可靠性评分                                                                        | 未开始     |
| 流动性风险       | 1    | - 流动性不足风险评分                                                                          | 未开始     |
| 技术面评分       | 2    | - 技术指标综合评分<br>- 技术指标一致性评分                                                    | 未开始     |
| 基本面评分       | 2    | - 宏观经济评分<br>- 链上数据评分                                                              | 未开始     |
| 市场时机评分     | 2    | - 当前市场周期评分<br>- 市场情绪评分                                                          | 未开始     |
| 综合建议得分     | 3    | - 综合风险评分<br>- 总体建议评分<br>- 风险调整后收益评分                                      | 未开始     |

---

请按照以上任务清单逐项完成，每个任务完成后讲由人类手动更新相应的“完成状态”。
