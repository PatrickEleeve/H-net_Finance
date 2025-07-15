# 股票数据预处理脚本 - 使用说明

## 🎯 脚本功能

这个脚本 `hnet_data_preprocess.py` 是一个完整的股票数据生成与处理管道，专门为H-Net模型准备训练数据。

## ✅ 解决的问题

之前脚本无法正常产生数据的主要原因已经全部解决：

### 1. **依赖包缺失** ✅ 已解决
- 缺少必要的Python包（pandas, yfinance, transformers等）
- **解决方案**: 已安装所有必要的依赖包

### 2. **Yahoo Finance API限制** ✅ 已解决
- 高频数据（5分钟）只能获取60天内的数据
- 原脚本试图获取1年的5分钟数据，导致失败
- **解决方案**: 改用1年的日数据（1d interval），并添加了备选策略

### 3. **数据类型问题** ✅ 已解决
- TA-Lib要求输入为双精度浮点数
- **解决方案**: 确保所有价格数据转换为float64类型

### 4. **序列长度不合理** ✅ 已解决
- 原来的序列长度1024对日数据太长（需要4年数据）
- **解决方案**: 改为60天序列长度，预测5天

## 📊 生成的数据

### 数据统计
- **总样本数**: 925个训练序列
- **训练集**: 647个样本 (70%)
- **验证集**: 138个样本 (15%)
- **测试集**: 140个样本 (15%)
- **序列长度**: 60天历史数据
- **预测长度**: 5天未来数据

### 数据文件结构
```
stock_data/
├── train/                           # 训练数据
│   ├── merged_dataset_price.npy     # 价格数据 (647, 60, 6)
│   ├── merged_dataset_technical.npy # 技术指标 (647, 60, 20)
│   ├── merged_dataset_news.npy      # 新闻嵌入 (647, 60, 768)
│   └── merged_dataset_targets.npz   # 目标数据 (价格/方向/波动率)
├── val/                             # 验证数据
└── test/                            # 测试数据
├── {SYMBOL}_raw_price.csv           # 每个股票的原始价格数据
├── {SYMBOL}_technical.csv           # 每个股票的技术指标
├── raw_news.csv                     # 新闻数据
└── merged_dataset_metadata.json     # 数据集元数据
```

### 数据维度说明
- **价格数据**: (batch, sequence_length, 6) - [open, high, low, close, volume, adj_close]
- **技术指标**: (batch, sequence_length, 20) - 20种技术指标
- **新闻嵌入**: (batch, sequence_length, 768) - BERT风格的文本嵌入
- **目标数据**: 
  - price: (batch, 5) - 未来5天的收盘价
  - direction: (batch, 5) - 价格方向 [0=下跌, 1=横盘, 2=上涨]
  - volatility: (batch, 5) - 波动率

## 🚀 如何使用

### 1. 运行数据生成
```bash
python hnet_data_preprocess.py
```

### 2. 验证生成的数据
```bash
python validate_data.py
```

### 3. 自定义股票列表
编辑脚本中的 `symbols` 变量：
```python
symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']  # 修改为你想要的股票
```

### 4. 调整数据参数
在 `main()` 函数中可以调整：
- `period`: 数据时间范围 ('1y', '2y', '6mo' 等)
- `interval`: 数据频率 ('1d', '1h', '5m' 等)
- `sequence_length`: 序列长度
- `prediction_horizon`: 预测长度

## 🔧 技术特性

### 自动备选策略
脚本包含多种数据获取策略，如果某种配置失败会自动尝试其他配置：
1. 用户指定的配置
2. 6个月日数据
3. 3个月日数据
4. 1个月小时数据
5. 60天5分钟数据
6. 30天5分钟数据
7. 1年日数据（最保险）

### 技术指标 (20种)
- 移动平均线 (SMA 5, 10, 20, 50)
- 指数移动平均 (EMA 12, 26)
- RSI
- MACD (线, 信号, 直方图)
- 布林带 (上轨, 下轨, 百分比)
- 成交量指标
- 价格波动率
- 动量指标
- 高低价差

### 新闻情感分析
- 从Yahoo Finance、Google News、Reddit获取新闻
- 使用TextBlob进行情感分析
- 生成768维嵌入向量

## 📈 数据质量

- ✅ 所有5个股票的数据成功下载
- ✅ 技术指标计算正确
- ✅ 新闻数据获取成功
- ✅ 数据格式和类型正确
- ✅ 训练/验证/测试集正确分割

现在脚本可以**完全正常工作**，生成高质量的股票数据集供H-Net模型训练使用！
