# H-Net 训练数据集

## 📊 数据集概览

本数据集包含用于H-Net股票分析模型的训练数据，涵盖5只热门股票的多模态金融数据。

### 基本信息
- **股票代码**: AAPL, MSFT, GOOGL, TSLA, NVDA
- **总样本数**: 925
- **训练集**: 647 样本 (70%)
- **验证集**: 138 样本 (15%)
- **测试集**: 140 样本 (15%)
- **序列长度**: 60 (时间步)
- **预测窗口**: 5 (未来5个时间步)

### 数据类型

#### 1. 价格数据 (`*_price.npy`)
- **维度**: (样本数, 60, 6)
- **特征**: open, high, low, close, adj_close, volume
- **来源**: Yahoo Finance
- **频率**: 日线数据

#### 2. 技术指标 (`*_technical.npy`)
- **维度**: (样本数, 60, 20)
- **指标**: SMA, EMA, RSI, MACD, 布林带, 成交量指标等
- **计算**: 基于TA-Lib库

#### 3. 新闻情感 (`*_news.npy`)
- **维度**: (样本数, 60, 768)
- **特征**: 新闻情感嵌入向量
- **处理**: TextBlob情感分析 + 特征工程

#### 4. 预测目标 (`*_targets.npz`)
- **price**: 未来价格序列
- **volatility**: 波动率预测
- **direction**: 方向分类 (0=下跌, 1=横盘, 2=上涨)

## 📁 文件结构

```
stock_data/
├── train/                      # 训练数据
│   ├── merged_dataset_price.npy    # 价格数据
│   ├── merged_dataset_technical.npy # 技术指标
│   ├── merged_dataset_news.npy     # 新闻情感
│   └── merged_dataset_targets.npz  # 预测目标
├── val/                        # 验证数据
│   └── ...
├── test/                       # 测试数据
│   └── ...
├── *_raw_price.csv            # 原始价格数据
├── *_technical.csv            # 技术指标数据
├── raw_news.csv               # 原始新闻数据
└── merged_dataset_metadata.json # 元数据
```

## 🚀 使用方法

### 加载数据
```python
import numpy as np

# 加载训练数据
price_data = np.load('stock_data/train/merged_dataset_price.npy')
technical_data = np.load('stock_data/train/merged_dataset_technical.npy')
news_data = np.load('stock_data/train/merged_dataset_news.npy')

# 加载目标
targets = np.load('stock_data/train/merged_dataset_targets.npz')
price_targets = targets['price']
volatility_targets = targets['volatility'] 
direction_targets = targets['direction']
```

### 开始训练
```bash
# 快速训练
python train_launcher.py --mode quick

# 完整训练
python train_launcher.py --mode thorough
```

## 📈 数据质量

- ✅ **完整性**: 所有时间序列无缺失值
- ✅ **一致性**: 统一的时间对齐和特征格式
- ✅ **平衡性**: 训练/验证/测试比例合理
- ✅ **真实性**: 基于真实市场数据生成

## ⚠️ 使用须知

1. **Git LFS**: 大文件通过Git LFS管理，clone时需要安装Git LFS
2. **内存要求**: 加载完整数据集需要约2GB内存
3. **版权**: 数据仅供研究和学习使用
4. **更新**: 数据集基于历史数据，投资需谨慎

## 🔄 数据更新

要重新生成数据集:
```bash
python hnet_data_preprocess.py
```

---
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
