# H-Net Stock Market Analysis

一个基于H-Net架构的股票市场实时分析模型，集成多模态数据（价格、技术指标、新闻情感）进行股价预测。

## 🎯 项目特色

- **多模态融合**: 集成价格数据、技术指标和新闻情感分析
- **H-Net架构**: 采用动态分块和分层序列建模
- **实时预测**: 支持价格预测、波动率预测和方向分类
- **完整管道**: 从数据收集到模型训练的端到端解决方案


## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd H-net_Finance

# 安装依赖
pip install torch numpy pandas yfinance transformers textblob feedparser beautifulsoup4 tqdm
```

### 2. 数据生成

```bash
# 生成训练数据（包含AAPL, MSFT, GOOGL, TSLA, NVDA）
python hnet_data_preprocess.py
```

### 3. 模型训练

```bash
# 快速训练（5个epoch，3-5分钟）
python train_launcher.py --mode quick

# 20个epoch训练（8-12分钟）
python train_20epochs.py

# 交互式选择训练模式
python train_launcher.py --interactive
```

### 4. 模型测试

```bash
# 测试训练好的模型
python test_model.py
```

## 📁 项目结构

```
H-net_Finance/
├── hnet_data_preprocess.py     # 数据预处理和生成
├── hnet_stock_training.py      # H-Net模型定义和训练
├── start_training.py           # 简易训练启动器
├── train_launcher.py           # 多模式训练启动器
├── train_20epochs.py           # 20轮训练脚本
├── test_model.py               # 模型测试和评估
├── validate_data.py            # 数据验证脚本
├── stock_data/                 # 生成的训练数据
│   ├── train/                  # 训练集
│   ├── val/                    # 验证集
│   └── test/                   # 测试集
├── best_stock_hnet.pth         # 最佳训练模型
└── requirements.txt            # 依赖包列表
```

## 🔧 训练模式

| 模式 | 时间 | 参数量 | 轮数 | 适用场景 |
|------|------|--------|------|----------|
| quick | 3-5分钟 | 1.3M | 5 | 快速验证 |
| medium | 8-12分钟 | 3.2M | 20 | 标准训练 |
| balanced | 10-15分钟 | 3.2M | 15 | 平衡性能 |
| thorough | 30-60分钟 | 11.6M | 30 | 最佳性能 |

## 📈 数据来源

- **价格数据**: Yahoo Finance API
- **技术指标**: TA-Lib (SMA, EMA, RSI, MACD, 布林带等)
- **新闻情感**: 多源新闻聚合 + 情感分析

## 🧠 模型架构

- **输入层**: 多模态特征融合（价格6维 + 技术指标20维 + 新闻768维）
- **编码器**: 分层Mamba编码器
- **动态分块**: 基于市场制度变化的自适应分块
- **主网络**: Transformer块进行序列建模
- **解码器**: 分层解码器重构序列
- **输出头**: 多任务预测（价格、波动率、方向）

## 🎯 使用案例

### 实时推理

```python
from hnet_stock_training import RealTimeInference, HNetConfig

# 加载模型
config = HNetConfig()
inference = RealTimeInference('best_stock_hnet.pth', config)

# 更新数据并预测
inference.update_data(new_price, new_technical, new_news)
predictions = inference.predict()

print(f"价格预测: {predictions['price_forecast']}")
print(f"方向概率: {predictions['direction_probs']}")
```

### 批量评估

```python
from hnet_stock_training import EvaluationMetrics

# 评估模型性能
metrics = EvaluationMetrics.evaluate_model_performance(model, test_loader)
print(f"方向准确率: {metrics['direction_accuracy']:.2%}")
```

## 📋 系统要求

- Python 3.8+
- PyTorch 1.9+
- 8GB+ RAM (推荐)
- CPU/GPU 支持

## 📜 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 联系

如有问题，请创建Issue或联系维护者。

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！
