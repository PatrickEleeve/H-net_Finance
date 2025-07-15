# H-Net股票市场实时分析 - 完整使用指南

## 📋 目录
1. [环境要求](#环境要求)
2. [数据要求与准备](#数据要求与准备)
3. [安装步骤](#安装步骤)
4. [数据预处理](#数据预处理)
5. [模型训练](#模型训练)
6. [实时推理](#实时推理)
7. [性能优化](#性能优化)
8. [故障排除](#故障排除)

---

## 🖥️ 环境要求

### 硬件要求
```
推荐配置:
- GPU: NVIDIA RTX 4090 (24GB VRAM) 或更高
- CPU: Intel i9-12900K 或 AMD Ryzen 9 5900X
- RAM: 64GB DDR4
- 存储: 1TB NVMe SSD

最低配置:
- GPU: NVIDIA RTX 3080 (10GB VRAM)
- CPU: Intel i7-10700K 或 AMD Ryzen 7 3700X
- RAM: 32GB DDR4
- 存储: 500GB SSD
```

### 软件环境
```
操作系统: Ubuntu 20.04+ / Windows 10+ / macOS 12+
Python: 3.9-3.11
CUDA: 11.8+ (for GPU training)
Docker: 20.10+ (可选)
```

---

## 📊 数据要求与准备

### 数据格式要求

#### 1. 价格数据 (OHLCVA)
```python
# CSV格式，每行一个时间点
# 文件名: price_data.csv
timestamp,open,high,low,close,volume,adj_close
2024-01-01 09:30:00,150.25,150.80,149.90,150.50,1000000,150.50
2024-01-01 09:31:00,150.50,151.20,150.30,151.00,1200000,151.00
...
```

#### 2. 技术指标数据
```python
# 文件名: technical_indicators.csv
# 20个技术指标列
timestamp,sma_5,sma_10,sma_20,sma_50,ema_12,ema_26,rsi_14,macd_line,macd_signal,macd_hist,bb_upper,bb_lower,bb_percent,vol_sma,vol_ratio,price_vol,momentum_1,momentum_5,momentum_10,hl_spread
2024-01-01 09:30:00,150.1,149.8,148.5,147.2,150.3,149.1,65.5,0.8,0.6,0.2,152.1,148.9,0.6,950000,1.05,2.1,0.002,0.015,0.08,0.006
...
```

#### 3. 新闻情感数据
```python
# 文件名: news_sentiment.csv
# 768维BERT嵌入向量
timestamp,embed_0,embed_1,embed_2,...,embed_767
2024-01-01 09:30:00,0.125,-0.334,0.891,...,0.445
...
```

### 数据获取建议

#### 免费数据源
```python
# 价格数据
import yfinance as yf
import pandas as pd

# 获取股票数据
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2024-01-01", interval="1m")
data.to_csv(f"{ticker}_price_data.csv")
```

#### 付费数据源
- **Alpha Vantage**: 实时和历史数据
- **Quandl**: 高质量金融数据
- **Bloomberg API**: 专业级数据
- **Refinitiv Eikon**: 机构级数据

---

## 🛠️ 安装步骤

### 1. 创建Python环境
```bash
# 使用conda
conda create -n hnet-stock python=3.10
conda activate hnet-stock

# 或使用venv
python -m venv hnet-stock
source hnet-stock/bin/activate  # Linux/Mac
# 或 hnet-stock\Scripts\activate  # Windows
```

### 2. 安装依赖包
```bash
# 核心深度学习框架
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 数据处理
pip install pandas==2.1.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.2

# 金融数据
pip install yfinance==0.2.24
pip install ta-lib==0.4.28
pip install pandas-ta==0.3.14b

# 可视化
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
pip install plotly==5.17.0

# 模型优化
pip install optuna==3.4.0
pip install onnx==1.15.0
pip install onnxruntime-gpu==1.16.3

# 文本处理(新闻情感)
pip install transformers==4.36.2
pip install sentence-transformers==2.2.2

# 其他工具
pip install tqdm==4.66.1
pip install tensorboard==2.15.1
pip install wandb==0.16.1  # 可选，用于实验跟踪
```

### 3. 验证安装
```python
# test_installation.py
import torch
import pandas as pd
import numpy as np

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# 测试基本功能
x = torch.randn(2, 3).cuda() if torch.cuda.is_available() else torch.randn(2, 3)
print("Installation successful!")
```

---

## 🔄 数据预处理

### 1. 创建数据预处理脚本
```python
# data_preprocessing.py
import pandas as pd
import numpy as np
import ta
from transformers import AutoTokenizer, AutoModel
import torch
import os
from tqdm import tqdm

class StockDataPreprocessor:
    def __init__(self, raw_data_path, output_path):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        self.sentiment_model = None
        
    def load_price_data(self, symbol):
        """加载价格数据"""
        file_path = f"{self.raw_data_path}/{symbol}_price.csv"
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        return df.sort_values('timestamp')
    
    def compute_technical_indicators(self, df):
        """计算技术指标"""
        # 使用ta库计算技术指标
        indicators = pd.DataFrame()
        
        # 移动平均线
        indicators['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
        indicators['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        indicators['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        indicators['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        
        # 指数移动平均
        indicators['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        indicators['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        
        # RSI
        indicators['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        indicators['macd_line'] = macd.macd()
        indicators['macd_signal'] = macd.macd_signal()
        indicators['macd_hist'] = macd.macd_diff()
        
        # 布林带
        bb = ta.volatility.BollingerBands(df['close'])
        indicators['bb_upper'] = bb.bollinger_hband()
        indicators['bb_lower'] = bb.bollinger_lband()
        indicators['bb_percent'] = bb.bollinger_pband()
        
        # 成交量指标
        indicators['vol_sma'] = ta.volume.volume_sma(df['close'], df['volume'])
        indicators['vol_ratio'] = df['volume'] / indicators['vol_sma']
        
        # 价格波动率
        indicators['price_vol'] = df['close'].rolling(20).std()
        
        # 动量指标
        indicators['momentum_1'] = df['close'].pct_change(1)
        indicators['momentum_5'] = df['close'].pct_change(5)
        indicators['momentum_10'] = df['close'].pct_change(10)
        
        # 高低价差
        indicators['hl_spread'] = (df['high'] - df['low']) / df['close']
        
        return indicators.fillna(0)
    
    def process_news_sentiment(self, news_file):
        """处理新闻情感数据"""
        if self.sentiment_model is None:
            self.sentiment_model = AutoModel.from_pretrained('ProsusAI/finbert')
            self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        
        # 读取新闻数据
        news_df = pd.read_csv(news_file)
        
        embeddings = []
        for text in tqdm(news_df['content'], desc="Processing news"):
            inputs = self.tokenizer(text, return_tensors='pt', 
                                  truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                # 使用[CLS]标记的嵌入
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(embedding.flatten())
        
        return np.array(embeddings)
    
    def create_sequences(self, price_data, technical_data, news_data, 
                        sequence_length=1024, prediction_horizon=20):
        """创建训练序列"""
        sequences = []
        
        total_length = len(price_data)
        
        for i in range(sequence_length, total_length - prediction_horizon):
            # 输入序列
            price_seq = price_data[i-sequence_length:i]
            tech_seq = technical_data[i-sequence_length:i]
            news_seq = news_data[i-sequence_length:i]
            
            # 预测目标
            future_prices = price_data[i:i+prediction_horizon, 3]  # close prices
            future_vols = technical_data[i:i+prediction_horizon, -4]  # price volatility
            
            # 方向标签 (0: 下跌, 1: 持平, 2: 上涨)
            price_changes = np.diff(future_prices)
            directions = np.where(price_changes < -0.001, 0,
                                np.where(price_changes > 0.001, 2, 1))
            
            sequences.append({
                'price': price_seq,
                'technical': tech_seq,
                'news': news_seq,
                'target_price': future_prices,
                'target_volatility': future_vols,
                'target_direction': directions
            })
        
        return sequences
    
    def save_processed_data(self, sequences, split_ratios=(0.7, 0.15, 0.15)):
        """保存处理后的数据"""
        total_samples = len(sequences)
        train_size = int(total_samples * split_ratios[0])
        val_size = int(total_samples * split_ratios[1])
        
        # 分割数据
        train_data = sequences[:train_size]
        val_data = sequences[train_size:train_size + val_size]
        test_data = sequences[train_size + val_size:]
        
        # 保存数据
        for split, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            os.makedirs(f"{self.output_path}/{split}", exist_ok=True)
            
            # 分别保存不同类型的数据
            price_data = np.array([seq['price'] for seq in data])
            tech_data = np.array([seq['technical'] for seq in data])
            news_data = np.array([seq['news'] for seq in data])
            
            np.save(f"{self.output_path}/{split}/price_data.npy", price_data)
            np.save(f"{self.output_path}/{split}/technical_data.npy", tech_data)
            np.save(f"{self.output_path}/{split}/news_data.npy", news_data)
            
            # 保存目标数据
            targets = {
                'price': np.array([seq['target_price'] for seq in data]),
                'volatility': np.array([seq['target_volatility'] for seq in data]),
                'direction': np.array([seq['target_direction'] for seq in data])
            }
            
            np.save(f"{self.output_path}/{split}/targets.npy", targets)
        
        print(f"Data saved: Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

# 使用示例
if __name__ == "__main__":
    preprocessor = StockDataPreprocessor("raw_data", "processed_data")
    
    # 处理单个股票
    symbol = "AAPL"
    price_df = preprocessor.load_price_data(symbol)
    technical_df = preprocessor.compute_technical_indicators(price_df)
    
    # 如果有新闻数据
    # news_embeddings = preprocessor.process_news_sentiment("raw_data/news.csv")
    # 否则使用随机嵌入作为占位符
    news_embeddings = np.random.randn(len(price_df), 768)
    
    # 创建序列
    sequences = preprocessor.create_sequences(
        price_df[['open', 'high', 'low', 'close', 'volume', 'adj_close']].values,
        technical_df.values,
        news_embeddings
    )
    
    # 保存数据
    preprocessor.save_processed_data(sequences)
```

### 2. 运行数据预处理
```bash
# 创建目录结构
mkdir -p data/{raw_data,processed_data}

# 下载示例数据
python download_sample_data.py  # 需要创建此脚本

# 运行预处理
python data_preprocessing.py
```

---

## 🚀 模型训练

### 1. 基础训练
```bash
# 使用默认配置训练
python hnet_stock_training.py --mode train --data_path processed_data/

# 自定义配置训练
python hnet_stock_training.py --mode train \
    --data_path processed_data/ \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --max_epochs 50
```

### 2. 分布式训练（多GPU）
```bash
# 使用torch.distributed
torchrun --nproc_per_node=4 hnet_stock_training.py \
    --mode train \
    --data_path processed_data/ \
    --batch_size 8  # 每个GPU的batch size
```

### 3. 使用配置文件训练
```yaml
# config.yaml
model:
  d_model: 512
  num_stages: 2
  encoder_layers: 4
  decoder_layers: 4
  main_layers: 16
  chunk_ratios: [4, 3]
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 5e-5
  weight_decay: 1e-5
  max_epochs: 100
  warmup_steps: 1000

data:
  sequence_length: 1024
  prediction_horizon: 20
  price_features: 6
  technical_features: 20
  news_embed_dim: 768
```

```python
# train_with_config.py
import yaml
from hnet_stock_training import HNetConfig, StockTrainer, StockDataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # 合并配置
    config = HNetConfig()
    for section, params in config_dict.items():
        for key, value in params.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return config

if __name__ == "__main__":
    config = load_config("config.yaml")
    trainer = StockTrainer(config)
    
    train_dataset = StockDataset('processed_data/train/', config)
    val_dataset = StockDataset('processed_data/val/', config)
    
    trainer.train(train_dataset, val_dataset)
```

### 4. 超参数优化
```bash
# 自动超参数搜索
python hnet_stock_training.py --mode optimize --data_path processed_data/
```

### 5. 训练监控
```python
# 使用TensorBoard
tensorboard --logdir runs/

# 使用Weights & Biases (可选)
import wandb

wandb.init(project="hnet-stock-analysis")
# 在训练代码中添加日志记录
wandb.log({"train_loss": train_loss, "val_loss": val_loss})
```

---

## 🔮 实时推理

### 1. 模型加载和推理
```python
# real_time_inference.py
from hnet_stock_training import RealTimeInference, HNetConfig
import numpy as np
import time

# 初始化推理引擎
config = HNetConfig()
inference = RealTimeInference("best_stock_hnet.pth", config)

# 模拟实时数据流
while True:
    # 获取新数据 (这里需要连接到实际数据源)
    new_price = np.random.randn(6)  # OHLCVA
    new_technical = np.random.randn(20)  # 技术指标
    new_news = np.random.randn(768)  # 新闻嵌入
    
    # 更新数据缓冲区
    inference.update_data(new_price, new_technical, new_news)
    
    # 进行预测
    predictions = inference.predict()
    
    print(f"Price forecast: {predictions['price_forecast'][:5]}")
    print(f"Volatility forecast: {predictions['volatility_forecast'][:5]}")
    print(f"Direction probabilities: {predictions['direction_probs'][:5]}")
    
    time.sleep(60)  # 每分钟预测一次
```

### 2. REST API服务
```python
# api_server.py
from flask import Flask, request, jsonify
from hnet_stock_training import RealTimeInference, HNetConfig
import numpy as np

app = Flask(__name__)

# 初始化模型
config = HNetConfig()
inference = RealTimeInference("best_stock_hnet.pth", config)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # 解析输入数据
    price_data = np.array(data['price'])
    technical_data = np.array(data['technical'])
    news_data = np.array(data['news'])
    
    # 更新数据并预测
    inference.update_data(price_data, technical_data, news_data)
    predictions = inference.predict()
    
    return jsonify(predictions)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 3. 使用Docker部署
```dockerfile
# Dockerfile
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "api_server.py"]
```

```bash
# 构建和运行
docker build -t hnet-stock-api .
docker run -p 5000:5000 --gpus all hnet-stock-api
```

---

## ⚡ 性能优化

### 1. 模型量化
```python
# quantize_model.py
from hnet_stock_training import ModelOptimizer, StockDataset, HNetConfig

config = HNetConfig()
dataset = StockDataset('processed_data/test/', config)
dataloader = DataLoader(dataset, batch_size=1)

# 量化模型
quantized_model = ModelOptimizer.quantize_model(model, dataloader)

# 保存量化模型
torch.save(quantized_model.state_dict(), 'quantized_model.pth')
```

### 2. ONNX导出
```python
# export_onnx.py
from hnet_stock_training import ModelOptimizer, StockHNet, HNetConfig

config = HNetConfig()
model = StockHNet(config)
model.load_state_dict(torch.load('best_stock_hnet.pth')['model_state_dict'])

# 导出ONNX
ModelOptimizer.export_to_onnx(model, config, 'stock_hnet.onnx')
```

### 3. 推理加速
```python
# 使用ONNX Runtime推理
import onnxruntime as ort

session = ort.InferenceSession('stock_hnet.onnx', providers=['CUDAExecutionProvider'])

def fast_predict(price_data, technical_data, news_data):
    inputs = {
        'price_data': price_data.numpy(),
        'technical_data': technical_data.numpy(),
        'news_data': news_data.numpy()
    }
    
    outputs = session.run(None, inputs)
    return outputs[0]
```

---

## 🔧 故障排除

### 常见问题和解决方案

#### 1. 内存不足
```bash
# 减小batch size
python hnet_stock_training.py --batch_size 8

# 使用梯度累积
python hnet_stock_training.py --gradient_accumulation_steps 4
```

#### 2. CUDA内存错误
```python
# 在代码中添加内存清理
torch.cuda.empty_cache()

# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### 3. 训练不收敛
- 检查学习率设置
- 验证数据预处理是否正确
- 尝试不同的优化器

#### 4. 数据加载慢
```python
# 增加数据加载worker数量
dataloader = DataLoader(dataset, batch_size=32, num_workers=8, pin_memory=True)
```

### 日志和调试
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
```

---

## 📚 性能基准测试

### 预期性能指标
```
在标准配置下（RTX 4090，16GB数据）:
- 训练时间: ~2-3小时/epoch
- 推理延迟: <50ms
- 内存使用: ~18GB GPU内存
- 方向准确率: >55%
- 价格预测MAE: <0.5%
```

### 基准测试脚本
```python
# benchmark.py
import time
import torch
from hnet_stock_training import StockHNet, HNetConfig

def benchmark_inference(model, config, num_samples=1000):
    model.eval()
    
    # 准备测试数据
    price_data = torch.randn(1, config.sequence_length, config.price_features)
    technical_data = torch.randn(1, config.sequence_length, config.technical_features)
    news_data = torch.randn(1, config.sequence_length, config.news_embed_dim)
    
    # 预热
    for _ in range(10):
        with torch.no_grad():
            _ = model(price_data, technical_data, news_data)
    
    # 基准测试
    start_time = time.time()
    for _ in range(num_samples):
        with torch.no_grad():
            _ = model(price_data, technical_data, news_data)
    
    end_time = time.time()
    avg_latency = (end_time - start_time) / num_samples * 1000  # ms
    
    print(f"Average inference latency: {avg_latency:.2f}ms")
    return avg_latency

if __name__ == "__main__":
    config = HNetConfig()
    model = StockHNet(config)
    benchmark_inference(model, config)
```

---

## 🎯 下一步建议

1. **数据质量提升**: 集成更多高质量数据源
2. **模型架构优化**: 尝试不同的压缩比例和层数配置
3. **特征工程**: 添加更多金融特定特征
4. **在线学习**: 实现增量学习机制
5. **风险管理**: 添加不确定性量化

---

**注意**: 这是一个研究原型，实际部署前请进行充分的回测和风险评估。金融市场预测存在固有风险，使用时请谨慎。