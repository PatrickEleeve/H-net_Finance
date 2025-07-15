#!/usr/bin/env python3
"""
完整的股票数据生成与处理管道
从零开始构建H-Net所需的所有数据类型
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import talib as ta
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
from textblob import TextBlob
import feedparser
import re
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StockDataGenerator:
    """完整的股票数据生成器"""
    
    def __init__(self, output_dir="generated_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化新闻情感分析模型
        self.sentiment_analyzer = None
        self.news_model = None
        self.news_tokenizer = None
        
    def download_price_data(self, symbols, period="1y", interval="1d"):
        """
        下载股票价格数据
        
        Args:
            symbols: 股票代码列表 ['AAPL', 'MSFT', 'GOOGL']
            period: 时间周期 '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
            interval: 时间间隔 '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        """
        logger.info(f"Downloading price data for {len(symbols)} symbols...")
        logger.info(f"Using period={period}, interval={interval}")
        
        all_data = {}
        
        # 定义备选的期间和间隔组合
        fallback_configs = [
            (period, interval),  # 用户指定的配置
            ("6mo", "1d"),       # 6个月的日数据
            ("3mo", "1d"),       # 3个月的日数据
            ("1mo", "1h"),       # 1个月的小时数据
            ("60d", "5m"),       # 60天的5分钟数据
            ("30d", "5m"),       # 30天的5分钟数据
            ("1y", "1d"),        # 1年的日数据（最保险的选择）
        ]
        
        for symbol in tqdm(symbols, desc="Downloading price data"):
            data = None
            config_used = None
            
            # 尝试不同的配置直到成功
            for config_period, config_interval in fallback_configs:
                try:
                    logger.info(f"Trying {symbol} with period={config_period}, interval={config_interval}")
                    
                    # 下载数据
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period=config_period, interval=config_interval)
                    
                    if not data.empty:
                        config_used = (config_period, config_interval)
                        logger.info(f"Success for {symbol} with {config_used}")
                        break
                    else:
                        logger.warning(f"Empty data for {symbol} with period={config_period}, interval={config_interval}")
                        
                except Exception as e:
                    logger.warning(f"Failed for {symbol} with period={config_period}, interval={config_interval}: {e}")
                    continue
            
            if data is None or data.empty:
                logger.error(f"Failed to get any data for {symbol} with all configurations")
                continue            # 重命名列为标准格式
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # 添加调整后收盘价
            data['adj_close'] = data['close']
            
            # 重置索引，将时间作为列
            data = data.reset_index()
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'timestamp'})
            elif 'Date' in data.columns:
                data = data.rename(columns={'Date': 'timestamp'})
                
            # 保存原始数据
            data.to_csv(f"{self.output_dir}/{symbol}_raw_price.csv", index=False)
            all_data[symbol] = data
            
            logger.info(f"Downloaded {len(data)} records for {symbol} using config {config_used}")
            
            # 避免API限制
            time.sleep(0.1)
            
        return all_data
    
    def compute_technical_indicators(self, price_data, symbol):
        """
        计算技术指标
        """
        logger.info(f"Computing technical indicators for {symbol}...")
        
        df = price_data.copy()
        indicators = pd.DataFrame(index=df.index)
        
        # 价格数组 - 确保类型为float64
        open_prices = df['open'].astype(np.float64).values
        high_prices = df['high'].astype(np.float64).values
        low_prices = df['low'].astype(np.float64).values
        close_prices = df['close'].astype(np.float64).values
        volume = df['volume'].astype(np.float64).values
        
        try:
            # 1. 移动平均线 (Simple Moving Average)
            indicators['sma_5'] = ta.SMA(close_prices, timeperiod=5)
            indicators['sma_10'] = ta.SMA(close_prices, timeperiod=10)
            indicators['sma_20'] = ta.SMA(close_prices, timeperiod=20)
            indicators['sma_50'] = ta.SMA(close_prices, timeperiod=50)
            
            # 2. 指数移动平均 (Exponential Moving Average)
            indicators['ema_12'] = ta.EMA(close_prices, timeperiod=12)
            indicators['ema_26'] = ta.EMA(close_prices, timeperiod=26)
            
            # 3. RSI (Relative Strength Index)
            indicators['rsi_14'] = ta.RSI(close_prices, timeperiod=14)
            
            # 4. MACD
            macd, macd_signal, macd_hist = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            indicators['macd_line'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_hist'] = macd_hist
            
            # 5. 布林带 (Bollinger Bands)  
            bb_upper, bb_middle, bb_lower = ta.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            indicators['bb_upper'] = bb_upper
            indicators['bb_lower'] = bb_lower
            indicators['bb_percent'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
            
            # 6. 成交量指标
            indicators['vol_sma'] = ta.SMA(volume, timeperiod=20)
            indicators['vol_ratio'] = volume / indicators['vol_sma']
            
            # 7. 波动率 (Price Volatility)
            indicators['price_vol'] = ta.STDDEV(close_prices, timeperiod=20, nbdev=1)
            
            # 8. 动量指标 (Momentum)
            indicators['momentum_1'] = ta.MOM(close_prices, timeperiod=1)
            indicators['momentum_5'] = ta.MOM(close_prices, timeperiod=5)
            indicators['momentum_10'] = ta.MOM(close_prices, timeperiod=10)
            
            # 9. 高低价差
            indicators['hl_spread'] = (high_prices - low_prices) / close_prices
            
            # 填充NaN值 - 使用新的pandas语法
            indicators = indicators.bfill().fillna(0)
            
            # 保存技术指标
            indicators.to_csv(f"{self.output_dir}/{symbol}_technical.csv", index=False)
            
            logger.info(f"Generated {len(indicators.columns)} technical indicators for {symbol}")
            
        except Exception as e:
            logger.error(f"Error computing technical indicators for {symbol}: {e}")
            # 如果计算失败，生成零填充的指标
            indicators = pd.DataFrame(np.zeros((len(df), 20)), 
                                    columns=[f'indicator_{i}' for i in range(20)])
        
        return indicators
    
    def collect_news_data(self, symbols, days_back=30):
        """
        收集新闻数据 - 使用多种免费新闻源
        """
        logger.info(f"Collecting news data for {len(symbols)} symbols...")
        
        all_news = []
        
        for symbol in symbols:
            # 1. Yahoo Finance新闻
            news_data = self._get_yahoo_news(symbol)
            all_news.extend(news_data)
            
            # 2. RSS新闻源
            rss_news = self._get_rss_news(symbol)
            all_news.extend(rss_news)
            
            # 3. Reddit情感 (r/stocks, r/investing)
            reddit_news = self._get_reddit_sentiment(symbol)
            all_news.extend(reddit_news)
            
            time.sleep(1)  # 避免被限制
        
        # 保存新闻数据
        news_df = pd.DataFrame(all_news)
        if not news_df.empty:
            news_df = news_df.drop_duplicates(subset=['title', 'symbol'])
            news_df.to_csv(f"{self.output_dir}/raw_news.csv", index=False)
            logger.info(f"Collected {len(news_df)} news articles")
        
        return news_df
    
    def _get_yahoo_news(self, symbol):
        """获取Yahoo Finance新闻"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            news_data = []
            for article in news[:10]:  # 取前10条
                news_data.append({
                    'symbol': symbol,
                    'timestamp': datetime.fromtimestamp(article.get('providerPublishTime', time.time())),
                    'title': article.get('title', ''),
                    'content': article.get('summary', ''),
                    'source': 'yahoo'
                })
            
            return news_data
            
        except Exception as e:
            logger.warning(f"Error getting Yahoo news for {symbol}: {e}")
            return []
    
    def _get_rss_news(self, symbol):
        """从RSS源获取新闻"""
        news_data = []
        
        # Google News RSS
        try:
            url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=en-US&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:5]:
                news_data.append({
                    'symbol': symbol,
                    'timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 24)),
                    'title': entry.title,
                    'content': entry.get('summary', entry.title),
                    'source': 'google_news'
                })
                
        except Exception as e:
            logger.warning(f"Error getting RSS news for {symbol}: {e}")
        
        return news_data
    
    def _get_reddit_sentiment(self, symbol):
        """获取Reddit情感数据（模拟）"""
        # 这里可以接入Reddit API或使用现有的Reddit数据
        # 暂时生成模拟数据
        news_data = []
        
        sentiments = ['positive', 'negative', 'neutral']
        reddit_texts = [
            f"{symbol} looking bullish today",
            f"Great earnings report from {symbol}",
            f"{symbol} stock analysis - buy or sell?",
            f"Technical analysis of {symbol}",
            f"{symbol} breaking resistance levels"
        ]
        
        for i in range(5):
            news_data.append({
                'symbol': symbol,
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 48)),
                'title': f"Reddit discussion: {reddit_texts[i]}",
                'content': f"Reddit sentiment about {symbol}: {np.random.choice(sentiments)}",
                'source': 'reddit'
            })
        
        return news_data
    
    def process_news_sentiment(self, news_df, method='finbert'):
        """
        处理新闻情感，生成嵌入向量
        
        Args:
            news_df: 新闻数据DataFrame
            method: 'finbert', 'textblob', 'transformer'
        """
        logger.info("Processing news sentiment...")
        
        if news_df.empty:
            logger.warning("No news data to process, generating random embeddings")
            return self._generate_random_embeddings(1000, 768)
        
        if method == 'finbert':
            return self._process_with_finbert(news_df)
        elif method == 'textblob':
            return self._process_with_textblob(news_df)
        elif method == 'transformer':
            return self._process_with_transformer(news_df)
        else:
            return self._generate_random_embeddings(len(news_df), 768)
    
    def _process_with_finbert(self, news_df):
        """使用FinBERT处理新闻情感"""
        try:
            # 初始化FinBERT模型
            if self.news_model is None:
                self.news_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
                self.news_model = AutoModel.from_pretrained('ProsusAI/finbert')
            
            embeddings = []
            
            for _, row in tqdm(news_df.iterrows(), total=len(news_df), desc="Processing with FinBERT"):
                text = f"{row['title']} {row['content']}"
                
                # 分词和编码
                inputs = self.news_tokenizer(text, return_tensors='pt', 
                                           truncation=True, max_length=512, padding=True)
                
                # 获取嵌入
                with torch.no_grad():
                    outputs = self.news_model(**inputs)
                    # 使用[CLS]标记的嵌入
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    embeddings.append(embedding)
            
            embeddings_array = np.array(embeddings)
            logger.info(f"Generated FinBERT embeddings: {embeddings_array.shape}")
            
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Error with FinBERT: {e}")
            return self._generate_random_embeddings(len(news_df), 768)
    
    def _process_with_textblob(self, news_df):
        """使用TextBlob进行简单情感分析"""
        try:
            sentiment_features = []
            
            for _, row in news_df.iterrows():
                text = f"{row['title']} {row['content']}"
                blob = TextBlob(text)
                
                # 基础情感特征
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # 扩展为768维向量（模拟BERT嵌入）
                feature_vector = np.zeros(768)
                feature_vector[0] = polarity
                feature_vector[1] = subjectivity
                
                # 添加文本长度、关键词等特征
                feature_vector[2] = len(text) / 1000.0  # 归一化文本长度
                feature_vector[3] = text.lower().count('buy') - text.lower().count('sell')
                feature_vector[4] = text.lower().count('bullish') - text.lower().count('bearish')
                
                # 用随机噪声填充剩余维度
                feature_vector[5:] = np.random.normal(0, 0.1, 763)
                
                sentiment_features.append(feature_vector)
            
            embeddings_array = np.array(sentiment_features)
            logger.info(f"Generated TextBlob embeddings: {embeddings_array.shape}")
            
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Error with TextBlob: {e}")
            return self._generate_random_embeddings(len(news_df), 768)
    
    def _process_with_transformer(self, news_df):
        """使用通用Transformer模型"""
        try:
            # 使用情感分析pipeline
            if self.sentiment_analyzer is None:
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                 model="nlptown/bert-base-multilingual-uncased-sentiment")
            
            embeddings = []
            
            for _, row in news_df.iterrows():
                text = f"{row['title']} {row['content']}"
                
                # 获取情感分析结果
                sentiment = self.sentiment_analyzer(text[:512])  # 限制长度
                
                # 创建特征向量
                feature_vector = np.zeros(768)
                
                # 情感分数
                if sentiment[0]['label'] == 'POSITIVE':
                    feature_vector[0] = sentiment[0]['score']
                else:
                    feature_vector[0] = -sentiment[0]['score']
                
                # 添加其他特征
                feature_vector[1] = len(text) / 1000.0
                feature_vector[2:] = np.random.normal(0, 0.1, 766)
                
                embeddings.append(feature_vector)
            
            embeddings_array = np.array(embeddings)
            logger.info(f"Generated Transformer embeddings: {embeddings_array.shape}")
            
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Error with Transformer: {e}")
            return self._generate_random_embeddings(len(news_df), 768)
    
    def _generate_random_embeddings(self, num_samples, embed_dim):
        """生成随机嵌入向量作为后备方案"""
        logger.info(f"Generating {num_samples} random embeddings of dimension {embed_dim}")
        return np.random.normal(0, 0.1, (num_samples, embed_dim))
    
    def align_timestamps(self, price_data, technical_data, news_embeddings, symbol):
        """
        对齐所有数据的时间戳
        """
        logger.info(f"Aligning timestamps for {symbol}...")
        
        # 获取价格数据的时间戳
        price_timestamps = pd.to_datetime(price_data['timestamp'])
        
        # 创建对齐后的数据结构
        aligned_data = {
            'timestamp': price_timestamps,
            'price': price_data[['open', 'high', 'low', 'close', 'volume', 'adj_close']].values,
            'technical': technical_data.values,
            'news': np.zeros((len(price_timestamps), 768))
        }
        
        # 为每个时间点分配新闻嵌入
        if len(news_embeddings) > 0:
            # 简单策略：循环使用新闻嵌入
            for i in range(len(price_timestamps)):
                news_idx = i % len(news_embeddings)
                aligned_data['news'][i] = news_embeddings[news_idx]
        else:
            # 如果没有新闻数据，使用随机嵌入
            aligned_data['news'] = self._generate_random_embeddings(len(price_timestamps), 768)
        
        return aligned_data
    
    def create_training_sequences(self, aligned_data, sequence_length=60, prediction_horizon=5):
        """
        创建训练序列
        """
        logger.info("Creating training sequences...")
        
        price_data = aligned_data['price']
        technical_data = aligned_data['technical']
        news_data = aligned_data['news']
        
        sequences = []
        total_length = len(price_data)
        
        for i in range(sequence_length, total_length - prediction_horizon):
            # 输入序列
            price_seq = price_data[i-sequence_length:i]
            tech_seq = technical_data[i-sequence_length:i]
            news_seq = news_data[i-sequence_length:i]
            
            # 预测目标
            future_prices = price_data[i:i+prediction_horizon, 3]  # close prices
            
            # 计算未来波动率
            if i + prediction_horizon < total_length:
                future_returns = np.diff(np.log(future_prices + 1e-8))
                future_volatility = np.std(future_returns) * np.sqrt(252)  # 年化波动率
            else:
                future_volatility = 0.1  # 默认值
            
            # 方向标签
            current_price = price_data[i-1, 3]
            final_price = future_prices[-1]
            price_change_pct = (final_price - current_price) / current_price
            
            if price_change_pct < -0.01:  # 下跌超过1%
                direction = 0
            elif price_change_pct > 0.01:  # 上涨超过1%
                direction = 2
            else:  # 横盘
                direction = 1
            
            sequences.append({
                'price': price_seq.astype(np.float32),
                'technical': tech_seq.astype(np.float32),
                'news': news_seq.astype(np.float32),
                'target_price': future_prices.astype(np.float32),
                'target_volatility': np.full(prediction_horizon, future_volatility, dtype=np.float32),
                'target_direction': np.full(prediction_horizon, direction, dtype=np.int64)
            })
        
        logger.info(f"Created {len(sequences)} training sequences")
        return sequences
    
    def save_processed_data(self, sequences, symbol, split_ratios=(0.7, 0.15, 0.15)):
        """
        保存处理后的数据
        """
        logger.info(f"Saving processed data for {symbol}...")
        
        if not sequences:
            logger.error("No sequences to save!")
            return
        
        total_samples = len(sequences)
        train_size = int(total_samples * split_ratios[0])
        val_size = int(total_samples * split_ratios[1])
        
        # 分割数据
        train_data = sequences[:train_size]
        val_data = sequences[train_size:train_size + val_size]
        test_data = sequences[train_size + val_size:]
        
        # 为每个分割创建目录
        for split, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            if not data:
                continue
                
            split_dir = f"{self.output_dir}/{split}"
            os.makedirs(split_dir, exist_ok=True)
            
            # 转换为numpy数组并保存
            price_array = np.array([seq['price'] for seq in data])
            tech_array = np.array([seq['technical'] for seq in data])
            news_array = np.array([seq['news'] for seq in data])
            
            target_price = np.array([seq['target_price'] for seq in data])
            target_vol = np.array([seq['target_volatility'] for seq in data])
            target_dir = np.array([seq['target_direction'] for seq in data])
            
            # 保存数据
            np.save(f"{split_dir}/{symbol}_price.npy", price_array)
            np.save(f"{split_dir}/{symbol}_technical.npy", tech_array)
            np.save(f"{split_dir}/{symbol}_news.npy", news_array)
            
            # 保存目标
            np.savez(f"{split_dir}/{symbol}_targets.npz", 
                    price=target_price,
                    volatility=target_vol,
                    direction=target_dir)
            
            logger.info(f"{split}: {len(data)} samples saved")
        
        # 保存元数据
        metadata = {
            'symbol': symbol,
            'total_samples': total_samples,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'sequence_length': len(sequences[0]['price']) if sequences else 0,
            'prediction_horizon': len(sequences[0]['target_price']) if sequences else 0,
            'created_at': datetime.now().isoformat()
        }
        
        with open(f"{self.output_dir}/{symbol}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_multi_symbol_dataset(self, symbols, merge_data=True):
        """
        生成多股票数据集
        """
        logger.info(f"Generating multi-symbol dataset for {symbols}")
        
        all_sequences = []
        
        for symbol in symbols:
            logger.info(f"Processing {symbol}...")
            
            # 1. 下载价格数据 - 修改为合理的时间间隔组合
            price_data_dict = self.download_price_data([symbol], period="1y", interval="1d")
            
            if symbol not in price_data_dict:
                logger.warning(f"No price data for {symbol}, skipping...")
                continue
            
            price_data = price_data_dict[symbol]
            
            # 2. 计算技术指标
            technical_data = self.compute_technical_indicators(price_data, symbol)
            
            # 3. 收集新闻数据
            news_df = self.collect_news_data([symbol])
            
            # 4. 处理新闻情感
            news_embeddings = self.process_news_sentiment(news_df, method='textblob')
            
            # 5. 对齐时间戳
            aligned_data = self.align_timestamps(price_data, technical_data, news_embeddings, symbol)
            
            # 6. 创建训练序列
            sequences = self.create_training_sequences(aligned_data)
            
            if sequences:
                if merge_data:
                    all_sequences.extend(sequences)
                else:
                    # 单独保存每个股票的数据
                    self.save_processed_data(sequences, symbol)
            
            logger.info(f"Completed processing {symbol}")
        
        # 如果合并数据，保存为一个大数据集
        if merge_data and all_sequences:
            self.save_processed_data(all_sequences, "merged_dataset")
        
        logger.info("Multi-symbol dataset generation completed!")

def main():
    """主函数 - 数据生成示例"""
    
    # 初始化数据生成器
    generator = StockDataGenerator("stock_data")
    
    # 定义要处理的股票
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print("🚀 开始生成股票数据集...")
    print(f"📊 处理股票: {symbols}")
    print(f"💾 输出目录: stock_data/")
    
    # 生成数据集
    generator.generate_multi_symbol_dataset(symbols, merge_data=True)
    
    print("✅ 数据生成完成!")
    print("\n📁 生成的文件:")
    for root, dirs, files in os.walk("stock_data"):
        for file in files:
            print(f"   {os.path.join(root, file)}")

if __name__ == "__main__":
    main()