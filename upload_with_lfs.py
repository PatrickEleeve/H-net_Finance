#!/usr/bin/env python3
"""
使用Git LFS上传大型数据文件的脚本
"""

import os
import subprocess
import sys
from datetime import datetime

def check_git_lfs():
    """检查Git LFS是否安装"""
    try:
        result = subprocess.run("git lfs version", shell=True, capture_output=True, text=True, check=True)
        print(f"✅ Git LFS已安装: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("❌ Git LFS未安装")
        print("💡 安装Git LFS:")
        print("   macOS: brew install git-lfs")
        print("   Windows: 从 https://git-lfs.github.com 下载")
        print("   Linux: sudo apt-get install git-lfs")
        return False

def setup_git_lfs():
    """设置Git LFS"""
    print("🔧 设置Git LFS...")
    
    commands = [
        ("git lfs install", "初始化Git LFS"),
        ("git lfs track '*.npy'", "跟踪.npy文件"),
        ("git lfs track '*.npz'", "跟踪.npz文件"),
        ("git lfs track 'stock_data/**/*.npy'", "跟踪数据目录中的.npy文件"),
        ("git lfs track 'stock_data/**/*.npz'", "跟踪数据目录中的.npz文件"),
    ]
    
    for command, description in commands:
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            print(f"✅ {description}")
            if result.stdout:
                print(f"   输出: {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} 失败: {e}")
            return False
    
    return True

def reset_and_recommit():
    """重置之前的提交并重新提交"""
    print("🔄 重置之前的提交...")
    
    try:
        # 重置到上一个提交
        subprocess.run("git reset HEAD~1", shell=True, check=True)
        print("✅ 已重置到上一个提交")
        
        # 添加.gitattributes文件
        subprocess.run("git add .gitattributes", shell=True, check=True)
        subprocess.run("git commit -m 'Add Git LFS configuration'", shell=True, check=True)
        print("✅ 已提交Git LFS配置")
        
        # 重新添加数据文件
        subprocess.run("git add .", shell=True, check=True)
        
        # 创建新的提交
        commit_message = f"""Add training dataset with Git LFS

📊 Dataset Summary:
- Total samples: 925
- Training samples: 647
- Validation samples: 138  
- Test samples: 140
- Sequence length: 60
- Prediction horizon: 5

🎯 Dataset Features:
- Multi-modal data (price + technical + news)
- Large files handled with Git LFS
- 5 stocks: AAPL, MSFT, GOOGL, TSLA, NVDA

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        subprocess.run(f'git commit -m "{commit_message}"', shell=True, check=True)
        print("✅ 已创建新的数据提交")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 重置和重新提交失败: {e}")
        return False

def push_with_lfs():
    """使用Git LFS推送"""
    print("📤 使用Git LFS推送数据...")
    
    try:
        result = subprocess.run("git push origin main", shell=True, capture_output=True, text=True, check=True)
        print("✅ 数据成功推送到GitHub!")
        if result.stdout:
            print(f"   输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 推送失败: {e}")
        if e.stderr:
            print(f"   错误信息: {e.stderr.strip()}")
        return False

def show_lfs_status():
    """显示Git LFS状态"""
    print("\n📊 Git LFS状态:")
    
    try:
        # 显示LFS跟踪的文件
        result = subprocess.run("git lfs ls-files", shell=True, capture_output=True, text=True)
        if result.stdout:
            print("🔍 LFS跟踪的文件:")
            for line in result.stdout.strip().split('\n')[:10]:  # 显示前10个
                print(f"   {line}")
        
        # 显示LFS状态
        result = subprocess.run("git lfs status", shell=True, capture_output=True, text=True)
        if result.stdout:
            print(f"\n📈 LFS状态:")
            print(f"   {result.stdout.strip()}")
            
    except Exception as e:
        print(f"⚠️  无法获取LFS状态: {e}")

def create_data_readme():
    """创建数据说明文档"""
    readme_content = """# H-Net 训练数据集

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
"""
    
    with open("stock_data/README.md", 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("📄 已创建数据说明文档: stock_data/README.md")

def main():
    """主函数"""
    print("🚀 Git LFS 大文件上传解决方案")
    print("=" * 50)
    
    # 1. 检查Git LFS
    if not check_git_lfs():
        print("\n💡 请先安装Git LFS，然后重新运行此脚本")
        sys.exit(1)
    
    # 2. 设置Git LFS
    if not setup_git_lfs():
        print("❌ Git LFS设置失败")
        sys.exit(1)
    
    # 3. 创建数据说明文档
    create_data_readme()
    
    # 4. 重置并重新提交
    if not reset_and_recommit():
        print("❌ 重新提交失败")
        sys.exit(1)
    
    # 5. 推送数据
    if push_with_lfs():
        print("\n🎉 数据成功上传到GitHub!")
        print("🔗 仓库地址: https://github.com/PatrickEleeve/H-net_Finance")
        
        # 显示LFS状态
        show_lfs_status()
        
        print(f"\n📋 上传摘要:")
        print(f"✅ 使用Git LFS处理大文件")
        print(f"✅ 所有数据文件已上传")
        print(f"✅ 创建了数据说明文档")
        
        print(f"\n🎯 下一步:")
        print(f"1. 其他用户clone时需要: git lfs pull")
        print(f"2. 查看GitHub上的数据文件")
        print(f"3. 开始使用数据训练模型")
        
    else:
        print("❌ 数据上传失败")
        print("💡 请检查网络连接和GitHub权限")
        sys.exit(1)

if __name__ == "__main__":
    main()
