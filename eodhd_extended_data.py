#!/usr/bin/env python3
"""
EODHD 免费版本优化策略 - 使用更多股票和更长时间的日线数据
"""

import os
import sys
import logging
from datetime import datetime
from hnet_data_preprocess import StockDataGenerator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_extended_stock_list():
    """获取扩展的股票列表来增加数据量"""
    stock_categories = {
        "科技股": ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'CRM'],
        "金融股": ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'AXP', 'V', 'MA', 'PYPL'],
        "医疗股": ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'LLY', 'MRK', 'AMGN', 'GILD'],
        "消费股": ['PG', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW'],
        "工业股": ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'FDX'],
        "能源股": ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PXD', 'KMI', 'OXY', 'PSX', 'VLO']
    }
    
    print("📊 扩展股票列表选择:")
    print("=" * 50)
    
    for i, (category, stocks) in enumerate(stock_categories.items(), 1):
        print(f"{i}. {category}: {len(stocks)}只股票")
        print(f"   {stocks[:5]}... (显示前5只)")
    
    print(f"\n7. 全部选择: {sum(len(stocks) for stocks in stock_categories.values())}只股票")
    print(f"8. 自定义选择")
    
    choice = input(f"\n请选择股票类别 (1-8): ").strip()
    
    if choice == "7":
        # 全部股票
        all_stocks = []
        for stocks in stock_categories.values():
            all_stocks.extend(stocks)
        return all_stocks
    elif choice == "8":
        # 自定义
        custom_stocks = input("请输入股票代码，用逗号分隔: ").strip().upper().split(',')
        return [s.strip() for s in custom_stocks if s.strip()]
    else:
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(stock_categories):
                category_name = list(stock_categories.keys())[choice_idx]
                return stock_categories[category_name]
        except ValueError:
            pass
        
        # 默认返回原始列表
        return ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']

def calculate_total_samples(num_stocks, period_years):
    """计算总样本数量"""
    # 每只股票每年约250个交易日
    # 序列长度60，所以每只股票每年约190个样本
    samples_per_stock_per_year = 190
    total_samples = num_stocks * period_years * samples_per_stock_per_year
    return int(total_samples)

def main():
    """主函数"""
    print("🚀 EODHD 免费版本优化策略")
    print("=" * 50)
    print("💡 免费版本限制: 只能使用日线数据")
    print("📈 增加数据量策略: 更多股票 + 更长时间")
    
    # 检查API密钥
    api_key = os.getenv('EODHD_API_KEY')
    if not api_key:
        print("❌ 未找到EODHD API密钥")
        print("💡 请先运行: python setup_eodhd.py")
        sys.exit(1)
    
    print(f"✅ 找到API密钥: {api_key[:8]}...")
    
    # 选择股票
    selected_stocks = get_extended_stock_list()
    print(f"\n✅ 已选择 {len(selected_stocks)} 只股票")
    print(f"   股票列表: {selected_stocks}")
    
    # 选择时间周期
    print(f"\n📅 时间周期选择:")
    period_options = [
        ("1年", "1y", 1),
        ("2年", "2y", 2), 
        ("3年", "3y", 3),
        ("5年", "5y", 5),
        ("10年", "10y", 10)
    ]
    
    for i, (name, code, years) in enumerate(period_options, 1):
        estimated_samples = calculate_total_samples(len(selected_stocks), years)
        print(f"{i}. {name}: ~{estimated_samples:,} 个样本")
    
    period_choice = input(f"\n请选择时间周期 (1-{len(period_options)}): ").strip()
    
    try:
        period_idx = int(period_choice) - 1
        if 0 <= period_idx < len(period_options):
            period_name, period_code, period_years = period_options[period_idx]
        else:
            period_name, period_code, period_years = period_options[1]  # 默认2年
    except ValueError:
        period_name, period_code, period_years = period_options[1]  # 默认2年
    
    # 计算预期数据量
    estimated_samples = calculate_total_samples(len(selected_stocks), period_years)
    
    print(f"\n✅ 已选择: {period_name}")
    print(f"   股票数量: {len(selected_stocks)}")
    print(f"   时间周期: {period_years}年")
    print(f"   预期样本: ~{estimated_samples:,} 个")
    print(f"   相比原始: 增加 {estimated_samples/925:.1f} 倍")
    
    # 过拟合分析
    if estimated_samples > 5000:
        overfitting_risk = "🟢 低"
    elif estimated_samples > 2000:
        overfitting_risk = "🟡 中"
    else:
        overfitting_risk = "🔴 高"
    
    print(f"   过拟合风险: {overfitting_risk}")
    
    # 用户确认
    response = input(f"\n是否开始数据预处理? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ 操作取消")
        sys.exit(0)
    
    try:
        print(f"\n⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 初始化数据生成器
        generator = StockDataGenerator("stock_data_eodhd_extended", eodhd_api_key=api_key)
        
        # 修改数据生成器以使用EODHD日线数据
        print(f"🔧 初始化数据生成器...")
        print(f"📊 将处理 {len(selected_stocks)} 只股票")
        print(f"💾 数据将保存到: stock_data_eodhd_extended/")
        
        # 生成数据集
        generator.generate_multi_symbol_dataset(
            symbols=selected_stocks,
            merge_data=True,
            use_eodhd=True,
            interval="1d",      # 日线数据
            period=period_code  # 选择的周期
        )
        
        print(f"\n✅ 数据预处理完成!")
        print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 显示生成的文件
        print(f"\n📁 生成的文件:")
        total_size = 0
        for root, dirs, files in os.walk("stock_data_eodhd_extended"):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    total_size += file_size
                    print(f"   {file_path} ({file_size:.1f} KB)")
        
        print(f"\n📊 数据集统计:")
        print(f"   总大小: {total_size/1024:.1f} MB")
        print(f"   股票数量: {len(selected_stocks)}")
        print(f"   时间周期: {period_years}年")
        print(f"   预期样本: ~{estimated_samples:,} 个")
        
        print(f"\n🎯 防过拟合分析:")
        print(f"   数据增长: {estimated_samples/925:.1f}倍")
        print(f"   过拟合风险: {overfitting_risk}")
        print(f"   建议训练轮数: {min(50, max(10, estimated_samples//1000))} epochs")
        
        print(f"\n🚀 下一步:")
        print(f"1. 验证数据: python validate_data.py --data-dir stock_data_eodhd_extended")
        print(f"2. 开始训练: python train_launcher.py --data-dir stock_data_eodhd_extended")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  操作被用户中断")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"数据预处理过程中出现错误: {e}")
        logger.exception("详细错误信息:")
        sys.exit(1)

if __name__ == "__main__":
    main()
