#!/usr/bin/env python3
"""
将训练数据上传到GitHub的脚本
"""

import os
import subprocess
import sys
import shutil
from datetime import datetime

def check_data_files():
    """检查数据文件完整性"""
    print("🔍 检查数据文件...")
    
    data_dir = "stock_data"
    required_dirs = ["train", "val", "test"]
    required_files = ["merged_dataset_price.npy", "merged_dataset_technical.npy", 
                     "merged_dataset_news.npy", "merged_dataset_targets.npz"]
    
    if not os.path.exists(data_dir):
        print(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    # 检查元数据文件
    metadata_file = os.path.join(data_dir, "merged_dataset_metadata.json")
    if not os.path.exists(metadata_file):
        print("❌ 找不到元数据文件")
        return False
    
    # 检查训练数据
    missing_files = []
    for split in required_dirs:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"❌ 找不到{split}数据目录")
            return False
        
        for file in required_files:
            file_path = os.path.join(split_dir, file)
            if not os.path.exists(file_path):
                missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ 缺少数据文件: {missing_files}")
        return False
    
    print("✅ 所有必要的数据文件都存在")
    return True

def calculate_data_size():
    """计算数据文件总大小"""
    total_size = 0
    data_dir = "stock_data"
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            total_size += os.path.getsize(file_path)
    
    # 转换为MB
    size_mb = total_size / (1024 * 1024)
    
    print(f"📊 数据文件总大小: {size_mb:.2f} MB")
    
    if size_mb > 100:
        print("⚠️  数据文件较大，上传可能需要较长时间")
        return False, size_mb
    
    return True, size_mb

def update_gitignore_for_data():
    """更新.gitignore以允许数据文件"""
    print("🔧 更新.gitignore文件...")
    
    gitignore_path = ".gitignore"
    
    # 读取现有内容
    with open(gitignore_path, 'r') as f:
        content = f.read()
    
    # 添加例外规则，允许stock_data目录
    data_exception = '''
# 允许stock_data目录中的数据文件
!stock_data/
!stock_data/**/*.npy
!stock_data/**/*.npz
!stock_data/**/*.csv
!stock_data/**/*.json
'''
    
    if "!stock_data/" not in content:
        content += data_exception
        
        with open(gitignore_path, 'w') as f:
            f.write(content)
        
        print("✅ 已更新.gitignore以允许数据文件")
    else:
        print("✅ .gitignore已配置为允许数据文件")

def run_git_command(command, description):
    """运行Git命令"""
    print(f"🔧 {description}")
    print(f"   命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        if result.stdout:
            print(f"   输出: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 错误: {e}")
        if e.stderr:
            print(f"   错误信息: {e.stderr.strip()}")
        return False

def upload_data_to_github():
    """上传数据到GitHub"""
    print("📤 上传数据到GitHub...")
    
    # 添加所有文件
    if not run_git_command("git add .", "添加所有文件"):
        return False
    
    # 创建提交信息
    commit_message = f"""Add training dataset

📊 Dataset Summary:
- Total samples: 925 (from metadata)
- Training samples: 647
- Validation samples: 138  
- Test samples: 140
- Sequence length: 60
- Prediction horizon: 5

🎯 Dataset Features:
- Multi-modal data (price + technical + news)
- 5 stocks: AAPL, MSFT, GOOGL, TSLA, NVDA
- Ready for H-Net training

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    # 提交更改
    if not run_git_command(f'git commit -m "{commit_message}"', "创建数据提交"):
        return False
    
    # 推送到GitHub
    if not run_git_command("git push origin main", "推送到GitHub"):
        return False
    
    return True

def show_upload_summary():
    """显示上传摘要"""
    print("\n🎉 数据上传完成!")
    print("=" * 50)
    
    print("📁 已上传的数据文件:")
    for root, dirs, files in os.walk("stock_data"):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   {file_path} ({file_size:.1f} KB)")
    
    print(f"\n🔗 GitHub仓库: https://github.com/PatrickEleeve/H-net_Finance")
    
    print(f"\n🚀 下一步:")
    print(f"1. 访问GitHub查看上传的数据")
    print(f"2. 其他开发者可以clone仓库获取数据")
    print(f"3. 运行训练: python train_launcher.py")
    print(f"4. 分享项目链接")

def main():
    """主函数"""
    print("📦 H-Net训练数据上传到GitHub")
    print("=" * 50)
    
    # 1. 检查数据文件
    if not check_data_files():
        print("❌ 数据文件检查失败")
        sys.exit(1)
    
    # 2. 计算数据大小
    size_ok, size_mb = calculate_data_size()
    if not size_ok:
        response = input(f"数据文件较大({size_mb:.2f}MB)，是否继续上传? (y/N): ").strip().lower()
        if response != 'y':
            print("❌ 上传取消")
            sys.exit(0)
    
    # 3. 更新.gitignore
    update_gitignore_for_data()
    
    # 4. 用户确认
    print(f"\n📋 准备上传:")
    print(f"   数据大小: {size_mb:.2f} MB")
    print(f"   目标仓库: https://github.com/PatrickEleeve/H-net_Finance")
    print(f"   数据目录: stock_data/")
    
    response = input(f"\n确认上传训练数据到GitHub? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ 上传取消")
        sys.exit(0)
    
    # 5. 上传数据
    try:
        if upload_data_to_github():
            show_upload_summary()
        else:
            print("❌ 上传过程中出现错误")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
