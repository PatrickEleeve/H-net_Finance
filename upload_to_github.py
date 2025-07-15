#!/usr/bin/env python3
"""
H-Net项目GitHub上传脚本
自动初始化Git仓库并上传到GitHub
"""

import os
import sys
import subprocess
import json
from datetime import datetime

def run_command(command, description="", check=True):
    """运行命令并处理错误"""
    print(f"🔧 {description}")
    print(f"   命令: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=check)
        if result.stdout:
            print(f"   输出: {result.stdout.strip()}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ 错误: {e}")
        if e.stderr:
            print(f"   错误信息: {e.stderr.strip()}")
        if check:
            sys.exit(1)
        return e

def check_git_installed():
    """检查Git是否安装"""
    try:
        result = subprocess.run("git --version", shell=True, capture_output=True, text=True, check=True)
        print(f"✅ Git已安装: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("❌ Git未安装，请先安装Git")
        print("   macOS: brew install git")
        print("   Windows: https://git-scm.com/download/win")
        print("   Linux: sudo apt-get install git")
        return False

def check_github_cli():
    """检查GitHub CLI是否安装"""
    try:
        result = subprocess.run("gh --version", shell=True, capture_output=True, text=True, check=True)
        print(f"✅ GitHub CLI已安装: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  GitHub CLI未安装")
        print("   可选安装: brew install gh")
        return False

def get_project_info():
    """获取项目信息"""
    print("\n📝 配置项目信息")
    print("=" * 50)
    
    # 获取用户输入
    github_username = input("请输入GitHub用户名: ").strip()
    if not github_username:
        print("❌ 用户名不能为空")
        sys.exit(1)
    
    repo_name = input("请输入仓库名称 (默认: H-net_Finance): ").strip()
    if not repo_name:
        repo_name = "H-net_Finance"
    
    repo_description = input("请输入仓库描述 (可选): ").strip()
    if not repo_description:
        repo_description = "H-Net Stock Market Analysis - Multi-modal financial time series prediction"
    
    is_private = input("是否设为私有仓库? (y/N): ").strip().lower() == 'y'
    
    return {
        'username': github_username,
        'repo_name': repo_name,
        'description': repo_description,
        'private': is_private,
        'repo_url': f"https://github.com/{github_username}/{repo_name}.git"
    }

def create_commit_message():
    """创建提交信息"""
    return f"""Initial commit: H-Net Stock Market Analysis

🎯 Features:
- Multi-modal financial data fusion (price + technical + news)
- H-Net architecture with dynamic chunking
- Real-time stock prediction (price, volatility, direction)
- Complete training pipeline with 57% directional accuracy

📊 Model Performance:
- Direction prediction: 57% accuracy (vs 33.3% random)
- Model sizes: 1.3M - 11.6M parameters
- Training time: 3-60 minutes

🚀 Quick Start:
1. pip install -r requirements.txt
2. python hnet_data_preprocess.py
3. python train_launcher.py --mode quick

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

def initialize_git_repo():
    """初始化Git仓库"""
    print("\n🔧 初始化Git仓库")
    print("=" * 30)
    
    # 检查是否已经是Git仓库
    if os.path.exists('.git'):
        print("✅ Git仓库已存在")
        return True
    
    # 初始化仓库
    run_command("git init", "初始化Git仓库")
    
    # 配置Git (如果需要)
    try:
        run_command("git config user.name", "检查Git用户名", check=False)
        run_command("git config user.email", "检查Git邮箱", check=False)
    except:
        print("⚠️  请配置Git用户信息:")
        print("   git config --global user.name '你的名字'")
        print("   git config --global user.email '你的邮箱'")
        
        response = input("是否现在配置? (y/N): ").strip().lower()
        if response == 'y':
            name = input("请输入用户名: ").strip()
            email = input("请输入邮箱: ").strip()
            
            if name and email:
                run_command(f'git config user.name "{name}"', "设置用户名")
                run_command(f'git config user.email "{email}"', "设置邮箱")
    
    return True

def create_github_repo(project_info, use_gh_cli=False):
    """创建GitHub仓库"""
    print(f"\n🌐 创建GitHub仓库: {project_info['repo_name']}")
    print("=" * 40)
    
    if use_gh_cli:
        # 使用GitHub CLI创建仓库
        privacy_flag = "--private" if project_info['private'] else "--public"
        description_flag = f'--description "{project_info["description"]}"'
        
        command = f'gh repo create {project_info["repo_name"]} {privacy_flag} {description_flag}'
        run_command(command, "使用GitHub CLI创建仓库")
        
        return True
    else:
        # 手动创建提示
        print("🔗 请手动在GitHub上创建仓库:")
        print(f"   1. 访问: https://github.com/new")
        print(f"   2. 仓库名: {project_info['repo_name']}")
        print(f"   3. 描述: {project_info['description']}")
        print(f"   4. 设置为: {'私有' if project_info['private'] else '公开'}")
        print(f"   5. 不要初始化README、.gitignore或LICENSE")
        
        input("创建完成后，按回车继续...")
        return True

def upload_to_github(project_info):
    """上传代码到GitHub"""
    print(f"\n📤 上传代码到GitHub")
    print("=" * 30)
    
    # 添加所有文件
    run_command("git add .", "添加所有文件到Git")
    
    # 创建初始提交
    commit_message = create_commit_message()
    run_command(f'git commit -m "{commit_message}"', "创建初始提交")
    
    # 设置主分支
    run_command("git branch -M main", "设置主分支为main")
    
    # 添加远程仓库
    run_command(f'git remote add origin {project_info["repo_url"]}', "添加远程仓库")
    
    # 推送到GitHub
    run_command("git push -u origin main", "推送到GitHub")
    
    return True

def create_project_summary():
    """创建项目摘要"""
    summary = {
        "project": "H-Net Stock Market Analysis",
        "version": "1.0.0",
        "created": datetime.now().isoformat(),
        "features": [
            "Multi-modal data fusion (price + technical + news)",
            "H-Net architecture with dynamic chunking",
            "Real-time prediction (price, volatility, direction)",
            "57% directional accuracy",
            "Complete training pipeline"
        ],
        "files": [
            "hnet_data_preprocess.py - Data preprocessing",
            "hnet_stock_training.py - Model definition and training",
            "train_launcher.py - Multi-mode training launcher",
            "test_model.py - Model testing and evaluation",
            "README.md - Project documentation",
            "requirements.txt - Dependencies"
        ],
        "performance": {
            "direction_accuracy": "57%",
            "model_sizes": "1.3M - 11.6M parameters",
            "training_time": "3-60 minutes"
        }
    }
    
    with open("project_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("📄 项目摘要已保存到: project_summary.json")

def main():
    """主函数"""
    print("🚀 H-Net项目GitHub上传向导")
    print("=" * 50)
    
    # 检查Git
    if not check_git_installed():
        sys.exit(1)
    
    # 检查GitHub CLI (可选)
    has_gh_cli = check_github_cli()
    
    # 获取项目信息
    project_info = get_project_info()
    
    print(f"\n📋 项目配置:")
    print(f"   用户名: {project_info['username']}")
    print(f"   仓库名: {project_info['repo_name']}")
    print(f"   描述: {project_info['description']}")
    print(f"   私有: {'是' if project_info['private'] else '否'}")
    print(f"   URL: {project_info['repo_url']}")
    
    # 确认继续
    response = input(f"\n确认上传项目到GitHub? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ 上传取消")
        sys.exit(0)
    
    try:
        # 1. 创建项目摘要
        create_project_summary()
        
        # 2. 初始化Git仓库
        initialize_git_repo()
        
        # 3. 创建GitHub仓库
        create_github_repo(project_info, has_gh_cli)
        
        # 4. 上传代码
        upload_to_github(project_info)
        
        # 5. 完成
        print(f"\n🎉 项目成功上传到GitHub!")
        print(f"🔗 仓库地址: https://github.com/{project_info['username']}/{project_info['repo_name']}")
        print(f"\n📋 下一步:")
        print(f"   1. 访问仓库页面验证上传")
        print(f"   2. 设置仓库描述和主题标签")
        print(f"   3. 添加Star和Watch")
        print(f"   4. 分享给其他开发者")
        
    except Exception as e:
        print(f"\n❌ 上传过程中出现错误: {e}")
        print(f"💡 请检查:")
        print(f"   1. 网络连接")
        print(f"   2. GitHub登录状态")
        print(f"   3. 仓库名是否已存在")
        sys.exit(1)

if __name__ == "__main__":
    main()
