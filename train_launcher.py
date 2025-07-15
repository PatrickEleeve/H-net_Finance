#!/usr/bin/env python3
"""
H-Net训练配置和启动脚本
提供不同的训练模式选择
"""

import os
import sys
import json
import argparse
from datetime import datetime

def create_training_configs():
    """创建不同的训练配置"""
    
    configs = {
        "quick": {
            "name": "快速训练 (3-5分钟)",
            "d_model": 128,
            "num_stages": 1,
            "encoder_layers": 2,
            "decoder_layers": 2,
            "main_layers": 4,
            "batch_size": 16,
            "max_epochs": 5,
            "learning_rate": 1e-3
        },
        
        "medium": {
            "name": "中等训练 20轮 (8-12分钟)",
            "d_model": 256,
            "num_stages": 2,
            "encoder_layers": 3,
            "decoder_layers": 3,
            "main_layers": 8,
            "batch_size": 8,
            "max_epochs": 20,
            "learning_rate": 5e-4
        },
        
        "balanced": {
            "name": "平衡训练 (10-15分钟)",
            "d_model": 256,
            "num_stages": 2,
            "encoder_layers": 3,
            "decoder_layers": 3,
            "main_layers": 8,
            "batch_size": 8,
            "max_epochs": 15,
            "learning_rate": 1e-4
        },
        
        "thorough": {
            "name": "深度训练 (30-60分钟)",
            "d_model": 512,
            "num_stages": 2,
            "encoder_layers": 4,
            "decoder_layers": 4,
            "main_layers": 12,
            "batch_size": 4,
            "max_epochs": 30,
            "learning_rate": 5e-5
        }
    }
    
    return configs

def save_config(config_name, config_params):
    """保存训练配置到文件"""
    config_file = f"training_config_{config_name}.json"
    
    full_config = {
        "config_name": config_name,
        "timestamp": datetime.now().isoformat(),
        **config_params
    }
    
    with open(config_file, 'w') as f:
        json.dump(full_config, f, indent=2)
    
    return config_file

def run_training_with_config(config_name, config_params):
    """使用指定配置运行训练"""
    print(f"\n🎯 开始{config_params['name']}...")
    
    # 保存配置
    config_file = save_config(config_name, config_params)
    print(f"📝 配置已保存到: {config_file}")
    
    # 创建训练脚本
    training_script = f"""
#!/usr/bin/env python3
import sys
sys.path.append('.')

from hnet_stock_training import *
import json

# 加载配置
with open('{config_file}', 'r') as f:
    saved_config = json.load(f)

# 创建H-Net配置
config = HNetConfig(
    d_model={config_params['d_model']},
    num_stages={config_params['num_stages']},
    encoder_layers={config_params['encoder_layers']},
    decoder_layers={config_params['decoder_layers']},
    main_layers={config_params['main_layers']},
    chunk_ratios=[4, 3],
    batch_size={config_params['batch_size']},
    learning_rate={config_params['learning_rate']},
    max_epochs={config_params['max_epochs']},
    sequence_length=60,
    prediction_horizon=5
)

# 训练
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {{device}}")

trainer = StockTrainer(config, device)

try:
    train_dataset = StockDataset("stock_data", config, 'train')
    val_dataset = StockDataset("stock_data", config, 'val')
    
    print(f"Train samples: {{len(train_dataset)}}")
    print(f"Val samples: {{len(val_dataset)}}")
    
    trainer.train(train_dataset, val_dataset)
    
except Exception as e:
    print(f"Training error: {{e}}")
    raise
"""
    
    # 保存并运行训练脚本
    script_file = f"run_training_{config_name}.py"
    with open(script_file, 'w') as f:
        f.write(training_script)
    
    print(f"🚀 执行训练脚本: {script_file}")
    
    # 运行训练
    os.system(f"python {script_file}")

def main():
    parser = argparse.ArgumentParser(description='H-Net股票分析模型训练')
    parser.add_argument('--mode', choices=['quick', 'medium', 'balanced', 'thorough'], 
                       help='训练模式')
    parser.add_argument('--interactive', action='store_true', 
                       help='交互式选择训练模式')
    
    args = parser.parse_args()
    
    configs = create_training_configs()
    
    print("🚀 H-Net 股票分析模型训练")
    print("=" * 50)
    
    if args.interactive or not args.mode:
        # 交互式模式
        print("\n📋 可用的训练模式:")
        for i, (key, config) in enumerate(configs.items(), 1):
            print(f"  {i}. {key}: {config['name']}")
            print(f"     - 模型大小: {config['d_model']}维, {config['main_layers']}层")
            print(f"     - 训练轮数: {config['max_epochs']} epochs")
            print(f"     - 批次大小: {config['batch_size']}")
            print()
        
        while True:
            try:
                choice = input("请选择训练模式 (1-3) 或 q 退出: ").strip()
                if choice.lower() == 'q':
                    print("❌ 训练取消")
                    sys.exit(0)
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(configs):
                    mode_key = list(configs.keys())[choice_idx]
                    break
                else:
                    print("❌ 无效选择，请重试")
            except ValueError:
                print("❌ 请输入数字")
    else:
        mode_key = args.mode
    
    # 运行选定的训练模式
    config = configs[mode_key]
    print(f"\n✅ 选择了: {config['name']}")
    
    # 确认开始训练
    response = input(f"确认开始训练? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ 训练取消")
        sys.exit(0)
    
    run_training_with_config(mode_key, config)

if __name__ == "__main__":
    main()
