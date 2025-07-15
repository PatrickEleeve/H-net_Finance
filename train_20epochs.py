#!/usr/bin/env python3
"""
直接启动20个epoch的H-Net训练
"""

import os
import sys
import json
from datetime import datetime

def main():
    print("🚀 开始20个epoch的H-Net训练")
    print("=" * 50)
    
    # 检查数据是否存在
    if not os.path.exists("stock_data/train"):
        print("❌ 找不到训练数据，请先运行数据预处理脚本")
        print("   python hnet_data_preprocess.py")
        return
    
    # 创建20epoch训练配置
    config_20epoch = {
        "config_name": "20epoch",
        "timestamp": datetime.now().isoformat(),
        "name": "20轮训练",
        "d_model": 256,
        "num_stages": 2,
        "encoder_layers": 3,
        "decoder_layers": 3,
        "main_layers": 8,
        "batch_size": 8,
        "max_epochs": 20,
        "learning_rate": 5e-4
    }
    
    # 保存配置
    config_file = "training_config_20epoch.json"
    with open(config_file, 'w') as f:
        json.dump(config_20epoch, f, indent=2)
    
    print(f"📝 训练配置已保存到: {config_file}")
    print(f"📊 配置详情:")
    print(f"   - 模型维度: {config_20epoch['d_model']}")
    print(f"   - 训练轮数: {config_20epoch['max_epochs']}")
    print(f"   - 批次大小: {config_20epoch['batch_size']}")
    print(f"   - 学习率: {config_20epoch['learning_rate']}")
    print(f"   - 预估时间: 8-12分钟")
    
    # 确认开始训练
    response = input(f"\n确认开始20个epoch的训练? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ 训练取消")
        return
    
    # 创建训练脚本
    training_script = f'''
#!/usr/bin/env python3
import sys
sys.path.append('.')

from hnet_stock_training import *
import json

# 加载配置
with open('{config_file}', 'r') as f:
    saved_config = json.load(f)

print("🎯 开始20个epoch训练...")
print(f"配置: {{saved_config['name']}}")

# 创建H-Net配置
config = HNetConfig(
    d_model={config_20epoch['d_model']},
    num_stages={config_20epoch['num_stages']},
    encoder_layers={config_20epoch['encoder_layers']},
    decoder_layers={config_20epoch['decoder_layers']},
    main_layers={config_20epoch['main_layers']},
    chunk_ratios=[4, 3],
    batch_size={config_20epoch['batch_size']},
    learning_rate={config_20epoch['learning_rate']},
    max_epochs={config_20epoch['max_epochs']},
    sequence_length=60,
    prediction_horizon=5
)

# 训练
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {{device}}")

trainer = StockTrainer(config, device)

try:
    train_dataset = StockDataset("stock_data", config, 'train')
    val_dataset = StockDataset("stock_data", config, 'val')
    
    print(f"训练样本: {{len(train_dataset)}}")
    print(f"验证样本: {{len(val_dataset)}}")
    
    print("\\n开始训练...")
    trainer.train(train_dataset, val_dataset)
    
    print("\\n✅ 20个epoch训练完成!")
    print("📁 模型已保存到: best_stock_hnet.pth")
    
except Exception as e:
    print(f"❌ 训练出错: {{e}}")
    raise
'''
    
    # 保存并运行训练脚本
    script_file = "run_training_20epoch.py"
    with open(script_file, 'w') as f:
        f.write(training_script)
    
    print(f"\n🚀 开始执行20个epoch训练...")
    print(f"📝 训练脚本: {script_file}")
    print(f"⏱️  预计耗时: 8-12分钟")
    print("=" * 50)
    
    # 执行训练
    os.system(f"python {script_file}")
    
    print("\n" + "=" * 50)
    print("🎯 训练完成! 检查结果:")
    print("   - best_stock_hnet.pth (训练好的模型)")
    print("   - training_config_20epoch.json (训练配置)")
    print("\n💡 运行测试:")
    print("   python test_model.py")

if __name__ == "__main__":
    main()
