#!/usr/bin/env python3
"""
Quick Results Summary Script
Run this after experiments to get a clean summary of all results
"""

import os
import glob
import re
from datetime import datetime

def parse_results_file():
    """
    Parse the most recent results file
    """
    # Find the most recent results file
    results_files = glob.glob("experiment_results_*.txt")
    if not results_files:
        print("❌ No results file found! Run experiments first.")
        return None
    
    # Get the most recent file
    latest_file = max(results_files, key=os.path.getctime)
    
    print(f"📁 Reading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        content = f.read()
    
    return content

def summarize_learning_curves():
    """
    Summarize available learning curve plots
    """
    plot_files = glob.glob("learning_curves_*.png")
    
    if not plot_files:
        print("❌ No learning curve plots found!")
        return
    
    print(f"\n📊 AVAILABLE LEARNING CURVE PLOTS ({len(plot_files)})")
    print("=" * 60)
    
    lr_experiments = []
    reg_experiments = []
    
    for filename in plot_files:
        # Extract hyperparameters from filename
        pattern = r'learning_curves_(\w+)_lr([\d.e-]+)_reg([\d.e-]+)(?:_h(\d+))?\.png'
        match = re.match(pattern, os.path.basename(filename))
        
        if match:
            model_type = match.group(1) 
            lr = float(match.group(2))
            reg = float(match.group(3))
            hidden_size = int(match.group(4)) if match.group(4) else 128
            
            info = {
                'filename': filename,
                'lr': lr,
                'reg': reg,
                'hidden_size': hidden_size
            }
            
            # Categorize experiments
            if reg == 0.001:  # Learning rate experiments
                lr_experiments.append(info)
            elif lr == 0.1:   # Regularization experiments
                reg_experiments.append(info)
    
    # Display learning rate experiments
    if lr_experiments:
        print("\n🎯 LEARNING RATE EXPERIMENTS (reg = 0.001)")
        print("-" * 50)
        lr_experiments.sort(key=lambda x: x['lr'])
        for exp in lr_experiments:
            print(f"   LR {exp['lr']:<8} → {exp['filename']}")
    
    # Display regularization experiments
    if reg_experiments:
        print("\n🎯 REGULARIZATION EXPERIMENTS (lr = 0.1)")
        print("-" * 50)
        reg_experiments.sort(key=lambda x: x['reg'])
        for exp in reg_experiments:
            print(f"   Reg {exp['reg']:<8} → {exp['filename']}")

def main():
    """
    Main summary function
    """
    print("📋 EXPERIMENT RESULTS SUMMARY")
    print("=" * 50)
    print(f"⏰ Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse and display results from text file
    results_content = parse_results_file()
    if results_content:
        print("\n" + results_content)
    
    # Summarize learning curve plots
    summarize_learning_curves()
    
    print(f"\n💡 ANALYSIS TIPS")
    print("-" * 30)
    print("1. 📈 Look for smooth, decreasing loss curves")
    print("2. 🎯 Find configs where train/val accuracy are close")
    print("3. 🏆 Choose the highest validation accuracy")
    print("4. 🚨 Avoid configs with wild loss fluctuations")
    
    print(f"\n🔍 NEXT STEPS")
    print("-" * 20)
    print("1. Open the best learning curve plots")
    print("2. Note the hyperparameters from the best configuration")
    print("3. Update configs/config_twolayer.yaml with optimal values:")
    print("   ```yaml")
    print("   Train:")
    print("     learning_rate: [best_lr_value]")
    print("     reg: [best_reg_value]")
    print("   ```")
    print("4. Run final training with optimal hyperparameters")
    
    print(f"\n🎉 Ready to use the best hyperparameters for your final model!")

if __name__ == "__main__":
    main()