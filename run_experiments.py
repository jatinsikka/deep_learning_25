#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for Two-Layer Neural Network
This script systematically tests different learning rates and regularization values
"""

import os
import sys
import yaml
import subprocess
import time
from datetime import datetime

def create_temp_config(base_config_path, learning_rate=None, reg=None, output_path="temp_config.yaml"):
    """
    Create a temporary config file with modified hyperparameters
    """
    # Load base configuration
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Modify hyperparameters if provided
    if learning_rate is not None:
        config['Train']['learning_rate'] = learning_rate
    if reg is not None:
        config['Train']['reg'] = reg
    
    # Save temporary config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config

def run_experiment(config_path, experiment_name):
    """
    Run a single experiment with the given config and extract accuracies
    """
    print(f"\n{'='*60}")
    print(f"🚀 Starting Experiment: {experiment_name}")
    print(f"📁 Config: {config_path}")
    print(f"⏰ Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    
    try:
        # Run the main.py script
        result = subprocess.run([
            sys.executable, "main.py", 
            "--config", config_path
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print(f"✅ {experiment_name} completed successfully!")
            
            # Extract accuracies from output
            output = result.stdout
            final_train_acc = None
            final_val_acc = None
            final_test_acc = None
            
            # Parse the output for accuracy information
            lines = output.split('\n')
            for line in lines:
                if "Average Accuracy of Epoch" in line:
                    # Get the last training accuracy
                    try:
                        final_train_acc = float(line.split()[-1])
                    except:
                        pass
                elif "Validation Accuracy:" in line:
                    # Get the last validation accuracy
                    try:
                        final_val_acc = float(line.split()[-1])
                    except:
                        pass
                elif "Final Accuracy on Test Data:" in line:
                    # Get the final test accuracy
                    try:
                        final_test_acc = float(line.split()[-1])
                    except:
                        pass
            
            return {
                'success': True,
                'train_accuracy': final_train_acc,
                'validation_accuracy': final_val_acc,
                'test_accuracy': final_test_acc,
                'output': output
            }
        else:
            print(f"❌ {experiment_name} failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return {'success': False, 'error': result.stderr}
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {experiment_name} timed out after 10 minutes!")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"💥 {experiment_name} crashed with error: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """
    Main experiment runner
    """
    print("🧪 HYPERPARAMETER TUNING EXPERIMENTS")
    print("="*50)
    
    base_config = "configs/config_twolayer.yaml"
    temp_config = "temp_experiment_config.yaml"
    
    # Check if base config exists
    if not os.path.exists(base_config):
        print(f"❌ Base config file not found: {base_config}")
        return
    
    # Create results directory
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    experiment_results = []
    successful_experiments = []
    failed_experiments = []
    
    # ==========================================================================
    # EXPERIMENT 1: Learning Rate Tuning (with default reg=0.001)
    # ==========================================================================
    print("\n🎯 PHASE 1: Learning Rate Experiments")
    print("🔧 Regularization fixed at 0.001")
    
    learning_rates = [1.0, 0.1, 0.01, 0.05]  # 1, 1e-1, 1e-2, 5e-2
    default_reg = 0.001
    
    for lr in learning_rates:
        experiment_name = f"LR_Experiment_lr{lr}_reg{default_reg}"
        
        # Create temporary config
        config = create_temp_config(base_config, learning_rate=lr, reg=default_reg, output_path=temp_config)
        
        print(f"\n📊 Testing Learning Rate: {lr}")
        print(f"   Regularization: {default_reg}")
        print(f"   Hidden Size: {config['Model'].get('hidden_size', 'N/A')}")
        print(f"   Batch Size: {config['Train']['batch_size']}")
        print(f"   Epochs: {config['Train']['epochs']}")
        
        # Run experiment
        result = run_experiment(temp_config, experiment_name)
        
        # Store result with hyperparameters
        experiment_info = {
            'name': experiment_name,
            'learning_rate': lr,
            'regularization': default_reg,
            'hidden_size': config['Model'].get('hidden_size', 128),
            'experiment_type': 'Learning Rate',
            'result': result
        }
        experiment_results.append(experiment_info)
        
        if result['success']:
            successful_experiments.append(experiment_name)
            train_acc = result.get('train_accuracy', 0)
            val_acc = result.get('validation_accuracy', 0)
            test_acc = result.get('test_accuracy', 0)
            print(f"📊 Results - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        else:
            failed_experiments.append(experiment_name)
        
        # Small delay between experiments
        time.sleep(2)
    
    # ==========================================================================
    # EXPERIMENT 2: Regularization Tuning (with default lr=0.1)
    # ==========================================================================
    print("\n🎯 PHASE 2: Regularization Experiments")
    print("🔧 Learning Rate fixed at 0.1")
    
    regularizations = [0.001, 0.01, 0.1, 0.0001, 1.0]  # 1e-3, 1e-2, 1e-1, 1e-4, 1e-0
    default_lr = 0.1
    
    for reg in regularizations:
        experiment_name = f"REG_Experiment_lr{default_lr}_reg{reg}"
        
        # Create temporary config
        config = create_temp_config(base_config, learning_rate=default_lr, reg=reg, output_path=temp_config)
        
        print(f"\n📊 Testing Regularization: {reg}")
        print(f"   Learning Rate: {default_lr}")
        print(f"   Hidden Size: {config['Model'].get('hidden_size', 'N/A')}")
        print(f"   Batch Size: {config['Train']['batch_size']}")
        print(f"   Epochs: {config['Train']['epochs']}")
        
        # Run experiment
        result = run_experiment(temp_config, experiment_name)
        
        # Store result with hyperparameters
        experiment_info = {
            'name': experiment_name,
            'learning_rate': default_lr,
            'regularization': reg,
            'hidden_size': config['Model'].get('hidden_size', 128),
            'experiment_type': 'Regularization',
            'result': result
        }
        experiment_results.append(experiment_info)
        
        if result['success']:
            successful_experiments.append(experiment_name)
            train_acc = result.get('train_accuracy', 0)
            val_acc = result.get('validation_accuracy', 0)
            test_acc = result.get('test_accuracy', 0)
            print(f"📊 Results - Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
        else:
            failed_experiments.append(experiment_name)
        
        # Small delay between experiments
        time.sleep(2)
    
    # ==========================================================================
    # CLEANUP AND SUMMARY
    # ==========================================================================
    
    # Remove temporary config file
    if os.path.exists(temp_config):
        os.remove(temp_config)
    
    # Print detailed results summary
    print("\n" + "="*80)
    print("🎉 COMPREHENSIVE EXPERIMENT RESULTS")
    print("="*80)
    
    # Print results table
    print(f"\n📊 DETAILED RESULTS TABLE")
    print("-" * 80)
    print(f"{'Experiment':<20} {'LR':<8} {'Reg':<8} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10}")
    print("-" * 80)
    
    successful_results = []
    for exp_info in experiment_results:
        if exp_info['result']['success']:
            result = exp_info['result']
            train_acc = result.get('train_accuracy', 0) or 0
            val_acc = result.get('validation_accuracy', 0) or 0
            test_acc = result.get('test_accuracy', 0) or 0
            
            exp_type = exp_info['experiment_type'][:2]  # LR or RE
            lr = exp_info['learning_rate']
            reg = exp_info['regularization']
            
            print(f"{exp_type:<20} {lr:<8} {reg:<8} {train_acc:<10.4f} {val_acc:<10.4f} {test_acc:<10.4f}")
            
            successful_results.append({
                'type': exp_info['experiment_type'],
                'lr': lr,
                'reg': reg,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc
            })
    
    # Find best configurations
    if successful_results:
        best_val_result = max(successful_results, key=lambda x: x['val_acc'])
        best_test_result = max(successful_results, key=lambda x: x['test_acc'])
        
        print(f"\n🏆 BEST CONFIGURATIONS")
        print("-" * 50)
        print(f"🥇 Best Validation Accuracy: {best_val_result['val_acc']:.4f}")
        print(f"   • Learning Rate: {best_val_result['lr']}")
        print(f"   • Regularization: {best_val_result['reg']}")
        print(f"   • Train Acc: {best_val_result['train_acc']:.4f}")
        print(f"   • Test Acc: {best_val_result['test_acc']:.4f}")
        
        print(f"\n🎯 Best Test Accuracy: {best_test_result['test_acc']:.4f}")
        print(f"   • Learning Rate: {best_test_result['lr']}")
        print(f"   • Regularization: {best_test_result['reg']}")
        print(f"   • Train Acc: {best_test_result['train_acc']:.4f}")
        print(f"   • Val Acc: {best_test_result['val_acc']:.4f}")
    
    # Summary statistics
    print(f"\n✅ Successful Experiments ({len(successful_experiments)}):")
    for exp in successful_experiments:
        print(f"   • {exp}")
    
    if failed_experiments:
        print(f"\n❌ Failed Experiments ({len(failed_experiments)}):")
        for exp in failed_experiments:
            print(f"   • {exp}")
    
    total_experiments = len(successful_experiments) + len(failed_experiments)
    success_rate = (len(successful_experiments) / total_experiments * 100) if total_experiments > 0 else 0
    
    print(f"\n📈 Success Rate: {success_rate:.1f}% ({len(successful_experiments)}/{total_experiments})")
    
    # Save results to file
    results_file = f"experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(results_file, 'w') as f:
        f.write("HYPERPARAMETER TUNING RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"{'Experiment':<20} {'LR':<8} {'Reg':<8} {'Train Acc':<10} {'Val Acc':<10} {'Test Acc':<10}\n")
        f.write("-"*80 + "\n")
        
        for exp_info in experiment_results:
            if exp_info['result']['success']:
                result = exp_info['result']
                train_acc = result.get('train_accuracy', 0) or 0
                val_acc = result.get('validation_accuracy', 0) or 0
                test_acc = result.get('test_accuracy', 0) or 0
                
                exp_type = exp_info['experiment_type'][:2]
                lr = exp_info['learning_rate']
                reg = exp_info['regularization']
                
                f.write(f"{exp_type:<20} {lr:<8} {reg:<8} {train_acc:<10.4f} {val_acc:<10.4f} {test_acc:<10.4f}\n")
        
        if successful_results:
            f.write(f"\nBest Validation Accuracy: {best_val_result['val_acc']:.4f} (LR={best_val_result['lr']}, Reg={best_val_result['reg']})\n")
            f.write(f"Best Test Accuracy: {best_test_result['test_acc']:.4f} (LR={best_test_result['lr']}, Reg={best_test_result['reg']})\n")
    
    print(f"\n📁 Learning curve plots and results saved!")
    print(f"   • Plots: learning_curves_TwoLayerNet_lr<value>_reg<value>_h<hidden_size>.png")
    print(f"   • Results table: {results_file}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Check the generated learning curve plots")
    print(f"   2. Use the best hyperparameters shown above")
    print(f"   3. Update your config file with optimal values")
    
    print("\n🚀 All experiments completed with detailed results!")

if __name__ == "__main__":
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()