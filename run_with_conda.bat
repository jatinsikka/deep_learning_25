@echo off
echo 🐍 Conda Environment Setup and Experiment Runner
echo =================================================

echo 📋 Step 1: Checking conda installation...
conda --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Conda not found! Please install Anaconda/Miniconda first.
    pause
    exit /b 1
) else (
    echo ✅ Conda found!
)

echo.
echo 📋 Step 2: Activating conda environment...
echo ⚠️  If you have a specific environment, modify this script to use it.
echo 🔧 Currently using 'base' environment...
call conda activate base

echo.
echo 📋 Step 3: Installing required packages...
echo 🔧 Installing PyYAML...
pip install PyYAML

echo 🔧 Installing other dependencies if needed...
pip install numpy matplotlib

echo.
echo 📋 Step 4: Running hyperparameter experiments...
echo 🚀 Starting experiments...
python run_experiments.py

echo.
echo 🎉 All done! Check the results above and the generated plot files.
pause