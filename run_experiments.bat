@echo off
echo 🧪 Hyperparameter Tuning Experiments
echo ================================

echo 🚀 Starting experiments...
python run_experiments.py

echo.
echo 📊 Checking generated plots...
dir learning_curves_*.png /b 2>nul
if errorlevel 1 (
    echo ❌ No learning curve plots found
) else (
    echo ✅ Learning curve plots generated successfully!
)

echo.
echo 📋 To view a clean summary of results, run:
echo python view_results.py

echo.
echo 🎉 Experiments complete! All training/validation/test accuracies displayed above.
pause