@echo off
REM Script for full reproduction of research results (Windows)
REM Executes all steps from data generation to statistical analysis

echo ==========================================
echo   REPRODUCING RESEARCH RESULTS
echo ==========================================
echo.

cd /d "%~dp0\.."

REM Create necessary directories
echo [1/6] Creating directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "results" mkdir results
echo OK
echo.

REM Step 1: Generate data
echo ==========================================
echo STEP 1: Generating Dataset
echo ==========================================
echo WARNING: This will take 1-2 hours!
echo You can skip this if you already have data
echo.
set /p CONTINUE="Continue? (y/n): "
if /i "%CONTINUE%"=="y" (
    echo Generating 1,000 robot configurations (Nelder-Mead optimization)...
    python src\generate_data_optimized.py
    echo OK
) else (
    echo Skipped. Use existing dataset or run later:
    echo   python src\generate_data_optimized.py
)
echo.

REM Step 2: Prepare data
echo ==========================================
echo STEP 2: Preparing Training Data
echo ==========================================
echo Preparing data...
python src\prepare_training_data.py
echo OK
echo.

REM Step 3: Train model
echo ==========================================
echo STEP 3: Training Model
echo ==========================================
echo Training neural network...
python src\train_model.py
echo OK
echo.

REM Step 4: Check model
echo ==========================================
echo STEP 4: Checking Model
echo ==========================================
python src\check_model.py
echo.

REM Step 5: Experiments
echo ==========================================
echo STEP 5: Running Experiments
echo ==========================================
echo Running experiments (5-10 minutes)...
python src\experiments.py
echo OK
echo.

REM Step 6: Statistical analysis
echo ==========================================
echo STEP 6: Statistical Analysis
echo ==========================================
echo Statistical analysis...
python src\statistical_analysis_improved.py
echo OK
echo.

REM Summary
echo ==========================================
echo   REPRODUCTION COMPLETE
echo ==========================================
echo.
echo All results reproduced!
echo.
echo Results saved in:
echo    - results\improvement_distribution.png
echo    - results\noise_robustness.png
echo    - results\results_comparison.png
echo    - results\statistical_comparison_baseline.png
echo    - results\statistical_comparison_cc.png
echo    - results\statistical_comparison_chr.png
echo    - results\statistical_results_improved.json
echo.
echo To use the model:
echo    python src\predict_pid.py ^<mass^> ^<damping_coeff^> ^<inertia^>
echo.
pause
