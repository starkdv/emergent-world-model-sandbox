@echo off
REM Quick test script for agent survival improvements
REM Runs simulation with easy training config and learning enabled

echo ========================================
echo AGENT SURVIVAL TEST - EASY MODE
echo ========================================
echo.
echo Configuration: training_easy.yaml
echo Learning: ENABLED
echo.
echo Expected improvements:
echo   - Agents survive 300-500+ ticks (was ~187)
echo   - Agents learn to find food
echo   - Better survival rates
echo.
echo Press Ctrl+C to stop simulation
echo ========================================
echo.

python main.py --gui --learning --config config/training_easy.yaml

echo.
echo ========================================
echo Test completed!
echo ========================================
