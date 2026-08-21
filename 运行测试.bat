@echo off
chcp 936 >nul
title 运行自动测试（T1-T13 验收）
cd /d "C:\Users\DELL\deepseek_test\heritage_lighting_optimizer"
echo ============================================================
echo  正在运行自动测试，约需 1 分钟...
echo  看到 "28 passed" 即表示全部验收项通过。
echo ============================================================
"C:\Users\DELL\deepseek_test\.venv\Scripts\python.exe" -m pytest tests/ -v
pause
