@echo off
chcp 936 >nul
title 文物照明保护-展示双目标优化工具
cd /d "C:\Users\DELL\deepseek_test\heritage_lighting_optimizer"
echo ============================================================
echo  正在启动应用，请稍候，浏览器将自动打开...
echo  如果浏览器没有自动打开，请手动访问:  http://localhost:8501
echo  关闭本窗口 = 关闭应用。请勿在应用使用期间关闭本窗口。
echo ============================================================
start "" cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:8501"
"C:\Users\DELL\deepseek_test\.venv\Scripts\python.exe" -m streamlit run app.py --server.headless=true --server.port 8501 --browser.gatherUsageStats=false
pause
