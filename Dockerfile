# 文物照明保护—展示双目标优化工具 · Docker 镜像
# 构建:  docker build -t heritage-lighting-optimizer .
# 运行:  docker run -d -p 8501:8501 --name heritage-opt heritage-lighting-optimizer
# 访问:  http://<服务器公网IP>:8501
FROM python:3.11-slim

WORKDIR /app

# 中文字体（PNG 图表渲染中文需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
