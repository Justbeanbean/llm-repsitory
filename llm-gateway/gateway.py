"""薄入口：uvicorn 启动（开发用，带热重载）。

生产请使用 Dockerfile / docker-compose（无 reload，非 root 用户运行）。
在项目根目录执行：python gateway.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("llm_gateway.main:app", host="127.0.0.1", port=8000, reload=True)
