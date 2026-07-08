# 视觉伺服云台控制系统

## 项目简介
基于 Python 和 OpenCV 的云台视觉伺服系统，后期移植到瑞萨开发板。

## 环境要求
- Ubuntu 22.04+
- Python 3.10+
- uv（Python 包管理器）

## 快速开始

### 1. 安装 uvp
```bash
pipx install uv
uv venv
uv add opencv-python numpy pyserial
uv run python src/entrypoints/main.py
├── data/          # 数据文件（标定、参数、图片）
├── scripts/       # 工具脚本
├── src/           # 核心代码
│   ├── entrypoints/   # 主程序入口
│   ├── vision/        # 视觉模块
│   ├── control/       # 控制模块
│   └── utils/         # 工具模块
├── docs/          # 文档
└── MY_NOTES.md    # 个人踩坑记录

---

### 创建 `.vscode/settings.json`（VSCode 自动识别虚拟环境）

```bash
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": false,
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
