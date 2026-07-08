# uv使用与环境配置
## 管理依赖
- 添加：`uv add <包名>`
- 移除：`uv remove <包名>`
- 查看：`uv tree`

## VSCode 配置
- 选择 Python 解释器为 `.venv/bin/python`
- 不需要手动 `source .venv/bin/activate`
---

# 🔧 uv 安装与 PATH 配置（2026-07-06）

## 问题现象
执行 `uv --version` 报错：
找不到命令 “uv”，但可以通过以下软件包安装它：
sudo snap install astral-uv
但明明之前已经用 `pipx install uv` 成功安装了。

## 原因分析
`pipx` 默认把可执行文件安装到 `~/.local/bin/`，但该目录**不在系统的 PATH 环境变量中**，所以终端找不到 `uv` 命令。

## 验证 uv 是否真的安装了
```bash
find ~/.local -name "uv" 2>/dev/null
如果输出包含 /home/yln/.local/bin/uv，说明 uv 已安装，只是 PATH 没配置好。
# 方法一：让 pipx 自动配置 PATH（推荐）
pipx ensurepath
source ~/.bashrc

# 方法二：手动添加 PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 验证
uv --version   # 应输出 uv 0.11.26
which uv       # 应输出 /home/yln/.local/bin/uv

pipx ensurepath 的作用：
自动将 ~/.local/bin 添加到 PATH，并写入 ~/.bashrc，永久生效。
