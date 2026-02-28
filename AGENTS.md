# AGENTS.md

本项目使用 `uv` 作为 Python 包管理器和运行环境。

## 环境配置

### 设置环境变量

在运行任何命令之前，必须设置以下环境变量：

```bash
export UV_PROJECT_ENVIRONMENT=/home/featurize/venv
```

建议将此行添加到 `~/.bashrc` 或 `~/.zshrc` 中以确保持久生效：

```bash
echo 'export UV_PROJECT_ENVIRONMENT=/home/featurize/venv' >> ~/.bashrc
source ~/.bashrc
```

## 使用 uv 运行脚本

### 运行 Python 脚本

使用 `uv run` 来运行 Python 脚本，这会自动使用项目配置的虚拟环境：

```bash
# 运行训练脚本
uv run train_sf-yolo.py

# 运行验证脚本
uv run val.py

# 运行其他脚本
uv run benchmarks.py
```

### 管理依赖

所有依赖都在 `pyproject.toml` 中定义，使用以下命令管理：

```bash
# 安装项目依赖（根据 pyproject.toml 和 uv.lock）
uv sync

# 添加新依赖
uv add <package-name>

# 添加开发依赖
uv add --dev <package-name>

# 更新依赖
uv sync --upgrade

# 更新特定包
uv add --upgrade <package-name>
```

### 进入项目环境

如果需要进入交互式 Python shell 或手动执行命令：

```bash
# 进入项目的 Python 环境
uv run python

# 或者使用 uv run 直接执行命令
uv run python -c "import torch; print(torch.__version__)"
```

## 重要提示

- **不要**直接使用系统 Python 或 pip 运行脚本或安装包
- **不要**手动激活虚拟环境，使用 `uv run` 代替
- 始终确保 `UV_PROJECT_ENVIRONMENT` 环境变量已设置
- 项目使用清华镜像源加速依赖下载，并在需要时从 PyTorch 官方源安装 CUDA 版本的 torch/torchvision

## 项目依赖说明

主要依赖包括：
- PyTorch (CUDA 13.0 版本)
- ultralytics
- 其他数据处理和可视化库

详见 `pyproject.toml` 获取完整依赖列表。
