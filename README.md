# baobao-ai-painting-backend

## 项目介绍

本项目是一个基于 Django 的 AI 绘画后端服务，为绘画应用提供图像生成、任务队列和相关数据管理接口。项目通过 HTTP API 接收客户端请求，并调用 Stable Diffusion 系列模型及 Flux 的生成控制器处理任务。

### 主要功能

- **AI 图像生成**：提供 SDXL、Stable Diffusion 3 和 Flux 的文生图接口，并支持按队列提交任务、查询生成状态及取消任务。
- **图像处理**：支持图像上传、Base64 图像保存、作品入库，以及基于 SDXL 的局部重绘（Inpaint）。
- **绘画数据管理**：提供作品、模型、提示词、提示词分类、字典及日志等管理接口。
- **用户与认证**：提供用户信息管理及认证相关接口。
- **辅助能力**：包含队列状态查询和文本翻译接口。

### 技术栈

- Python、Django
- SQLAlchemy 数据模型与数据库连接
- Hugging Face Diffusers、Transformers、PyTorch 等模型推理相关组件

### 项目结构

```text
.
├── manage.py                    # Django 命令行入口
├── stableDiffusion/
│   ├── settings.py               # Django 项目配置
│   ├── urls*.py                  # API 路由
│   ├── controllers/              # 生成、用户、图像及数据管理等控制器
│   └── models/                   # 数据模型
├── requirements.txt              # Python 依赖
├── requirements_macos.txt        # macOS 环境依赖
└── requirements_linux.txt        # Linux 环境依赖
```

## 在 PyCharm 中启动

1. 在 PyCharm 中打开项目根目录。
2. 打开 **Run（运行） > Edit Configurations…（编辑配置）**，点击 **+** 并选择 **Python**。
3. 配置运行项：
   - **Name（名称）**：`Django runserver`
   - **Script path（脚本路径）**：项目根目录下的 `manage.py`
   - **Parameters（参数）**：`runserver`
   - **Working directory（工作目录）**：项目根目录
   - **Python interpreter（Python 解释器）**：项目虚拟环境中的解释器，例如 `venv/bin/python`（Windows 下为 `venv\Scripts\python.exe`）
4. 点击 **Apply（应用）** 和 **OK（确定）**，然后运行该配置。

也可以在 PyCharm 的 Terminal 中执行：

```bash
venv/bin/python manage.py runserver
```

Windows 下：

```powershell
venv\Scripts\python.exe manage.py runserver
```
