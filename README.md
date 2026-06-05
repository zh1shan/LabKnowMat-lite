# LabKnowMat-lite

基于 VLM 智能体和原子化标注工具的图表智能重构系统。

## 快速开始

### 1. 创建 Conda 环境

```bash
conda create -n labknowmat-lite python=3.13
conda activate labknowmat-lite
```

### 2. 安装核心依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```env
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1/chat/completions
LLM_MODEL=google/gemini-3.1-pro-preview-customtools
LLM_MODEL_PHASE_3=google/gemini-3.1-pro-preview
```

> 获取 API Key: 注册 [OpenRouter](https://openrouter.ai/) 并创建 API Key。

---

## 可选依赖安装

以下工具为懒加载，仅在需要时调用。根据图表类型按需安装。

### PaddleOCR（文本识别）

用于提取图表中的文本和坐标，**处理大多数图表都需要**。

```bash
# 安装 PaddlePaddle（CPU 版）
pip install paddlepaddle

# 安装 PaddleOCR
pip install paddleocr
```

> 如需 GPU 加速，请安装对应的 CUDA 版本：https://paddlepaddle.org.cn/

### SAM3（图像分割）

用于提取柱状图柱体、饼图扇形等复杂图形，**仅在使用相关工具时需要**。

#### 3.1 安装 PyTorch

> **注意**：PyTorch 2.7.0 已弃用 CUDA 12.1，请使用以下受支持的 CUDA 版本。

```bash
# CUDA 12.8（推荐，需要较新的 NVIDIA 驱动）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.8（适用于较老的 GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CPU 版本（无需 GPU）
pip install torch torchvision torchaudio
```

> 请根据你的 GPU 驱动和 CUDA 版本选择合适的命令。可通过 `nvidia-smi` 查看驱动支持的最高 CUDA 版本。

#### 3.2 授权 Hugging Face

SAM3 模型权重托管在 Hugging Face，需要先登录才能自动下载。

```bash
# 安装 huggingface-cli
pip install huggingface-hub

# 登录（访问 https://huggingface.co/settings/tokens 获取 token）
huggingface-cli login
```

#### 3.3 克隆并安装 SAM3

```bash
# 在项目根目录下执行
git clone https://github.com/facebookresearch/sam3.git
cd sam3
git checkout 11dec2936de97f2857c1f76b66d982d5a001155d
pip install -e .
cd ..
```

> 首次运行 SAM3 相关工具时会自动下载模型权重到 `~/.cache/huggingface/`。

---

## 运行

### 处理单张图表

```bash
conda activate labknowmat-lite
python main.py -i <图片路径> -o <输出目录>
```

**示例：**

```bash
python main.py -i ./test_images/chart.png -o ./output/chart_out
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `-i, --img` | 输入图表图片路径 |
| `-o, --out` | 输出文件夹路径 |
| `-v, --visualize` | 可选，启用工具调用可视化（保存每次工具调用的标注图到 `tool_call_history/`） |

**输出文件：**

- `info.txt` - 语义结构和详细标注信息
- `render_chart.py` - 生成的图表重建脚本
- `chart.png` - 重建后的图表
- `data.csv` - 导出的数据表格
- `log.txt` - 执行日志

### 批量处理

```bash
python batch.py -i <图片文件夹>
```

**示例：**

```bash
python batch.py -i ./test_images/ -v
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `-i, --input` | 包含图表图片的文件夹路径 |
| `-v, --visualize` | 可选，为每张图表启用工具调用可视化 |

> 批量处理会自动遍历文件夹中的所有图片（支持 `.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`），并为每张图表创建 `<图片名>_out` 输出文件夹。

---

## 项目结构

```
LabKnowMat-lite/
├── main.py                 # 单图表处理入口
├── batch.py                # 批量处理脚本
├── requirements.txt        # 核心依赖
├── .env.example            # 环境变量模板
├── agent_framework/        # 核心框架
│   ├── agent.py            # ReAct 智能体主循环
│   ├── llm.py              # LLM API 封装
│   ├── planner.py          # Phase 1: 语义解析
│   ├── generator.py        # Phase 3: 代码生成
│   ├── visualizer.py       # 工具调用可视化
│   └── tools/              # 原子化工具库
├── prompt/                 # LLM 提示词模板
├── guidelines/             # VLM 分析指南
├── sam3/                   # SAM3 源码（可选，需手动安装）
└── docs/                   # 文档
```

详细架构说明请参阅 `docs/architecture.md`，开发指南请参阅 `docs/developer_guide.md`。
