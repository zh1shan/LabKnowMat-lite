# LabKnowMat-lite

LabKnowMat-lite 是一个基于大语言模型的图表智能体重建系统，核心理念为 **"VLM 作为大脑，原子工具作为手脚"**。系统通过三阶段流水线将图表图像重建为 Python 代码与结构化数据：

1. **语义解析 (Phase 1)**：VLM 观察图表，输出 JSON 格式的语义结构树（图表类型、组件、颜色、空间布局等）。
2. **迭代式信息提取 (Phase 2)**：VLM 智能体在 ReAct 循环中自主调用原子化工具（OCR、线段检测、散点聚类、SAM3 分割、热力图数字化等），提取像素级坐标与颜色信息。
3. **代码重建 (Phase 3)**：LLM 将标注信息转化为 matplotlib Python 脚本并执行，输出重建图表与数据文件。

---

## 快速开始

### 1. 克隆 SAM3 源码

在项目根目录下克隆 SAM3 仓库并切换到指定提交：

```bash
git clone https://github.com/facebookresearch/sam3.git
cd sam3
git checkout 11dec2936de97f2857c1f76b66d982d5a001155d
cd ..
```

### 2. 配置 conda 环境

创建并激活环境（PyTorch 及所有 Python 依赖已通过 `requirements-locked.txt` 自动安装）：

```bash
conda env create -f environment.yml
conda activate labknowmat-lite
```

### 3. 安装 SAM3

```bash
cd sam3
pip install -e .
cd ..
```

### 4. 配置 API Key

复制 `.env.example` 为 `.env`，填入你的 OpenRouter API Key：

```bash
cp .env.example .env
# 编辑 .env，填入 OPENROUTER_API_KEY
```

`.env` 中的默认模型配置：

| 变量 | 用途 | 默认值 |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API 密钥 | （必填） |
| `OPENROUTER_BASE_URL` | API 端点 | `https://openrouter.ai/api/v1/chat/completions` |
| `LLM_MODEL` | Phase 1 & 2 模型 | `google/gemini-3.1-pro-preview-customtools` |
| `LLM_MODEL_PHASE_3` | Phase 3 模型 | `google/gemini-3.1-pro-preview` |

### 5. 配置 SAM3 模型访问权限

1. 前往 [facebook/sam3 - Hugging Face](https://huggingface.co/facebook/sam3) 申请模型访问权限，获批后复制你的 Access Token。
2. 在已激活的 conda 环境中登录 Hugging Face：

```bash
hf auth login
# 粘贴你的 Access Token
```

3. 首次调用 SAM3 工具时，模型权重会自动从 Hugging Face 下载并缓存，请耐心等待。

---

## 使用方法

### 单张图表处理

```bash
python main.py -i <图片路径> -o <输出目录> [-v]
```

| 参数 | 说明 |
|---|---|
| `-i, --img` | 输入图表图像路径（必填） |
| `-o, --out` | 输出目录路径（必填） |
| `-v, --visualize` | 启用工具调用可视化，保存每步调试图像到 `tool_call_history/` |

**示例：**

```bash
python main.py -i charts/bar_chart.png -o output/bar_chart_out -v
```

### 批量处理

对文件夹中的所有图表图像（png/jpg/jpeg/gif/webp）进行批量处理，每张图像的输出目录自动命名为 `<图片路径>_out`：

```bash
python batch.py -i <图片文件夹路径> [-v]
```

| 参数 | 说明 |
|---|---|
| `-i, --input` | 包含图表图像的文件夹路径（必填） |
| `-v, --visualize` | 启用工具调用可视化 |

**示例：**

```bash
python batch.py -i charts/ -v
```

---

## 输出说明

每个输出目录包含以下文件：

| 文件 | 说明 |
|---|---|
| `log.txt` | 完整执行日志 |
| `info.txt` | 语义结构与详细标注信息 |
| `render_chart.py` | LLM 生成的 matplotlib 重建脚本 |
| `chart.png` | 重建的图表图像 |
| `data.csv` | 提取的结构化表格数据 |
| `heatmap_data.json` | 热力图数值矩阵（仅热力图图表） |
| `tool_call_history/` | 工具调用可视化图像（仅 `-v` 模式） |

---

## 开源协议

本项目基于 [MIT License](LICENSE) 开源。
