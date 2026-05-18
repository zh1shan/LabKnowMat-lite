# SAM3 环境配置指南

`LabKnowMat-lite` 使用 SAM3（Segment Anything Model 3）来进行基于文本提示（Text Prompt）的零样本图像分割，主要用于提取：
1. **柱状图柱体** (`SAM3BoxExtractor`)
2. **饼图扇形** (`PieSliceExtractor`)

由于 SAM3 依赖较多且对 PyTorch 版本有一定要求，请按照以下步骤在您的环境中配置 SAM3。

## 配置步骤

### 1. 激活虚拟环境
确保您已经激活了 `LabKnowMat-lite` 的 Conda 环境：
```bash
conda activate labknowmat-lite
```

### 2. 安装 PyTorch
SAM3 官方要求 Python 3.12+ 和 PyTorch 2.7+。但根据经验，如果在 Python 3.10 下使用较新的 PyTorch 版本（如 2.3+）通常也能运行。请根据您的显卡 CUDA 版本安装对应的 PyTorch：
```bash
# 例如，如果您的 CUDA 版本为 12.1（请根据实际情况调整）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 安装 SAM3 及依赖
本地的 `reference/SAM3/sam3` 目录中已经包含了 SAM3 的源码，您可以直接通过 pip 以可编辑模式（editable mode）安装：
```bash
# 在 LabKnowMat-lite 项目根目录下运行
pip install -e reference/SAM3/sam3
```

如果您在安装过程中缺少某些基础库，可以直接参考 `reference/SAM3/environment.yml` 中的 `pip` 依赖列表进行安装（例如 `timm`, `einops`, `pycocotools`, `decord` 等）。

### 4. （可选）下载预训练权重
SAM3 模型在第一次运行时会自动从 HuggingFace 下载预训练权重（如 `sam3_base.pt` 或类似模型）。
如果由于网络问题下载失败，您可能需要：
1. 配置 HuggingFace 的镜像站或代理（如设置环境变量 `HF_ENDPOINT=https://hf-mirror.com`）。
2. 或前往 HuggingFace 申请 SAM3 权重访问权限，使用 `huggingface-cli login` 登录后再运行。

## 验证安装

配置完成后，您可以运行我们提供的测试脚本来验证提取工具是否正常工作：
```bash
python LabKnowMat-lite/test_script/test_sam3_extractors.py
```
