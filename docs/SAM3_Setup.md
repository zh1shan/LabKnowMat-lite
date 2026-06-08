# SAM3 环境配置指南 (SAM3 Environment Setup Guide)

`LabKnowMat-lite` 使用 [SAM3](https://github.com/facebookresearch/sam3) (Segment Anything Model 3) 来提取图表中复杂的图形区域，如柱状图的柱体和饼图的扇形轮廓。

由于 SAM3 模型以及其依赖相对庞大，并且需要处理环境冲突，本项目在架构上遵循**“代码内置，环境外挂”**的原则：
1. 我们已经在 `LabKnowMat-lite/agent_framework/tools/extractor.py` 中完美封装好了调用接口，并从 `reference/SAM3` 迁移了所有无需外部依赖的后处理代码（见 `pie_utils.py`）。
2. `SAM3` 的模型推理库则作为第三方依赖项，需要按照本指南在运行环境中进行配置。

## 安装步骤

请确保你已经激活了 `labknowmat-lite` conda 环境：
```bash
conda activate labknowmat-lite
```

### 1. 安装 PyTorch 及其生态
推荐安装支持 CUDA 的版本（根据您的显卡情况调整）。如果仅使用 CPU，可以安装 cpu 版本。
```bash
# 示例：安装 PyTorch (CUDA 12.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 或 CUDA 11.8（较老 GPU）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CPU 版本
pip install torch torchvision torchaudio
```

### 2. 安装 SAM3 及其他依赖
你需要将 SAM3 安装到环境中。在 `LabKnowMat-lite` 项目根目录下运行：

```bash
# 下载 SAM3 源码
git clone https://github.com/facebookresearch/sam3.git

# 切换到指定分支，保证项目兼容性正常运行
cd sam3
git checkout 11dec2936de97f2857c1f76b66d982d5a001155d

# 安装 SAM3
pip install -e .

# 退回项目根目录
cd ..
```

*注意：`sam3` 的安装脚本可能会自动拉取一些额外的依赖如 `timm`, `huggingface-hub`, `pillow` 等。*

### 3. 模型权重说明
你不需要手动去下载几个 GB 的权重文件。
当你第一次运行测试脚本或触发 Agent 调用 `SAM3BoxExtractor` / `PieSliceExtractor` 时，Hugging Face `transformers` 会自动从远端拉取 `sam3.pt` 并在本机的默认缓存目录（通常是 `~/.cache/huggingface`）中进行缓存。

## 测试配置是否成功

配置完成后，你可以运行预设的测试脚本来验证 SAM3 是否能正常工作（它将自动下载权重并生成带有边界框和多边形轮廓的可视化图像）：

```bash
conda run -n labknowmat-lite python LabKnowMat-lite/test_script/test_sam3.py
```