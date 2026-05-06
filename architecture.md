# LabKnowMat-lite: 基于智能体与原子化标注工具的图表重构系统

## 1. 项目背景与可行性分析 (Feasibility Analysis)

### 1.1 背景
当前的 `LabKnowMat` 系统采用了一种“单次触发、全程规则”的图表识别范式（即：VLM 仅做图表分类，然后路由到特定的图表识别工具中，该工具利用长篇大论的硬编码规则完成从坐标轴检测到图例匹配再到数据提取的全流程）。这种方式存在代码耦合度高、规则脆弱（OOD能力差）、维护困难的问题。

受 `paper_annotation` 人工标注工作流的启发，人工标注是通过自然语言与像素坐标结合（如 `annotate.txt`）描述图表元素，再交由 LLM 生成 Python 脚本重建图表。`LabKnowMat-lite` 旨在**用 VLM 智能体模拟人类的标注过程**。

### 1.2 可行性分析
**完全可行且具有高度潜力。** 
1. **现有工具资产丰富**：`LabKnowMat` 及 `reference` 中已积累了大量基础图像处理能力（如 `scatter_tool.py` 的颜色聚类点检测、`heatmap_tool.py` 的网格检测、SAM3 的零样本分割、PaddleOCR 的文本检测等）。这些工具的底层逻辑可以被复用。
2. **大模型能力跃升**：现代 VLM（如 GPT-4o, Claude 3.5 Sonnet）具备极强的空间语义理解能力，完全可以胜任“理解图表结构”的任务。
3. **解耦带来高容错**：将大一统的硬规则工具拆分为“原子化标注工具”，把流程控制权交给 VLM，可以极大提高系统的泛化能力（如应对混合图表、双轴图表等）。

---

## 2. 系统整体架构 (System Architecture)

`LabKnowMat-lite` 的核心架构由三大模块组成：**语义规划智能体 (Semantic Planner)**、**原子化工具库 (Atomic Tool Library)** 和 **代码重构生成器 (Code Generator)**。

```text
┌─────────────────────────────────────────────────────────────┐
│                      用户输入 (User Input)                  │
│                     (包含目标图表的图像文件)                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 1. 语义规划智能体 (VLM Planner)             │
│ - 识别图表类型及组成元素（如：包含2个系列柱状图和1条折线）  │
│ - 生成结构化标注大纲（类似于空的 annotate.txt）             │
│ - 制定工具调用策略（决定先调用轴定位，再调用颜色拾取等）    │
└──────────────────────────────┬──────────────────────────────┘
                               │ 循环调用 (Iterative Call)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 2. 原子化工具库 (Atomic Tool Library)       │
│ ┌────────────────┐ ┌────────────────┐ ┌───────────────────┐ │
│ │  文本检测工具  │ │  SAM3 分割工具 │ │  颜色聚类与定位   │ │
│ │ (OCR Detector) │ │ (Mask Extractor) │ │ (Color Clusterer)   │ │
│ └────────────────┘ └────────────────┘ └───────────────────┘ │
│ ┌────────────────┐ ┌────────────────┐ ┌───────────────────┐ │
│ │  坐标轴解析器  │ │ 像素<->数值映射│ │ 图元中心点提取器  │ │
│ │ (Axis Locator) │ │ (Coord Mapper) │ │ (Point Extractor) │ │
│ └────────────────┘ └────────────────┘ └───────────────────┘ │
└──────────────────────────────┬──────────────────────────────┘
                               │ 返回具体坐标、颜色、数值
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 3. 标注汇总与代码生成 (LLM Generator)       │
│ - 将收集到的元信息填充至 annotate.txt 格式中                │
│ - 将 annotate.txt 发送给 LLM                                │
│ - 生成 render_chart.py 用于图表重建或直接输出 data.csv      │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 工作流详述 (Workflow)

### 阶段 1: 语义与结构解析 (Semantic Parsing)
VLM 接收原始图表图像，输出高维语义结构。
- **输出示例**：
  ```json
  {
    "chart_type": "mixed",
    "components": [
      {"type": "x_axis", "label": "Fault types"},
      {"type": "y_axis_left", "label": "Precision & Recall (%)"},
      {"type": "bar_series", "name": "Precision", "color": "purple_like"},
      {"type": "line_series", "name": "F1-score", "color": "dark_blue"}
    ]
  }
  ```

### 阶段 2: 迭代式原子信息提取 (Iterative Annotation)
VLM 作为一个 Agent，通过 ReAct (Reasoning and Acting) 框架，不断调用小型工具来填补属性。
- **动作 A**：调用 `ocr_detector` 获取 X 轴所有刻度及其像素坐标。
- **动作 B**：调用 `color_clusterer` 提供 "purple_like" 提示，返回所有紫色柱体的 Bounding Box (左上角、右下角坐标)。
- **动作 C**：调用 `sam3_segmenter` 配合点提示，提取不规则扇形或区域。
- **循环结束条件**：所有 `components` 的像素坐标、颜色 HEX 值均被准确锚定。

### 阶段 3: 标注合成与代码重建 (Synthesis & Reconstruction)
将 Agent 在阶段 2 收集到的所有信息汇总为类似 `paper_annotation/annotate.txt` 的自然语言与坐标混合文件：
```text
x轴为类别轴，从左到右依次为"Fault 1"到"Fault 8"，像素坐标分别为...
左侧y轴为数值轴，名称为"Precision & Recall (%)"...
下面是柱体"Precision"的左上角和右下角坐标...
```
将该文件作为 Prompt 输入给专注于代码生成的 LLM（如 Claude-3.5-Sonnet 或 DeepSeek-Coder），输出 `render_chart.py` 以及结构化数据表。

---

## 4. 原子化工具库设计规范 (Tool Library Development Docs)

为使 Agent 自由调用，必须将原有 `LabKnowMat` 中庞大、耦合的 Extractor 拆解为单一职责的原子工具。

### 4.1 核心原子工具列表

| 工具名称 | 功能描述 | 输入 | 输出 | 来源借鉴 |
|---|---|---|---|---|
| `ocr_text_locator` | 获取图中所有文本及其 bbox 中心坐标 | `image` | `[{"text": "50", "cx": 93, "cy": 523}, ...]` | `tools/common/axis/ocr_detector.py` |
| `axis_tick_mapper` | 根据给定的文本区，拟合像素与数值的线性/对数关系 | `[{"px": 523, "val": 50}, ...]` | `slope, intercept, scale_type` | `tools/families/scatter_tool.py` 里的 `_build_coord_maps` |
| `color_point_extractor` | 提取指定颜色或区域内的散点/折线拐点像素坐标 | `image`, `target_color` (可选) | `[{"cx": 167, "cy": 203, "color": "#201955"}, ...]` | `reference/scatter/scatter_tool.py` 里的 `_detect_points` |
| `sam3_box_extractor` | 利用 SAM3 与文本 Prompt(如 "bar") 提取矩形条的 bbox | `image`, `text_prompt="bar"` | `[[x_min, y_min, x_max, y_max], ...]` | `reference/SAM3/.../sam3_bar_extractor.py` |
| `grid_color_sampler` | (热力图专用) 给定网格行列数和区域，对每个单元格中心点采样颜色 | `image`, `bbox`, `n_rows`, `n_cols` | `[{"row": 0, "col": 1, "color": "#ff0000"}, ...]`| `reference/heatmap/heatmap_tool.py` 的池化逻辑 |

### 4.2 工具 API 接口规范
所有的原子工具应遵循极简的输入输出，避免内部做复杂的业务逻辑假设。
```python
class BaseAtomicTool:
    name: str
    description: str

    def run(self, image: np.ndarray, **kwargs) -> dict:
        """返回扁平化的 JSON 兼容字典"""
        pass
```

---

## 5. 开发实施路线图 (Roadmap)

### Phase 1: 基础设施搭建 (Week 1)
- 初始化 `LabKnowMat-lite` 项目目录。
- 引入 LangGraph 或直接使用轻量级的 ReAct 框架作为 Agent 运行时。
- 从 `LabKnowMat` 中移植并封装前 3 个核心基础工具（OCR、SAM3、Color Cluster）。

### Phase 2: Agent Prompt 与 Workflow 调试 (Week 2)
- 编写 VLM Planner 的 System Prompt，使其能够准确输出图表结构。
- 构建单例图表（如单一系列柱状图）的端到端跑通（VLM -> Tools -> annotate.txt -> LLM -> Python）。

### Phase 3: 复杂图表支持与原子工具扩充 (Week 3-4)
- 增加对折线图、散点图、热力图的工具支持（`point_extractor`, `grid_sampler`）。
- 引入基于 `paper_annotation` 验证集的效果评估机制。
- 调整 LLM 提示词工程，优化 Python 重建代码的成功率与准确率。

## 6. 优势总结
- **高自由度**：可以轻松处理混合图表（比如同时包含折线和柱状图），原版 `LabKnowMat` 对此无能为力。
- **可解释性强**：Agent 收集的中间产物 `annotate.txt` 对人类完全可读，可以很方便地进行人工干预和错误排查。
- **易于扩展**：新增一种图表类型（如雷达图）不需要编写一整个复杂的流水线，只需补充一个 `radar_polygon_extractor` 工具，由大模型自行规划调用。