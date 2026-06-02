# LabKnowMat-lite 开发者指南 (Developer Guide)

欢迎来到 LabKnowMat-lite 项目。本文档旨在为后续开发者提供关于本项目的核心架构、工具库接口标准、已有功能以及未来开发规范的详尽说明。

## 1. 核心架构与原理 (Architecture & Principles)

`LabKnowMat-lite` 的核心设计理念是 **"VLM 作为大脑，原子工具作为手脚"**。
传统的图表识别系统大多依赖脆弱且冗长的基于硬规则的提取管道，这导致系统的跨领域（OOD）能力极差。而在本项目中，我们将流程重构为以下三个阶段（详见 `architecture.md`）：

1.  **语义与结构解析 (Phase 1)**：大模型（如 Kimi-k2.6）观察图表，输出 JSON 格式的语义树（包含有几个图例、什么类型的坐标轴、几组数据等）。
2.  **迭代式原子信息提取 (Phase 2 - Agentic Workflow)**：通过 ReAct 循环工作流，智能体（Agent）根据自己对图表的认知，自主、灵活地调用小巧且纯粹的“原子化工具”来填充每个视觉元素的具体像素坐标或颜色信息。
3.  **信息合成与代码重建 (Phase 3)**：将带有准确物理数值和精确像素坐标的信息交由专注于生成的 LLM 输出最终图表重建代码（该功能预留）。

## 2. 工具库总览 (Available Atomic Tools)

所有的工具实现都位于 `agent_framework/tools/` 目录下，并且必须继承自基类 `BaseAtomicTool`。目前的可用工具如下：

### 2.1 文本识别工具 (`ocr_text_locator`)
- **功能**：基于 PaddleOCR 提取图表中的所有文本及其包围盒 (box)、中心点 (cx, cy)。处理了倾斜文本（如垂直的 Y 轴标签），并自动过滤了常见的字母混淆（如将 'O' 替换为 '0'）。
- **参数**：无特定参数。

### 2.2 坐标轴线段提取工具 (`axis_line_locator`)
- **功能**：通过 OpenCV 色度反转和多次宽容度迭代的形态学开闭运算（Morphology Ex），提取图中所有的**绝对水平/绝对垂直**的实线和虚线，返回线段的两个端点坐标和类型。
- **参数**：无特定参数。

### 2.3 散点提取与聚类工具 (`scatter_point_extractor_v1`)
- **功能**：结合了模板匹配、局部极大值 NMS 与自动构建 OCR 文本遮罩（防止将文字识别为散点），精确捕获图表中的离散点并根据颜色在 Lab 空间中进行聚类。
- **参数**：
  - `target_rect`: `[x_min, y_min, x_max, y_max]`（可选），若提供则只提取该矩形框内的点。
  - `match_threshold`: NCC 模板匹配阈值（默认 0.75）。
  - `color_cluster`: 颜色聚类距离阈值（默认 15.0）。

### 2.4 热力图数字化工具 (`heatmap_digitizer`)
- **功能**：基于 HSV 亮度与饱和度提取热力图数据区及图例色谱。通过图例色谱建立起 RGB 到 `0~1` 归一化数值的映射（KD 树加速），并输出固定为 `128x128` 尺寸的热力图数值矩阵。
- **参数**：
  - `sat_thresh`: 饱和度阈值（默认 55）。
  - `val_thresh`: 亮度阈值（默认 80）。
  - `min_area`: 最小可接受区域面积（默认 300）。

### 2.5 SAM3 图形提取工具组 (`SAM3BoxExtractor`, `PieSliceExtractor`)
- **功能**：包装了 Facebook 的 Segment Anything Model 3。可以通过文本提示（Text Prompt）来做 Zero-Shot 的遮罩提取，例如提取柱体（bar）或饼图扇形（pie slice）。其中 `PieSliceExtractor` 包含了复杂的颜色拉伸后处理与极坐标边界平滑算法（迁移自参考代码的 `pie_utils.py`）。
- **注意**：需要在运行环境中单独通过 pip 源码安装 SAM3 仓库（详见 `SAM3_Setup.md`）。

---

## 3. 智能体工作流执行逻辑 (Agent Execution Flow)

在 `agent_framework/agent.py` 中的 `LabKnowMatLiteAgent` 控制了整个执行过程。它借鉴了如 OpenCode 等先进智能体的设计：

1.  **工具声明 (Tool Declarations)**：每个继承了 `BaseAtomicTool` 的子类都会重写 `get_parameters_schema()`，返回符合标准的 JSON Schema。Agent 启动时会将所有已注册工具的“说明书”打包发送给 Kimi LLM，使其理解自己有什么能力。
2.  **LLM 自动调度 (Function Calling)**：大模型通过分析图表的视觉图像和上下文语义，按需生成需要调用的函数名称和参数（如请求提取某区域内的散点）。
3.  **框架拦截与注入 (Framework Interception)**：当大模型发出调用请求时，框架会在本地通过 Python 实例化运行该工具，**自动将 OpenCV 加载的 numpy 图像矩阵 `image` 注入到工具的首个参数中**（不需要、也不能让 LLM 去处理庞大的图像字节），将计算得到的轻量化 JSON 结果回传给 LLM。

---

## 4. 后续开发规范 (Development Guidelines)

后续的开发者在扩展功能或添加新图表支持时，请严格遵守以下规范：

### 4.1. 创建新的原子工具
1.  **单一职责原则**：不要将“检测文本”、“检测线段”和“聚类匹配”写在一个工具中。将它们拆分成独立的函数，供 LLM 通过多次 Tool Calls 串联。
2.  **继承与抽象**：必须继承 `agent_framework.tools.base.BaseAtomicTool`。
3.  **实现 Schema 接口**：必须实现 `get_parameters_schema(self)` 以告知 LLM 如何传递超参数（不要包含 `image` 参数，因为它是框架层自动拦截注入的）。
4.  **注册与暴露**：在 `agent_framework/tools/__init__.py` 的 `__all__` 数组中添加新的工具类，然后进入 `agent.py` 的 `self.tools` 字典中完成初始化注册。

### 4.2. API 与安全管理
- 我们使用 `python-dotenv` 管理敏感配置。请不要在代码中硬编码任何 API Keys。
- 在运行项目前，复制 `.env.example`（如果存在）或直接创建 `.env`，填入 `OPENROUTER_API_KEY`。

### 4.3. 外部依赖策略
- **绝不破坏项目独立性**：如果参考外部工程（如参考代码库的 `heatmap` 算法或 `scatter` 聚类算法），请一定要将其纯数学和逻辑部分内化并复制到本项目的 `_utils.py` 中（如已实现的 `pie_utils.py`, `heatmap_utils.py`, `scatter_utils.py`）。
- **外部大型模型依赖**：类似于 PaddleOCR、SAM3，作为独立组件需要开发者自行安装（见特定文档），在代码中要采取**懒加载 (Lazy Load)** 模式，只在 `run()` 方法中做 Import，防止污染整个框架的初始化速度。

### 4.4 测试准则
- 每次开发完新工具后，必须在 `test_script/` 下编写以该工具为单位的独立测试用例（如 `test_ocr.py`, `test_scatter.py`）。
- 测试用例必须生成可视化的输出图像（如画上 Bounding Box，标出质心与 RGB 颜色值等），并在肉眼核对通过后再将其整合至主链路。

### 4.5 提示词 (Prompt) 管理规范
- **禁止硬编码**：任何发给大模型（LLM/VLM）的自然语言提示词（Prompt），不论长短，**绝不允许**直接硬编码在 Python 源文件中。
- **集中存储**：必须将所有的 Prompt 模板存储于项目根目录的 `prompt/` 文件夹下，以 `.txt` 格式保存。
- **命名规范**：遵循 `<所属类型>_<功能描述>.txt` 的命名规则（例如 `agent_react_system.txt`, `tool_tick_aligner.txt`）。
- **变量注入**：在 Python 中使用 `os.path.join(..., "prompt", "xxx.txt")` 加载模板内容后，使用字符串的 `.replace("{variable_name}", value)` 方法进行动态变量注入。**不要**使用原生的 `string.format()`，因为原始 Prompt 中往往包含给 LLM 示例的 JSON 格式要求（如大量的 `{` 和 `}`），使用 `.format()` 会引发极其繁琐的转义问题，破坏文本的纯净度。