# Local MCP Literature Assistant

**文献搜索与问答助手，支持语义排序、原文溯源与科学证据核对。**

在本地搜索 arXiv、读取 PDF，并用可追溯的正文证据回答文献问题。
提供 Streamlit 聊天界面和独立的本地 stdio MCP Server。

从查找论文到核对回答依据，在一个界面中完成。科学证据核对会提供
片段支持度、数值与实验条件等诊断提示，帮助定位需要回到原文确认的内容；
它不认证结论真假，也不会自动修改回答。

本仓库面向希望在自己电脑上运行文献助手的用户。无需使用 Notebook、
MCP Inspector 或开发评测脚本，安装后直接通过浏览器使用。

## 功能

- **查找论文**：搜索 arXiv，用 SPECTER2 对标题与摘要候选进行语义排序。
- **基于原文问答**：读取 PDF，通过 BM25 查找相关正文片段，生成带引用的回答。
- **追溯回答依据**：将结论与证据片段关联，通过页码引用查看 PDF 原文。
- **核对科学证据**：结合本地 MiniCheck、简单计算、比例与单位规则，以及实验条件差异，提供只读诊断提示。
- **便于人工确认**：逐条查看诊断，并按需下载回答诊断或人工核对材料。
- **多种使用入口**：浏览器内提供搜索、分析、总结快捷任务，也可通过 MCP Tools、Resources 和 Prompts 接入其他客户端。

SPECTER2 只重排最多 10 篇 arXiv 候选，不是全库语义检索。
MiniCheck 只评估所给片段的支持信号，低分不等于论文反驳。

## 安装

以下命令用于 Windows 的 Anaconda Prompt；需要 Git、Conda 和网络连接。

```bat
git clone https://github.com/Remus-cloud/local-mcp-literature-assistant.git
cd local-mcp-literature-assistant
conda create -n literature-bot python=3.12 -y
conda activate literature-bot
python -m pip install -e ".[scientific,verification]"
python scripts/prepare_verification.py
```

环境已存在时跳过创建命令。MiniCheck 准备步骤会下载约 3.2 GB 固定版本权重；
SPECTER2 在首次搜索时下载并缓存权重。已有缓存不必重新下载。
无 Git 时可下载仓库 ZIP，解压后在项目根目录执行后续命令。

已验证的 Windows / Python 3.12 依赖版本记录在
[版本快照](docs/tested-windows-py312.txt)。若需要尽量复现本次验证环境，
可将安装命令换为：

```bat
python -m pip install -c docs/tested-windows-py312.txt -e ".[scientific,verification]"
```

这是测试时的版本约束，不是跨平台锁文件；其他系统不保证使用完全相同的版本。

## 配置接口

首次使用：

```bat
if not exist .env copy .env.example .env
notepad .env
```

不要覆盖已有 `.env`。填写：

```dotenv
DASHSCOPE_API_KEY=你的API密钥
DASHSCOPE_BASE_URL=https://你的服务地址/v1
DASHSCOPE_MODEL=服务商提供的模型ID
LITERATURE_BOT_EXTRA_BODY={}
```

沿用的 `DASHSCOPE_*` 变量名不限定服务商。接口需支持 **OpenAI 格式的
Chat Completions、工具调用及工具结果消息**，结构化绑定默认还使用
`response_format={"type":"json_object"}`。仅支持普通聊天不够。

默认不发送私有扩展参数。若当前 Qwen 服务需要关闭思考模式，可显式设置：

```dotenv
LITERATURE_BOT_EXTRA_BODY={"enable_thinking":false}
```

问答及辅助模型调用使用同一配置，无需修改源码。
扩展配置不允许覆盖 model/messages/tools 等核心字段。
接口兼容性仍需由服务商确认，不承诺所有模型都接受相同的 token 或 JSON 模式参数。
SDK 扩展机制见 [OpenAI Python 官方文档](https://developers.openai.com/api/reference/python)。

不支持 JSON 模式的服务可将 `LITERATURE_BOT_BINDING_FORMAT=prompt_only`；
此模式仍严格校验返回内容，但格式失败风险更高。
不要把真实 `.env`、密钥或研究会话记录提交到仓库。

## 启动与使用

在项目根目录运行：

```bat
python scripts/check_setup.py
python -m streamlit run src/literature_bot/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

浏览器访问 **http://localhost:8501**。保持终端开启，停止时按 `Ctrl+C`。
以后重新进入项目目录并 `conda activate literature-bot`，执行同一启动命令即可。

### 页面中的三个选择

- **论文排序**：SPECTER2 按标题和摘要重排候选论文；正文证据仍由 BM25 检索。
- **运行科学核验**：为新回答增加 MiniCheck 及数值等检查，可能增加耗时与 API 费用。
- **显示诊断**：查看已经生成的提示；隐藏或显示不会重新调用模型。

页面选项只影响当前会话的使用；关闭核验运行只影响之后的新问题，
不会删除已有诊断，结构化引用仍保留。
首次使用建议先完成一次搜索，再针对指定论文提问并开启核验。

示例：

> 搜索 5 篇关于 multimodal object detection 的论文。
>
> 分析 arXiv <填入所选论文的完整 ID> 的实验设置、主要结果和局限，并给出原文依据。

每次提问独立，页面旧消息不构成模型长期记忆；请明确给出 arXiv ID。
“查看原文”的页码是 PDF 文件页序，不一定是印刷页码。

### 如何理解诊断

“支持信号”只表示所提供片段与结论具有一定一致性，不等于结论已被科学证明。
“低支持信号”也不等于论文反驳；可能是证据片段不足、翻译偏差或模型误判。
未评分、证据不足和条件不同均不能当作核验通过。请结合页码引用查看原文，
特别留意表格、实验条件与比较对象。

## 可选：MCP 与命令行

网页会自动启动 MCP 子进程，无需单独运行 Inspector 或 Server。

```text
Streamlit → Chatbot → MCP Client → 本地 stdio MCP Server
                                     ├─ arXiv / SPECTER2
                                     └─ PDF / BM25
回答与证据 → MiniCheck / 数值规则 / 统一诊断 → 只读提示
```

可独立启动：

```bat
literature-bot-mcp
literature-bot-chat --question "搜索三篇视觉语言模型论文"
```

第一条用于 MCP 客户端通信，不会启动网页。支持四个文献 Tools、
论文元数据/页面 Resources，以及搜索/分析/总结 Prompts。

## 限制与数据处理

- 面向本地单用户，不包含公网服务、登录或多用户隔离。
- 检索只覆盖 arXiv 候选及选中的正文片段，不保证全文审阅。
- 指定历史 arXiv 版本时，当前流程可能返回其他版本；涉及版本对比或精确复现时，请先核对返回的论文 ID 与 PDF 版本。
- 复杂任务可能达到工具调用轮次上限而无法完成；可先搜索并确认论文，再将分析任务拆成较小的问题。
- 不支持扫描 PDF OCR；复杂表格、公式和多栏排版可能损坏。
- 数值与条件规则覆盖较窄，不验证因果、统计显著性或实验可重复性。
- MiniCheck 最多检查 8 条已绑定结论；未评分不算通过，分数未经领域校准。
- 语义核验软预算不是硬超时；首次加载和 CPU 推理可能较慢。
- 问题、证据和核验翻译会发送到所配置的 API，可能产生费用。
- 普通聊天仅保存在浏览器会话；论文缓存和主动导出的文件保存在本地。

## 常见问题

**启动提示缺少模块或找不到命令**

确认已激活 `literature-bot` 环境，并在含有 `pyproject.toml` 的项目根目录
执行安装命令。`python scripts/check_setup.py` 可检查配置项是否填写、安装版本
和部分模型缓存，不会调用付费 API；它不代表网络连接或实际推理已验证。

**接口能聊天，但文献问答失败**

检查工具调用与 JSON 模式支持，以及模型 ID、额度和扩展参数。
若服务不支持 JSON 模式，可尝试上文的 `prompt_only` 配置。
配置修改后应停止并重新启动应用。

**模型下载失败或首次搜索很慢**

科学组件需要访问 Hugging Face 下载权重，并占用数 GB 磁盘和额外内存。
先确认网络及磁盘空间，MiniCheck 下载失败后可重新运行
`python scripts/prepare_verification.py`。CPU 推理可能较慢；
暂时不使用科学组件时，可在页面选择不使用 SPECTER2 的排序方式并关闭科学核验。

**arXiv 搜索或 PDF 下载失败**

检查本机网络、代理和 arXiv 访问情况，稍后重试。当前仅支持可提取文字的 PDF，
扫描版论文无法通过本项目自动 OCR。

**端口 8501 被占用**

将启动命令中的 `--server.port 8501` 改为 `--server.port 8502`，
然后访问 `http://localhost:8502`。

## 组件如何配合

SPECTER2 帮助排序候选论文，BM25 查找正文证据，配置的聊天模型负责生成回答。
结构化绑定记录结论与证据的对应关系；MiniCheck 和规则检查在此基础上提供诊断。
这些组件各司其职，不是将主模型替换为另一个“科学大模型”。

主模型可使用满足上文接口要求的服务。论文缓存与模型权重按需在本地生成或下载，
不随仓库分发；安装与使用无需开发用 Notebook 或内部评测数据。
