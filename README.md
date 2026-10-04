# 文献搜索与问答机器人

一个在本地运行的文献助手：搜索 arXiv 论文、读取 PDF，并根据检索到的正文证据回答问题。通过 Streamlit 提供聊天界面，通过 MCP 将文献能力封装为 Tools、Resources 和 Prompts。

适合查找研究资料、了解论文方法和核对回答出处。当前版本面向本地使用，不是已经部署的公共在线服务。

## 功能

- 根据研究主题搜索 arXiv，查看标题、作者、摘要和论文链接。
- 下载并解析论文 PDF，检索与问题相关的正文片段。
- 使用支持工具调用的 OpenAI 兼容接口生成回答。
- 提供「搜索文献」「分析论文」「总结论文」三个快捷入口。
- 为涉及正文的关键结论提供页码引用；在回答下方展开对应页的提取文本，或打开原始 PDF。
- 提供可独立使用的本地 MCP Server，支持 Tools、Resources 和 Prompts。

## 快速开始

以下命令用于 **Windows 的 Anaconda Prompt**。需要已经安装 Anaconda 或 Miniconda，并能够访问模型服务及 arXiv。

### 1. 获取项目

在 GitHub 仓库页面选择 **Code → Download ZIP**，解压后，将整个项目文件夹放到 `F:\literature-bot`。

确认该目录下直接包含 `pyproject.toml` 和 `src`，而不是又嵌套了一层项目文件夹。也可以使用 Git 克隆仓库；后续命令均需在项目根目录运行。

如果保存到其他位置，请将下面的 `F:\literature-bot` 替换为实际路径。

### 2. 创建环境并安装项目

```bat
conda create -n literature-bot python=3.12 -y
conda activate literature-bot
cd /d F:\literature-bot
python -m pip install -e .
```

如果已创建同名环境，跳过第一条命令。安装命令会根据项目配置自动安装运行所需的软件包，无需逐个安装。

### 3. 配置模型接口

首次配置时执行：

```bat
copy .env.example .env
notepad .env
```

如果已有 `.env`，不要重新复制覆盖，直接编辑现有文件。填写下面三个配置项，并保存文件：

```dotenv
DASHSCOPE_API_KEY=替换为你的API密钥
DASHSCOPE_BASE_URL=https://你的服务地址/v1
DASHSCOPE_MODEL=替换为服务商提供的模型ID
```

这些变量名是当前代码沿用的名称，**不代表必须使用阿里云**。程序使用 OpenAI Python SDK 的 **Chat Completions** 接口，即 `client.chat.completions.create(...)`。

接口及模型需要支持：

- OpenAI 兼容的 Chat Completions 请求和响应格式。
- `tools`、`tool_choice="auto"` 以及响应中的 `tool_calls`。
- 将工具执行结果以 `tool` 消息返回给模型。

`BASE_URL` 填写服务商给出的接口基础地址，不要自行追加 `/chat/completions`。模型 ID 以服务商文档为准，仅支持普通文本聊天的模型不一定能运行本项目。

**当前版本的兼容性注意事项：** 模型请求仍包含 `extra_body={"enable_thinking": False}`。这是服务商扩展参数，不属于通用 OpenAI 接口字段。如果目标服务不接受该参数，需要先删除或按服务商条件传入这一参数。相关调用位于 `src/literature_bot/mcp_chatbot.py` 和 `src/literature_bot/llm_service.py`；仅替换 `.env` 尚不能保证兼容所有服务商。

不要把真实密钥提交到 GitHub。项目的 `.gitignore` 已忽略 `.env`，公开配置示例中只应保留占位值。

### 4. 启动聊天界面

在项目根目录执行：

```bat
python -m streamlit run src/literature_bot/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

浏览器打开 <http://localhost:8501>。如果没有自动打开，手动访问该地址即可。

使用时保持 Anaconda Prompt 窗口开启；停止程序时，在该窗口按 `Ctrl+C`。

以后再次启动只需：

```bat
conda activate literature-bot
cd /d F:\literature-bot
python -m streamlit run src/literature_bot/streamlit_app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

## 如何使用

可以点击快捷入口填写任务，也可以直接在页面底部提问，例如：

```text
帮我搜索 5 篇关于 multimodal object detection 的 arXiv 论文。

请分析 arXiv 论文 2508.19294v2 的主要方法，并给出页码依据。

总结 arXiv 论文 2508.19294v2 的研究问题、主要贡献和局限。
```

涉及具体论文时，建议提供 **arXiv ID**，而不是仅引用上一条消息中的「第二篇」。当前每次提问都是独立任务，页面显示之前的消息，不意味着模型具有跨问题的对话记忆。

回答包含有效页码引用时，可以在回答下方展开「查看原文」核对对应页的提取文本。PDF 链接中的页码是 **PDF 文件页序号**，不一定等于论文印刷页码；浏览器是否直接跳转到指定页，取决于其 PDF 阅读器。

## MCP 接口

程序的主要调用关系为：

```text
Streamlit 界面 → Chatbot（模型与工具调用循环）→ MCP Client
                                                ↓ 本地 stdio
                                            MCP Server
                                                ↓
                                      arXiv / PDF / 正文检索
```

聊天界面会通过 MCP Client 启动本地 Server 子进程，**正常使用时无需另外启动 Server，也无需运行 MCP Inspector**。MCP Inspector 用于开发和验证接口，不是聊天界面的运行桥梁。

### Tools

| 名称 | 用途 |
| --- | --- |
| `literature_search_papers` | 搜索 arXiv 论文 |
| `literature_get_paper_details` | 获取论文详情 |
| `literature_prepare_paper` | 下载、解析 PDF 并准备检索 |
| `literature_retrieve_paper_evidence` | 检索与问题相关的正文证据 |

### Resources

| URI | 内容 |
| --- | --- |
| `literature://guide` | 服务使用指南 |
| `literature://papers/{arxiv_id}/metadata` | 论文元数据 |
| `literature://papers/{arxiv_id}/pages/{page_number}` | 已准备论文的指定页文本 |

读取页面 Resource 前，需要在同一个 Server 会话中调用 `literature_prepare_paper`。页面 Resource 本身不会自动下载论文。

### Prompts

- `search_literature`：搜索文献任务模板。
- `analyze_paper`：围绕指定问题分析论文。
- `summarize_paper`：总结论文。

Prompt 是任务模板，不是另一个模型，也不会仅因获取模板就自动执行工具。

如需将 Server 接入其他 MCP 客户端，在安装项目的 Conda 环境中可使用以下启动命令：

```bat
literature-bot-mcp
```

传输方式为 **stdio**。该命令等待 MCP 客户端通信，不会启动网页或出现聊天输入框。其他客户端的启动配置应使用此环境中 Python 的绝对路径，并以 `-m literature_bot.mcp_integration.server` 为参数，将工作目录设为项目根目录。

## 命令行使用

除了网页，也可以在同一环境、同一项目目录运行：

```bat
literature-bot-chat
```

或只提问一次：

```bat
literature-bot-chat --question "帮我搜索 3 篇关于 multimodal object detection 的论文"
```

## 项目结构

```text
literature-bot/
├── src/literature_bot/       # 文献业务逻辑、Chatbot 和界面
│   ├── mcp_integration/     # MCP Server、Client、Tools、Resources、Prompts
│   ├── mcp_chatbot.py       # 模型与 MCP 工具调用循环
│   ├── streamlit_app.py     # 网页聊天入口
│   └── citations.py         # 页码引用处理
├── notebooks/              # 开发与分步验证用的 Notebook
├── tests/                  # 自动化测试
├── data/papers/             # 运行时下载的论文 PDF 缓存
├── .env.example            # 不含密钥的配置示例
├── .env                    # 本地配置，不提交到 Git
├── .gitignore
├── pyproject.toml          # 项目安装配置
└── README.md
```

使用网页界面不需要先运行 Notebook。`data/papers/` 是论文文件缓存，不是聊天记录。

## 运行测试

在项目根目录、已激活的 Conda 环境中执行：

```bat
python -m pip install -e ".[test]"
python -m pytest
```

自动化测试不能替代真实服务验证。更换模型服务后，还应通过一次实际文献搜索与正文问答，确认工具调用和网络连接正常。

## 限制与数据说明

- 当前文献来源为 arXiv，不覆盖全部学术数据库。
- 当前版本不提供扫描 PDF 的 OCR；公式、表格和多栏排版的文本提取可能不完整。
- 正文问答基于检索出的证据片段，不保证已逐页审阅整篇论文；模型输出仍需人工核对。
- 不持久化聊天记录；页面消息用于当前浏览器会话展示。
- 问题和相关论文片段会发送到所配置的模型服务。调用可能产生费用，数据处理政策由相应服务商决定。
- 当前 MCP 使用本地 stdio，没有实现供公网访问的 HTTP 服务、登录或多用户隔离。上传到 GitHub 不等于已经部署为在线服务。

## 常见问题

**提示找不到模块或命令**

确认已执行 `conda activate literature-bot`，并在包含 `pyproject.toml` 的目录运行 `python -m pip install -e .`。

**提示缺少配置或认证失败**

检查项目根目录中的 `.env`、密钥、接口基础地址和模型 ID。不要在 Issue 或截图中公开密钥。

**模型能聊天，但不能完成文献任务**

确认接口与模型支持工具调用。如果报错指出 `enable_thinking` 不被支持，请按上面的兼容性说明处理。

**arXiv 搜索或 PDF 下载失败**

检查本机网络、代理和 arXiv 是否可访问。网络限制、服务端限流或暂时不可用都可能导致失败，稍后重试。

**8501 端口已被占用**

将启动命令中的 `--server.port 8501` 改为 `--server.port 8502`，然后访问 <http://localhost:8502>。
"# local-mcp-literature-assistant" 
