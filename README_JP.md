# Agentic RAG — 12段階ハンズオン学習プロジェクト

**[English version here](README.md)**

---

> **このリポジトリについて**
>
> 本リポジトリは作者（[@ShigoZhihao](https://github.com/ShigoZhihao)）の個人ハンズオン学習プロジェクトです。現代の Agentic RAG システムがどう組み上がっているかを、ゼロから12段階で実装しながら身につけることを目的としています。L1 の素の LLM 呼び出しから、L12 の自律的マルチエージェントシステムまで、段階的に積み上げていきます。
>
> - **コードはすべて作者本人が手書きで実装しています**。`main.py` も各モジュールもテストも、学習プロセスの一環として一字一句タイプしています。コード生成ツールによる出力そのままの採用、コピペのショートカットは使っていません。
> - **README（本ファイル含む）はすべて Claude（Anthropic）が執筆しています**。設計判断・トレードオフ・バグの議論を作者と Claude の間で行い、その結果を Claude が README として構造化しています。「実装は人間、ドキュメントは統一された AI」という役割分担です。

---

> **謝辞**
>
> 本プロジェクトは [DeepLearning.AI](https://www.deeplearning.ai/) の
> **[Retrieval Augmented Generation (RAG) Course](https://www.deeplearning.ai/courses/retrieval-augmented-generation-rag/)**
> の内容に触発されています。高品質なコースを無償で公開してくださった DeepLearning.AI チームに感謝申し上げます。

---

## 背景と目的

現代の AI システムは **単発の LLM 呼び出し** から、**検索・推論・検証・実行を自律的に行うマルチエージェント構成** へと進化しつつあります。このスタックを上から下まで本当に理解するには、各層を自分で組み上げる以外に近道はありません。

本リポジトリはその学習過程を **12の自己完結したレベル** に分解し、各レベルを **動く MVP** として整備しています。各レベルは以下を満たします:

- 独自の `src/`, `pyproject.toml`, `config.yaml`, `README.md` を持つ
- レベル間で import 依存がなく、独立して動く
- 前レベルに対して **概念を1つだけ** 追加する
- 完全ローカルで動作（Ollama + Weaviate）、API コストゼロ

構成は以下の通り進んでいきます: 素の OpenAI 互換 SDK 呼び出し (L1–L2) → LangChain ベースの RAG (L3–L5) → LangGraph ワークフローとマルチエージェント (L6–L7) → エージェントハーネス高度化 (L8–L12)。

### 現在の進捗

| ステータス | レベル |
|---|---|
| ✅ 実装済み | Level 1, Level 2 |
| 🚧 計画中 | Level 3 〜 Level 12 |

---

## レベル比較

### 概要

| # | レベル | フォルダ | フレームワーク | モデル | MVP でできること |
|---|---|---|---|---|---|
| 1 | Prompt Only | `level_01_prompt_only` | openai SDK | gemma4:e4b | ステートレスな単発チャット — user input → LLM → ストリーミング出力 |
| 2 | Prompt Engineering | `level_02_prompt_engineering` | openai SDK | gemma4:e4b | 会話履歴 + Clarifier→Rewriter パイプラインによるプロンプト改善 |
| 3 | Basic RAG | `level_03_basic_rag` | LangChain | gemma4:e2b | Weaviate 経由でローカル文書をセマンティック検索 |
| 4 | Advanced RAG | `level_04_advanced_rag` | LangChain | gemma4:e4b | ハイブリッド検索（BM25 + semantic）+ re-rank + PDF/PPTX 取り込み |
| 5 | Single Agent | `level_05_single_agent` | LangChain | gemma4:e4b | ReAct スタイルのエージェント（tool use） |
| 6 | Workflow Patterns | `level_06_workflow_patterns` | LangGraph | e2b+e4b | Evaluator-Optimizer ループを state graph で明示的にモデル化 |
| 7 | Multi-Agent | `level_07_multi_agent` | LangGraph | e2b+e4b | 4エージェントループ（Facilitator / Synthesizer / Researcher / Validator）+ HITL |
| 8 | MCP | `level_08_mcp` | LangGraph + MCP | TBD | Model Context Protocol 経由の外部ツール連携 |
| 9 | Harness | `level_09_harness` | LangGraph | TBD | コンテキスト圧縮、永続化、セッション管理 |
| 10 | Sub-Agents | `level_10_sub_agents` | LangGraph | TBD | 分解可能なタスクを並列サブグラフで処理 |
| 11 | Skills | `level_11_skills` | LangGraph | TBD | 合成可能なスキル単位（`SKILL.md` パターン） |
| 12 | Autonomous | `level_12_autonomous` | LangGraph | TBD | スケジュール実行 + CI/CD 統合 |

### 実装済みレベル — 定義された関数一覧

**Level 1** (`level_01_prompt_only`)

| モジュール | 関数 / クラス | 役割 |
|---|---|---|
| `src/config.py` | `OllamaConfig`, `Config`, `get_config()` | YAML → Pydantic 変換 |
| `src/llm_client.py` | `create_client(cfg)` | Ollama 向け OpenAI クライアント生成 |
| `src/llm_client.py` | `stream_response(client, cfg, message)` | ストリーミング応答 + state machine による `<think>` 除去 |
| `main.py` | `main()` | 単一 `while True` ループ。各ターンはステートレス |

**Level 2** (`level_02_prompt_engineering`)

| モジュール | 関数 / クラス | 役割 |
|---|---|---|
| `src/config.py` | `OllamaConfig`, `Config`, `get_config()` | L1 と同じ |
| `src/llm_client.py` | `create_client(cfg)` | L1 と同じ |
| `src/llm_client.py` | `response(client, cfg, message)` | **新規**。非ストリーミング応答。`re.sub` で `<think>` 除去 |
| `src/llm_client.py` | `stream_response(client, cfg, message)` | L1 と同じ |
| `src/prompts.py` | `MAIN_SYSTEM` | Main LLM のペルソナ・禁止事項 |
| `src/prompts.py` | `CLARIFIER_SYSTEM` | Clarifier に `Q1/Q2/Q3` 形式で質問生成を指示 |
| `src/prompts.py` | `REWRITER_SYSTEM` | Rewriter に Q&A を統合させ、スコープ拡張を禁止 |
| `main.py` | `main()` | 外側の会話ループ + 内側の改善ループ（2重） |
| `main_copy.py` | `main()` | 改善プロセスを裏で走らせる試行バージョン |

Level 3〜12: 各レベル実装時に追記予定。

---

## 目標とするテックスタック（Level 7 到達時）

| コンポーネント | 技術 |
|-----------|-----------|
| LLM | Ollama (gemma4:e2b / gemma4:e4b, thinking model) |
| 埋め込み | BAAI/bge-m3 (1024次元, sentence-transformers, GPU) |
| ベクトル DB | Weaviate 1.28 (Docker, 外部埋め込み) |
| エージェントフレームワーク | LangGraph (StateGraph + MemorySaver) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Python | 3.12 |
| UI | CLI のみ（GUI なし） |

Level 1〜2 は `openai`, `pydantic`, `pyyaml` のみ（L2 に `pyreadline3` 追加）。

---

## 前提条件

### 1. Ollama

[Ollama](https://ollama.com/) をインストールし、必要なモデルを pull:

```bash
ollama pull gemma4:e2b    # executor — Level 3 と 6+
ollama pull gemma4:e4b    # planner — Level 1, 2, 4+
```

- Ollama は起動後 `http://127.0.0.1:11434` で待ち受け
- OpenAI 互換 API: `http://127.0.0.1:11434/v1`
- モデル名は各レベルの `config.yaml` で変更可能

### 2. Docker Desktop（Level 3 以降）

Weaviate 用。[Docker Desktop](https://www.docker.com/products/docker-desktop/) をインストール。

### 3. Python 3.12

Python 3.12 必須（3.13 は高レベルで使う一部依存と非互換）。

---

## 使い始め方

各レベルは自己完結しています。任意のレベルを選んで実行してください:

```bash
cd levels/level_01_prompt_only
uv venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS/Linux
uv pip install -e .
python main.py
```

レベルごとのセットアップ・使い方は各レベルの `README.md` 参照。

---

## プロジェクト構成

```
local-agentic-rag/
├── README.md                         # 英語版
├── README_JP.md                      # このファイル
├── CLAUDE.md                         # Claude Code / Copilot 用の規約
├── archive/
│   └── plan.md                       # 当初の計画ドキュメント（アーカイブ済み）
└── levels/
    ├── level_01_prompt_only/         # ✅ openai SDK — 素の LLM チャット
    ├── level_02_prompt_engineering/  # ✅ openai SDK — Clarifier→Rewriter
    ├── level_03_basic_rag/           # 🚧 LangChain — セマンティック検索 + Weaviate
    ├── level_04_advanced_rag/        # 🚧 LangChain — ハイブリッド検索 + re-rank
    ├── level_05_single_agent/        # 🚧 LangChain — ReAct エージェント
    ├── level_06_workflow_patterns/   # 🚧 LangGraph — Evaluator-Optimizer
    ├── level_07_multi_agent/         # 🚧 LangGraph — 4エージェント + HITL
    ├── level_08_mcp/                 # 🚧 LangGraph + MCP
    ├── level_09_harness/             # 🚧 コンテキスト圧縮・永続化
    ├── level_10_sub_agents/          # 🚧 並列サブグラフ生成
    ├── level_11_skills/              # 🚧 合成可能スキル
    └── level_12_autonomous/          # 🚧 スケジュール実行・CI/CD
```

---

## 執筆体制のメモ

- **コード**: 作者（人間）。学習目的でゼロから手書き。Claude によるバグレビュー・設計レビューを経る
- **README**: Claude（Anthropic）。作者との議論を経て Claude が構造化
- **コミット**: 必要に応じて Claude を co-author として記録

---

## ライセンス

[MIT License](LICENSE)
