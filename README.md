# Agentic RAG System

完全ローカルで動作する Agentic RAG (Retrieval-Augmented Generation) システム。
API コスト不要、OSS のみで構成。

## Architecture

LangGraph で制御する 4 エージェント構成:

```
User Query
    │
    ▼
┌──────────────┐    質問あり    ┌──────────────┐
│  Facilitator │ ◄──────────── │  User Input  │
│ (qwen2.5:9b) │ ──────────►  │              │
└──────┬───────┘               └──────────────┘
       │ enriched_prompt
       ▼
┌──────────────┐    自力回答OK   ┌──────────────┐
│  Synthesizer │ ──────────►    │  Validator   │
│ (qwen2.5:2b) │               │ (qwen2.5:9b) │
└──────┬───────┘               └──────┬───────┘
       │ 検索が必要                     │
       ▼                              │ FAIL (avg < 80)
┌──────────────┐                      │
│  Researcher  │                      ▼
│ (LLM不使用)  │ ──── citations ──► Facilitator
└──────────────┘                    (最大3ループ)
```

| Agent | 役割 | モデル |
|-------|------|--------|
| **Facilitator** | クエリ理解・エンリッチ・不足情報の質問 | qwen2.5:9b (planner) |
| **Synthesizer** | 自力回答判断・引用ベース回答生成 | qwen2.5:2b (executor) |
| **Researcher** | Hybrid Search → Filter → Re-rank → 引用ステッチ | LLM 不使用 |
| **Validator** | LLM-as-Judge 4 軸採点 (completeness, accuracy, relevance, faithfulness) | qwen2.5:9b (planner) |

> **現在のモデル選定はテスト用途。** 実用時は qwen2.5:9b (planner) を qwen2.5:27b 相当に、
> qwen2.5:2b (executor) を qwen2.5:9b に変更推奨。

### 検索パイプライン

```
Hybrid Search (BM25 + Semantic HNSW, alpha=0.5)
    ↓
Metadata Filter (source_type, min_score 等)
    ↓
Cross-Encoder Re-rank (ms-marco-MiniLM-L-6-v2)
    ↓
[Optional] ColBERT Late Interaction (top_k=10)
    ↓
Citation Stitching (NotebookLM 方式: original_text 保存)
```

## Tech Stack

| 区分 | 技術 |
|------|------|
| LLM | Ollama (qwen2.5:9b / qwen2.5:2b) |
| Vision LLM | Ollama (qwen3-vl:8b, 取り込み時のみ) |
| Embedding | BAAI/bge-m3 (1024 次元, sentence-transformers) |
| Vector DB | Weaviate 1.28 (Docker, 外部 embedding) |
| Agent Framework | LangGraph (StateGraph + MemorySaver) |
| Re-ranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| UI | Streamlit (4 ページ: Chat / Ingest / Tuning / Evaluation) |
| Python | 3.12 |

## Prerequisites

### 1. Ollama

[Ollama](https://ollama.com/) をインストールし、以下のモデルを pull:

```bash
ollama pull qwen2.5:9b    # Facilitator / Validator (planner)
ollama pull qwen2.5:2b    # Synthesizer (executor)
ollama pull qwen3-vl:8b   # Vision — PDF/PPTX 取り込み時のみ
```

- Ollama は起動後 `http://127.0.0.1:11434` で待ち受け
- OpenAI 互換 API: `http://127.0.0.1:11434/v1`
- `config.yaml` の `ollama.planner_model` / `ollama.executor_model` でモデル名を変更可

> **VRAM 注意 (RTX A4000 8GB):**
> - qwen2.5:9b + qwen2.5:2b を同時ロードするとメモリ 70-80% 程度消費
> - Vision LLM (qwen3-vl:8b) は取り込み時のみ使用。普段は `ollama stop qwen3-vl:8b` でアンロード
> - BGE-M3 は GPU で常駐 (~2.2GB)

### 2. Docker Desktop

[Docker Desktop](https://www.docker.com/products/docker-desktop/) をインストール・起動。

### 3. Python 3.12

Python 3.12 が必要です (3.13 は ragatouille が非対応)。

## Setup

> **`pip install -e .` の CPU 使用率が高い場合は、後述の「インストール手順 (低負荷版)」を使用してください。**

### 通常インストール

```bash
# 1. 仮想環境を作成
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. 依存パッケージのインストール
pip install -e .

# 3. BGE-M3 embedding モデルの事前ダウンロード (~1.5GB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"
```

### インストール手順 (低負荷版) — CPU 97%+ 問題の回避

`pip install -e .` が全依存を一括解決するため、特に PyTorch のダウンロード・展開で
CPU が 97%+ になります。以下の手順で分割インストールすると負荷を抑えられます:

```bash
# Step 1: PyTorch を先に入れる (最大の原因, ~2GB)
# RTX A4000 → CUDA 12.1 版を指定
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu121

# Step 2: 残りを少量ずつインストール
.venv\Scripts\pip install weaviate-client
.venv\Scripts\pip install sentence-transformers
.venv\Scripts\pip install langgraph langchain-core langchain-text-splitters
.venv\Scripts\pip install openai pydantic pydantic-settings pyyaml
.venv\Scripts\pip install pymupdf python-pptx beautifulsoup4
.venv\Scripts\pip install streamlit pandas ranx pytest pytest-mock

# Step 3: プロジェクト本体を依存解決なしで登録 (CPU 負荷ほぼゼロ)
.venv\Scripts\pip install -e . --no-deps
```

### Weaviate 起動・接続確認

```bash
# Weaviate 起動
docker compose up -d

# 接続確認
python -c "
from src.retrieval.weaviate_client import get_client, ensure_collection
client = get_client()
ensure_collection(client)
print('Weaviate OK')
client.close()
"
```

## Usage

### Streamlit UI の起動

```bash
streamlit run src/ui/app.py
```

ブラウザで `http://localhost:8501` が開きます。

### 4 ページ構成

| ページ | 機能 |
|--------|------|
| **Chat** | RAG チャット。エージェントが自動で検索→回答→検証。引用と検証スコアを表示 |
| **Ingest** | ファイルアップロード (TXT, MD, HTML, PY, PDF, PPTX)。Vision LLM トグル |
| **BM25 Tuning** | k1/b パラメーターのグリッドサーチ。評価データ (JSONL) を入力して実行 |
| **Evaluation** | Precision@k / Recall@k / MAP@k / MRR@k を計算。クエリ別ドリルダウン |

### 対応ドキュメント

| 拡張子 | チャンキング戦略 | 備考 |
|--------|----------------|------|
| `.txt`, `.md` | RecursiveChunker (512 chars) | テキスト分割 |
| `.html` | HTMLChunker | p, h1-h4, li タグで分割 |
| `.py` | PythonChunker | def / class 単位で分割 |
| `.pdf` | VisualChunker | 1 ページ = 1 チャンク + Vision LLM (任意) |
| `.pptx` | VisualChunker | 1 スライド = 1 チャンク + Vision LLM (任意) |

## Project Structure

```
src/
├── models.py                  # Pydantic データモデル
├── config.py                  # config.yaml → Settings ローダー
├── ingestion/
│   ├── loaders.py             # ファイル読み込み (全形式)
│   ├── chunkers.py            # 6 種チャンカー + ファクトリ
│   ├── embedder.py            # BGE-M3 embedding ラッパー
│   ├── vision_describer.py    # Vision LLM 画像→テキスト
│   └── pipeline.py            # 取り込みパイプライン統合
├── retrieval/
│   ├── weaviate_client.py     # Weaviate 接続・CRUD
│   ├── hybrid_search.py       # BM25 + Semantic 検索
│   ├── metadata_filter.py     # メタデータフィルター
│   └── colbert_search.py      # ColBERT (任意)
├── reranking/
│   └── cross_encoder.py       # Cross-Encoder 再ランキング
├── generation/
│   ├── llm_client.py          # Ollama 接続 (planner/executor/vision)
│   └── prompts.py             # 全プロンプトテンプレート
├── agents/
│   ├── state.py               # RAGState TypedDict
│   ├── graph.py               # LangGraph StateGraph 定義
│   ├── facilitator.py         # クエリ理解・エンリッチ
│   ├── synthesizer.py         # 回答生成
│   ├── researcher.py          # 検索・引用ステッチ
│   └── validator.py           # LLM-as-Judge 4 軸採点
├── evaluation/
│   ├── metrics.py             # Precision/Recall/MAP/MRR
│   └── bm25_tuner.py          # BM25 k1/b グリッドサーチ
└── ui/
    ├── app.py                 # Streamlit メインアプリ
    └── pages/
        ├── chat.py            # チャットページ
        ├── ingest.py          # 取り込みページ
        ├── tuning.py          # BM25 チューニングページ
        └── evaluation.py      # 評価ページ
```

## Configuration

全設定は `config.yaml` に集約。コード内でのハードコードは禁止。

```python
from src.config import get_config
cfg = get_config()
print(cfg.ollama.planner_model)    # qwen2.5:9b
print(cfg.retrieval.hybrid.alpha)  # 0.5
```

主要設定:

| セクション | 項目 | デフォルト |
|-----------|------|-----------|
| `ollama.planner_model` | Facilitator/Validator モデル | qwen2.5:9b |
| `ollama.executor_model` | Synthesizer モデル | qwen2.5:2b |
| `vision.model_name` | Vision LLM モデル | qwen3-vl:8b |
| `retrieval.hybrid.alpha` | BM25/Semantic バランス | 0.5 |
| `retrieval.bm25.k1` | BM25 k1 | 1.2 |
| `retrieval.bm25.b` | BM25 b | 0.75 |
| `reranking.top_k` | Re-rank 後の返却件数 | 5 |
| `agents.max_loop_count` | Validator→Facilitator 最大ループ | 3 |
| `agents.validation_threshold` | Validator 合格ライン | 80 |

## Testing

```bash
pytest tests/ -v
```

- Unit テスト: 外部依存なし (Weaviate, Ollama をモック)
- Integration テスト: Docker + モデルが必要
- フィクスチャ: `tests/conftest.py`

## Embedding の仕組み

Weaviate は**ベクトルを自分で生成しない** (`DEFAULT_VECTORIZER_MODULE: none`)。

```
Python (src/ingestion/embedder.py)
  └── sentence-transformers (BAAI/bge-m3, GPU 加速)
        ↓ 1024 次元ベクトル
Weaviate  ← 保存 + HNSW/BM25 検索のみ
```

- **取り込み時**: `Embedder.embed_texts()` → Weaviate に保存
- **検索時**: `Embedder.embed_query()` → Weaviate hybrid search に渡す
- 初回起動時に HuggingFace から自動ダウンロード (~1.5GB)
- キャッシュ先: `C:\Users\<user>\.cache\huggingface\hub\`

## License

Private — internal use only.
