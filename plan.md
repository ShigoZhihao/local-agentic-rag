# Agentic RAG System - Implementation Plan

## Context

社内ドキュメント（Markdown, HTML, テキスト, **PDF, PPTX**）とコード/スクリプト事例集の両方を対象とした、完全ローカル動作のAgentic RAGシステムを構築する。API課金ゼロ、全てOSSで構成。PDF/PPTXはマルチモーダル（図表、Shape模式図、画像）を含むため、Vision LLMで視覚要素を理解する。

**PCスペック**: RAM 64GB, NVIDIA RTX A4000 Laptop GPU (VRAM 8GB), Intel Xeon W-11955M
**LLM**: Ollama（http://127.0.0.1:11434）。OpenAI互換API経由、model名で呼び分け
**現在のモデル（テスト用）**: planner=qwen2.5:9b、executor=qwen2.5:2b（軽量モデルで動作確認）
**Vision LLM**: qwen3-vl:8b（取込時のみ使用）
**バックエンド**: LangGraph（エージェントグラフ制御）+ Streamlit UI（フロント）

---

## Architecture: 4エージェント・ループ構成

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI                                 │
│  [ユーザ入力] ──────────────────────────────── [結果表示 + 引用]     │
└──────┬──────────────────────────────────────────────▲───────────────┘
       │                                              │
       ▼                                              │ YES or ループ3回超過
┌──────┴──────────────────┐                  ┌────────┴───────────────┐
│  Facilitator            │                  │  Validator             │
│  (ファシリテーター)       │◄─── NO ────────│  (バリデーター)          │
│  qwen2.5:9b (planner)   │  足りない情報を   │  qwen2.5:9b (planner)  │
│                         │  フィードバック    │                        │
│  役割:                   │                  │  役割:                  │
│  ・promptの意図理解       │                  │  ・回答の妥当性検証      │
│  ・不足情報の判断・質問    │                  │  ・enriched promptの    │
│  ・ユーザ入力/お任せ選択   │                  │    要件達成チェック      │
│  ・promptのエンリッチ     │                  │  ・NO→Agent1に差戻し    │
│                         │                  │  ・3回ループ後は理由付き  │
│                         │                  │    で結果表示            │
└──────────┬──────────────┘                  └────────▲───────────────┘
           │ enriched prompt                          │
           ▼                                          │
┌──────────┴──────────────┐                           │
│  Synthesizer            │                           │
│  (シンセサイザー)         │──── 回答生成 ─────────────┘
│  qwen2.5:2b             │
│                         │
│  役割:                   │
│  ・自力回答可否の判断     │
│  ・回答可→直接生成       │
│  ・回答不可→Researcherへ │
│  ・検索結果+コンテキスト  │
│    から回答+引用生成      │
│                         │
└──────────┬──────────────┘
           │ 情報が足りない場合
           ▼
┌──────────┴──────────────┐
│  Researcher             │
│  (リサーチャー)           │
│  コード実行（LLM不使用） │
│                         │
│  役割:                   │
│  ・Hybrid Search実行     │
│    (BM25 + Semantic)     │
│  ・Metadata Filter適用   │
│  ・Cross-Encoder Re-rank │
│  ・ColBERT (オンデマンド) │
│  ・引用コンテンツの       │
│    つなぎ合わせ           │
│  ・生成より引用ステッチ   │
│    (NotebookLM方式)      │
│                         │
│  ┌───────────────────┐  │
│  │    Weaviate       │  │
│  │ BM25+HNSW+Filter  │  │
│  └───────────────────┘  │
└─────────────────────────┘


ドキュメント取込パイプライン（オフライン）:

┌──────────┐   ┌───────────────┐   ┌───────────┐   ┌───────────┐   ┌──────────┐
│ Raw Docs │──▶│  Loader       │──▶│ Chunker   │──▶│ Embedder  │──▶│ Weaviate │
│ .md .html│   │ .pdf→PyMuPDF  │   │(種類別戦略) │   │ (BGE-M3)  │   │ (Docker) │
│ .py .txt │   │ .pptx→画像化  │   │           │   │ GPU加速    │   └──────────┘
│ .pdf     │   │ +Vision LLM   │   └───────────┘   └───────────┘
│ .pptx    │   │  で視覚要素   │
└──────────┘   │  テキスト化   │
               └───────────────┘
```

### エージェントフロー詳細

```
1. ユーザがpromptを入力
2. Facilitator: promptの意図を理解、不足情報があれば質問を返す
   ├── ユーザが追加情報を入力 → Facilitatorがpromptをエンリッチ
   └── ユーザが「お任せ」選択 → Facilitatorが最善のpromptを構築
3. Facilitator → enriched prompt → Synthesizer
4. Synthesizer: 自力で回答できるか判断
   ├── YES → 直接回答生成 → Validator
   └── NO → 必要情報リストをResearcherに渡す
5. Researcher: Hybrid Search → Metadata Filter → Re-rank → 引用ステッチ
   └── 「引用元 + 抜粋テキスト」リスト形式 → Synthesizer
6. Synthesizer: 引用番号付きで回答生成（[1][2]形式） → Validator
7. Validator: LLM-as-Judgeで3点採点（完全性・正確性・関連性、各0-100点）
   ├── 平均80点以上 → ユーザに結果表示
   ├── 平均80点未満 (ループ < 3回) → 具体的不足情報をFacilitatorに
   │   └── Facilitatorがユーザに追加質問（例:「契約書のXX条が不足」）
   │       → ユーザ追加入力で即再開
   └── 平均80点未満 (ループ >= 3回) → 結果表示 + 採点詳細 + 具体的不足情報
```

### 共有状態（LangGraph State）

全エージェントが共有する状態オブジェクト:
```python
class RAGState(TypedDict):
    user_query: str                       # 元のユーザ入力
    enriched_prompt: str                  # Facilitatorがエンリッチしたプロンプト
    chat_history: list[ConversationTurn]  # 全会話履歴
    citations: list[Citation]             # Researcher検索結果
    answer: str                           # Synthesizer回答
    validation: ValidationResult          # Validator検証結果
    loop_count: int                       # 現在のループ回数
    needs_user_input: bool                # ユーザ入力待ちフラグ
    feedback_to_user: str                 # Facilitator→ユーザへの質問/フィードバック
```

### Validator採点基準（LLM-as-Judge）

```
Validatorは以下の4観点で回答を0-100点で採点:

1. 完全性 (Completeness): enriched promptの全要件が回答でカバーされているか
2. 正確性 (Accuracy): 引用元テキストと回答内容に矛盾がないか
3. 関連性 (Relevance): 回答がユーザの元の意図に沿っているか
4. 忠実性 (Faithfulness): 引用にない情報を生成していないか（ハルシネーション検出）

判定: 4点の平均 >= 80 → YES (合格)
      4点の平均 < 80 → NO (差戻し) + 低スコア項目の具体的理由
```

---

## Technology Stack

| コンポーネント | ライブラリ / モデル | 理由 |
|---|---|---|
| **Embedding** | `BAAI/bge-m3` (sentence-transformers) | 1024次元, Dense+Sparse+ColBERTモード対応 |
| **Vector DB** | Weaviate 1.28+ (Docker) | HNSW内蔵、BM25内蔵、ハイブリッド検索対応 |
| **ColBERT** | `ragatouille` (colbert-ir/colbertv2.0) | ColBERTを簡単にラップ。オンデマンド使用 |
| **Cross-Encoder** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22Mパラメータ、CPU高速、MS MARCO訓練済 |
| **LLM Client** | `openai` (Ollama OpenAI互換API経由) | 共通URL `http://127.0.0.1:11434/v1`、model名で呼び分け |
| **Agent Graph** | `langgraph` | エージェント状態管理・条件分岐・ループ制御 |
| **Vision LLM** | `qwen3-vl:8b` (Ollama) | PDF/PPTXの視覚要素理解。取込時のみ使用 |
| **PDF処理** | `pymupdf` (fitz) | PDF→画像レンダリング + テキスト抽出 |
| **PPTX処理** | `python-pptx` + `win32com`/LibreOffice | テキスト/Shape抽出 + スライド画像化 |
| **Chunking** | `langchain-text-splitters` + `beautifulsoup4` | Recursive/HTML/Semantic各戦略に対応 |
| **Config** | `pyyaml` + `pydantic` | YAML設定を型安全なPydanticモデルに変換 |
| **Evaluation** | `ranx` | MAP, MRR, Precision, Recall計算 |
| **UI** | `streamlit` | クエリ入力、結果表示、BM25チューニング画面 |
| **Testing** | `pytest` | 標準テストフレームワーク |

**Python バージョン**: 3.11または3.12（ragatouille互換性のため。3.13は避ける）

---

## Project Structure

```
D:/Work_Data/Initiative/RAG/
├── CLAUDE.md                      # Claude Code / Copilot用プロジェクト規約
├── pyproject.toml                 # 依存関係管理 (PEP 621)
├── config.yaml                    # 全チューニングパラメータ
├── docker-compose.yaml            # Weaviateコンテナ定義
├── .env.example                   # 環境変数テンプレート
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── config.py                  # config.yaml + .env → Pydantic Settings
│   ├── models.py                  # 共通データモデル
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pipeline.py            # 取込オーケストレータ
│   │   ├── loaders.py             # ファイル読込: .txt, .md, .html, .py, .pdf, .pptx
│   │   ├── vision_describer.py    # Vision LLMでスライド/ページ画像を文章化
│   │   ├── chunkers.py            # 全チャンキング戦略
│   │   └── embedder.py            # BGE-M3ラッパー (GPU加速)
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── weaviate_client.py     # Weaviate接続、スキーマ作成、CRUD
│   │   ├── hybrid_search.py       # BM25+Semantic統合検索 + メタデータフィルタ
│   │   ├── colbert_search.py      # ColBERTオンデマンド検索
│   │   └── metadata_filter.py     # 検索後メタデータフィルタリング
│   │
│   ├── reranking/
│   │   ├── __init__.py
│   │   └── cross_encoder.py       # Cross-Encoder Re-ranker
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm_client.py          # Ollama接続 (OpenAI互換)
│   │   ├── prompts.py             # 全プロンプトテンプレート
│   │   └── generator.py           # 回答生成エンジン
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── state.py               # RAGState定義 (LangGraph共有状態)
│   │   ├── facilitator.py         # Facilitator: prompt理解・エンリッチ・質問
│   │   ├── synthesizer.py         # Synthesizer: 回答生成・自力判断
│   │   ├── researcher.py          # Researcher: 検索・フィルタ・Re-rank・引用ステッチ
│   │   ├── validator.py           # Validator: LLM-as-Judge 3点採点
│   │   └── graph.py               # LangGraphグラフ定義 (ノード+条件分岐+ループ)
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Precision, Recall, MAP, MRR
│   │   ├── bm25_tuner.py          # k1, bパラメータグリッドサーチ
│   │   └── eval_runner.py         # 評価スイート実行
│   │
│   └── ui/
│       ├── __init__.py
│       ├── app.py                 # Streamlitメインアプリ
│       ├── pages/
│       │   ├── chat.py            # チャットインターフェース
│       │   ├── ingest.py          # ドキュメント取込ページ
│       │   ├── tuning.py          # BM25チューニングページ
│       │   └── evaluation.py      # 評価ダッシュボード
│       └── components/
│           ├── chat_message.py    # チャットメッセージ表示コンポーネント
│           └── source_viewer.py   # ソース引用表示コンポーネント
│
├── tests/
│   ├── conftest.py
│   ├── test_chunkers.py
│   ├── test_retrieval.py
│   ├── test_reranking.py
│   ├── test_metrics.py
│   └── test_workflow.py
│
├── data/
│   ├── raw/                       # 投入するソースドキュメント
│   └── eval/
│       └── test_queries.json      # 評価用クエリ+正解ドキュメントID
│
└── .reference/                    # 参考PDF (既存)
```

---

## Key Data Models (`src/models.py`)

```python
from pydantic import BaseModel, Field
from enum import Enum

class SourceType(str, Enum):
    TXT = "txt"
    MD = "md"
    HTML = "html"
    PY = "py"
    PDF = "pdf"
    PPTX = "pptx"

class ChunkStrategy(str, Enum):
    RECURSIVE = "recursive"
    HTML = "html"
    PYTHON = "python"
    SEMANTIC = "semantic"
    EXAMPLE = "example"
    VISUAL = "visual"          # Vision LLMで生成したスライド/ページ説明

class Document(BaseModel):
    doc_id: str
    content: str
    source_file: str
    source_type: SourceType
    metadata: dict = Field(default_factory=dict)

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    chunk_strategy: ChunkStrategy
    source_file: str
    source_type: SourceType
    page_number: int | None = None      # PDF/PPTXのページ/スライド番号
    image_path: str | None = None       # レンダリングされた画像パス
    embedding: list[float] | None = None
    metadata: dict = Field(default_factory=dict)

class SearchResult(BaseModel):
    chunk: Chunk
    score: float
    search_type: str

class Citation(BaseModel):
    """NotebookLM方式: 元テキストを引用、生成し直さない"""
    source_file: str
    page_number: int | None = None
    original_text: str                  # 元のテキストそのまま
    relevance_score: float

class AgentRoute(str, Enum):
    RETRIEVE_AND_ANSWER = "retrieve_and_answer"
    DIRECT_ANSWER = "direct_answer"
    CLARIFY = "clarify"

class ValidationScores(BaseModel):
    completeness: int                   # 0-100: enriched promptの要件カバー率
    accuracy: int                       # 0-100: 引用元との整合性
    relevance: int                      # 0-100: ユーザ意図との関連性
    faithfulness: int                   # 0-100: 引用外情報の生成がないか
    average: float                      # 4点平均

class ValidationResult(BaseModel):
    scores: ValidationScores
    is_valid: bool                      # average >= 80
    reason: str                         # 判断理由（低スコア項目の具体的説明）
    missing_info: list[str] = Field(default_factory=list)  # 具体的不足情報

class ConversationTurn(BaseModel):
    """Streamlit UI用の会話履歴"""
    role: str                           # "user", "facilitator", "synthesizer", "system"
    content: str
    citations: list[Citation] = Field(default_factory=list)
    validation: ValidationResult | None = None
    loop_count: int = 0
```

---

## Configuration (`config.yaml`) 主要設定

```yaml
ollama:
  base_url: "http://127.0.0.1:11434/v1"  # OpenAI互換API
  planner_model: "qwen2.5:9b"            # Facilitator/Validator (テスト用; 本番は27b推奨)
  executor_model: "qwen2.5:2b"           # Synthesizer (テスト用; 本番は9b推奨)
  temperature: 0.2
  max_tokens: 2048

vision:
  base_url: "http://127.0.0.1:11434/v1"
  model_name: "qwen3-vl:8b"              # 取込時のみ使用
  max_tokens: 1024
  dpi: 200                                         # PDF/PPTXレンダリング解像度

embeddings:
  model_name: "BAAI/bge-m3"
  device: "cuda"
  batch_size: 32
  dimension: 1024

weaviate:
  host: "localhost"
  port: 8080
  grpc_port: 50051
  collection_name: "Documents"

chunking:
  default_strategy: "recursive"
  recursive: { chunk_size: 512, overlap_percent: 0, separators: ["\n\n", "\n", ". ", " "] }
  html: { tags_to_split: ["p", "h1", "h2", "h3", "h4", "li"] }
  python: { split_on: ["def ", "class "], include_decorator: true }
  semantic: { cosine_threshold: 0.75, min_chunk_size: 100, max_chunk_size: 1000 }
  example: { delimiter: "---" }
  visual: { one_chunk_per_page: true }  # PDF/PPTXは1ページ=1チャンク(テキスト+視覚説明)

retrieval:
  bm25: { k1: 1.2, b: 0.75, top_k: 20 }
  semantic: { top_k: 20, distance_metric: "cosine" }
  hybrid: { alpha: 0.5, top_k: 20 }
  colbert: { enabled: false, model_name: "colbert-ir/colbertv2.0", top_k: 10 }

reranking:
  model_name: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 5
  batch_size: 16

agents:
  max_loop_count: 3                     # Validator→Facilitatorの最大ループ回数
  validation_threshold: 80              # 平均スコア合格ライン
  facilitator:
    model: "planner"                    # qwen2.5:9b
  synthesizer:
    model: "executor"                   # qwen2.5:2b
  validator:
    model: "planner"                    # qwen2.5:9b
    scoring_criteria:
      - "completeness"                  # enriched promptの全要件カバー
      - "accuracy"                      # 引用元テキストとの整合性
      - "relevance"                     # ユーザ元意図との関連性
      - "faithfulness"                  # 引用外情報の生成がないか

evaluation:
  metrics: ["precision@5", "recall@10", "map@10", "mrr@10"]
  bm25_tuning:
    k1_range: [0.5, 0.75, 1.0, 1.2, 1.5, 2.0]
    b_range: [0.0, 0.25, 0.5, 0.75, 1.0]
    optimize_metric: "map@10"
```

---

## Implementation Phases

### Phase 1: 基盤構築
**作成ファイル**: `pyproject.toml`, `config.yaml`, `docker-compose.yaml`, `.env.example`, `.gitignore`, `CLAUDE.md`, `src/__init__.py`, `src/config.py`, `src/models.py`, `src/retrieval/weaviate_client.py`

1. Python 3.12仮想環境作成、依存関係インストール

   > **`uv pip install -e .` でCPU 97%+ になる場合（解決策）**
   >
   > `uv pip install -e .` が全依存を一括解決するため、PyTorchのダウンロード・展開でCPUが跳ね上がる。
   > 以下の手順で分割インストールすると負荷を抑えられる:
   >
   > ```bash
   > # Step 1: PyTorchを先に入れる (最大の原因, ~2GB, CUDA 12.1版)
   > uv pip install torch --index-url https://download.pytorch.org/whl/cu121
   >
   > # Step 2: 残りを少量ずつインストール
   > uv pip install weaviate-client
   > uv pip install sentence-transformers
   > uv pip install langgraph langchain-core langchain-text-splitters
   > uv pip install openai pydantic pydantic-settings pyyaml
   > uv pip install pymupdf python-pptx beautifulsoup4
   > uv pip install streamlit pandas ranx pytest pytest-mock
   >
   > # Step 3: プロジェクト本体を依存解決なしで登録 (CPU負荷ほぼゼロ)
   > uv pip install -e . --no-deps
   > ```
2. `src/models.py` — 上記のPydanticデータモデル全体
3. `src/config.py` — config.yamlをPydantic Settingsクラスに読込
4. `docker-compose.yaml` — Weaviate単一ノード（`DEFAULT_VECTORIZER_MODULE: 'none'`）
5. `src/retrieval/weaviate_client.py` — Weaviate接続、コレクション作成（BM25パラメータ含む）、チャンクCRUD
6. `CLAUDE.md` — コーディング規約（Claude Code + Copilot両方が従う。詳細は下記セクション参照）
7. 動作確認: Weaviate起動→接続→テストデータ投入→取得

### Phase 1.5: 早期スモークテスト（最短ループ検証）
**目的**: テキストファイル10個だけで「Facilitator → Synthesizer（直接回答） → Validator」の最短ループを動かす

1. Ollamaでモデル2つロード
2. テスト用テキストファイル10個を`data/raw/`に配置
3. 簡易版`graph.py`: Facilitator → Synthesizer（直接回答のみ） → Validator の3ノード
4. Pythonスクリプトで実行、LangGraphの`debug=True`でステート遷移を確認
5. Validator採点結果（4軸スコア）が正しく返るか確認
6. この段階でOllamaとの接続、モデル呼び分け、LangGraphのcheckpointerが動作することを検証

### Phase 2: ドキュメント取込パイプライン（テキスト系）
**作成ファイル**: `src/ingestion/loaders.py`, `src/ingestion/chunkers.py`, `src/ingestion/embedder.py`, `src/ingestion/pipeline.py`, `tests/test_chunkers.py`

1. `loaders.py` — .txt, .md, .html, .pyファイル読込。メタデータ付与
2. `chunkers.py` — 5種類のチャンキング戦略:
   - **RecursiveChunker**: overlap設定可能(初期0%)
   - **HTMLChunker**: p/h1-h4/liタグ境界で分割
   - **PythonChunker**: `def `と`class `境界で分割
   - **SemanticChunker**: コサイン類似度閾値で分割
   - **ExampleChunker**: デリミタで事例ごと分割
   - 全チャンカーは`BaseChunker`(ABC)継承、`chunk(doc) -> list[Chunk]`統一
   - `get_chunker(source_type, config)` ファクトリ関数
3. `embedder.py` — BGE-M3 GPU(cuda)ロード、`embed_texts() -> list[list[float]]`
4. `pipeline.py` — load→種類検出→チャンカー選択→チャンク→エンベッド→Weaviate投入
5. テスト: 少量テストドキュメントで取込→Weaviate確認

### Phase 3: マルチモーダル取込（PDF/PPTX）
**作成ファイル**: `src/ingestion/vision_describer.py`, `loaders.py`拡張

1. `loaders.py`にPDF/PPTXローダー追加:
   - **PDF**: `pymupdf`でテキスト抽出 + ページ画像レンダリング(DPI設定可)
   - **PPTX**: `python-pptx`でテキスト/Shape抽出 + `win32com`でスライド画像エクスポート
     - `win32com`が使えない場合(PowerPoint未インストール): LibreOfficeインストールを推奨
   - 各ページ/スライドの画像を`data/rendered/`に保存
2. `vision_describer.py` — Vision LLM (qwen3-vl:8b) でスライド/ページ画像を文章化:
   - Ollamaの3つ目インスタンス(Ollamaの別プロセス)または取込時のみモデル切替
   - プロンプト: 「このスライド/ページに含まれるテキスト、図表、模式図、画像の内容を詳細に日本語で説明してください」
   - 出力: 視覚要素の構造化された説明文
3. PDF/PPTXチャンキング戦略:
   - **VisualChunker**: 1ページ/スライド = 1チャンク（抽出テキスト + Vision LLM説明を結合）
   - `chunk.image_path`にレンダリング画像パスを保存（UI表示用）
   - `chunk.page_number`でページ番号追跡
4. VRAM管理: 取込時はVision LLM + BGE-M3を順次ロード（同時ロードしない）
5. テスト: サンプルPDF/PPTXで取込→チャンク内容確認

### Phase 4: 検索 + Re-ranking
**作成ファイル**: `src/retrieval/hybrid_search.py`, `src/retrieval/metadata_filter.py`, `src/retrieval/colbert_search.py`, `src/reranking/cross_encoder.py`, `tests/test_retrieval.py`, `tests/test_reranking.py`

1. `hybrid_search.py` — Weaviate `collection.query.hybrid()` 使用。alpha値でBM25/Semantic比率制御
2. `metadata_filter.py` — source_type, 日付範囲, カスタムタグでフィルタ
3. `colbert_search.py` — RAGatouille経由ColBERT検索。オンデマンド
4. `cross_encoder.py` — `cross-encoder/ms-marco-MiniLM-L-6-v2`、CPU実行
5. テスト: 既知ドキュメントで検索精度、re-rank前後の順位変化確認

### Phase 5: LangGraph + 4エージェント・ワークフロー
**作成ファイル**: `src/generation/llm_client.py`, `src/generation/prompts.py`, `src/generation/generator.py`, `src/agents/state.py`, `src/agents/facilitator.py`, `src/agents/synthesizer.py`, `src/agents/researcher.py`, `src/agents/validator.py`, `src/agents/graph.py`, `tests/test_workflow.py`

1. `llm_client.py` — OpenAI互換クライアント（共通URL、model名で呼び分け）:
   ```python
   from openai import OpenAI

   client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")

   def call_planner(messages: list[dict]) -> str:
       """Facilitator/Validator用 (qwen2.5:9b)"""
       response = client.chat.completions.create(
           model="qwen2.5:9b-claude-4.6-opus-reasoning-distilled",
           messages=messages, temperature=0.3
       )
       return response.choices[0].message.content

   def call_executor(messages: list[dict]) -> str:
       """Synthesizer用 (qwen2.5:2b)"""
       response = client.chat.completions.create(
           model="qwen2.5:2b",
           messages=messages, temperature=0.3
       )
       return response.choices[0].message.content
   ```

2. `prompts.py` — 全プロンプトテンプレート:
   - `FACILITATOR_SYSTEM`: intent理解、不足情報判断、prompt enrichment
   - `SYNTHESIZER_SYSTEM`: 引用番号付き回答生成（[1][2]形式）、引用にない情報の生成禁止
   - `RESEARCHER_CITATION_TEMPLATE`: 引用ステッチ指示（NotebookLM方式）
   - `VALIDATOR_SYSTEM`: LLM-as-Judgeで3点採点（完全性・正確性・関連性、各0-100）

3. `state.py` — LangGraph共有状態（Pydanticモデルと統一）:
   ```python
   from typing import TypedDict, Annotated
   from src.models import Citation, ValidationResult, ConversationTurn

   class RAGState(TypedDict):
       user_query: str
       enriched_prompt: str
       chat_history: list[ConversationTurn]      # Pydanticモデル使用
       citations: list[Citation]                  # Pydanticモデル使用（型安全）
       answer: str
       validation: ValidationResult | None        # Pydanticモデル使用（型安全）
       loop_count: int
       needs_user_input: bool
       feedback_to_user: str
       needs_research: bool                       # Synthesizer→Researcher分岐用
   ```
   ※ `citations`と`validation`はPydanticモデルを直接使用し型安全性を確保

4. `facilitator.py` — Facilitatorノード:
   - `run(state: RAGState) -> RAGState`: 意図理解+エンリッチ
   - Validatorからのフィードバック時: `missing_info`を元に具体的な追加質問を生成
   - ユーザが「お任せ」→ 最善のpromptを自動構築、`needs_user_input=False`に設定

5. `synthesizer.py` — Synthesizerノード:
   - `run(state: RAGState) -> RAGState`: 自力回答可否判断 + 回答生成
   - 検索結果がある場合: 引用番号付きで回答 `"[1]によると...、また[2]では..."`
   - Researcherの出力フォーマットをそのまま活用

6. `researcher.py` — Researcherノード (LLM不使用):
   - `run(state: RAGState) -> RAGState`: 検索→フィルタ→Re-rank→引用構造化
   - ColBERT使用時: `top_k=10`固定、結果を即座にPythonリストに変換（インデックス全体は保持しない。RAM節約）
   - 出力形式:
     ```python
     citations = [
         {"id": 1, "source_file": "manual.pdf", "page": 3,
          "original_text": "元テキストそのまま", "score": 0.95},
         {"id": 2, "source_file": "example.py", "page": None,
          "original_text": "def configure_logging()...", "score": 0.87},
     ]
     ```

7. `validator.py` — Validatorノード (LLM-as-Judge):
   - `run(state: RAGState) -> RAGState`:
     - 3点採点: completeness(0-100), accuracy(0-100), relevance(0-100)
     - `average >= 80` → 合格、`needs_user_input=False`
     - `average < 80` → 不合格、`missing_info`に具体的不足項目
     - 3ループ目で不合格: 回答+採点詳細+具体的不足情報を返す

8. `graph.py` — LangGraphグラフ定義（Human-in-the-Loop対応）:
   ```python
   from langgraph.graph import StateGraph, END
   from langgraph.checkpoint.memory import MemorySaver

   def build_graph():
       graph = StateGraph(RAGState)

       # ノード登録
       graph.add_node("facilitator", facilitator.run)
       graph.add_node("synthesizer", synthesizer.run)
       graph.add_node("researcher", researcher.run)
       graph.add_node("validator", validator.run)

       # エッジ定義
       graph.set_entry_point("facilitator")
       graph.add_edge("facilitator", "synthesizer")

       # Synthesizer → 直接回答 or Researcher経由
       graph.add_conditional_edges("synthesizer", route_after_synthesizer,
           {"needs_research": "researcher", "has_answer": "validator"})

       graph.add_edge("researcher", "synthesizer")

       # Validator → 合格/不合格/ループ超過
       graph.add_conditional_edges("validator", route_after_validator,
           {"approved": END, "rejected": "facilitator", "max_loops": END})

       # Human-in-the-Loop: Facilitatorがユーザ入力を求める場合に中断
       # Streamlitのst.session_stateでグラフ状態を保持し、
       # ユーザ入力後にgraph.stream()で再開
       return graph.compile(
           checkpointer=MemorySaver(),
           interrupt_before=["facilitator"],  # ユーザ入力待ちで中断
           debug=True                         # デバッグログ有効（開発中）
       )
   ```

   **Streamlit連携パターン**:
   ```python
   # src/ui/pages/chat.py (概要)
   # st.session_stateにthread_idとgraph状態を保持
   # graph.stream()でイベント駆動
   # interrupt時: UIでユーザ入力を受け付け → graph.update_state() → stream再開
   # 3ループ後のフォールバック: 回答+採点+不足情報表示 → ユーザ追加入力で即再開可能
   ```

### Phase 6: 評価 + BM25チューニング
**作成ファイル**: `src/evaluation/metrics.py`, `src/evaluation/bm25_tuner.py`, `src/evaluation/eval_runner.py`, `data/eval/test_queries.json`

1. `metrics.py` — ranxライブラリ経由: Precision@k, Recall@k, MAP@k, MRR@k
2. `bm25_tuner.py` — k1×bグリッドサーチ（コレクション再作成→再インデックス→評価→MAP記録）
3. `eval_runner.py` — 検索モード別比較表
4. テスト評価データセット(20-50クエリ + 正解ドキュメントID)

### Phase 7: Streamlit UI
**作成ファイル**: `src/ui/app.py`, `src/ui/pages/chat.py`, `src/ui/pages/ingest.py`, `src/ui/pages/tuning.py`, `src/ui/pages/evaluation.py`, `src/ui/components/chat_message.py`, `src/ui/components/source_viewer.py`

1. **チャットページ** (`pages/chat.py`):
   - チャット形式の会話UI
   - Facilitatorの質問表示 + ユーザ入力 or 「お任せ」ボタン
   - 回答表示 + インライン引用（クリックで元テキスト展開）
   - PDF/PPTXソースの場合、レンダリング画像も表示
   - Validatorの検証結果表示（ループ状況）
   - サイドバー: 検索モード切替、re-ranking ON/OFF、ColBERT ON/OFF、alphaスライダー

2. **取込ページ** (`pages/ingest.py`):
   - フォルダ指定でバッチ取込
   - ファイル種類別のチャンキング戦略選択
   - 取込進捗表示
   - 取込済みドキュメント一覧

3. **チューニングページ** (`pages/tuning.py`):
   - BM25 k1, bスライダー + 実行ボタン
   - グリッドサーチ結果ヒートマップ
   - 最適パラメータ推薦表示

4. **評価ダッシュボード** (`pages/evaluation.py`):
   - 検索モード別メトリクス表
   - クエリ別詳細結果

---

## マルチモーダル処理の詳細

### PDF処理フロー
```
PDF → pymupdf.open(path)
  ├── page.get_text() → テキスト抽出
  ├── page.get_pixmap(dpi=200) → ページ画像レンダリング
  └── Vision LLM(画像) → 図表・画像の説明文
      ↓
  チャンク = テキスト + "\n\n[視覚要素]\n" + Vision LLM説明
```

### PPTX処理フロー
```
PPTX → python-pptx.Presentation(path)
  ├── slide.shapes → テキスト・Shape抽出 (python-pptxのみ)
  ├── win32com → スライド画像エクスポート (Windows専用、PowerPoint必須)
  │   ※LibreOfficeは使わない。PowerPoint未インストールの場合はテキスト抽出のみ
  └── Vision LLM(画像) → 模式図・図表の説明文
      ↓
  チャンク = テキスト + "\n\n[視覚要素]\n" + Vision LLM説明
```
※ python-pptxだけでもテキスト+Shape内テキストは抽出可能。
  win32comがあれば画像化→Vision LLMで補完。なくても動作する設計にする。

### VRAM管理戦略 (8GB制約)
取込時とクエリ時でモデルを分離:
- **取込時**: Vision LLM (Ollamaの別プロセス) → 完了後アンロード → BGE-M3 (GPU) でembedding
- **クエリ時**: BGE-M3 (GPU, ~2.2GB) + Cross-Encoder (CPU, ~90MB) のみ
- Ollamaの2インスタンス (27b, 9b) はRAM上で動作（VRAMが足りない場合はCPU推論）

---

## 引用ステッチ方式（NotebookLM参考）

Researcher (Agent3) の核心的な設計方針:

1. **生成より引用**: 検索で取得したチャンクの原文をそのまま保持する
2. **つなぎ合わせ**: 複数チャンクの関連部分を論理的な順序で連結
3. **メタデータ保存**: 各引用にソースファイル名、ページ番号、元テキストを記録
4. **Synthesizerへの渡し方**: 引用原文をそのまま渡し、Synthesizerはそれを元に回答を構築
5. **ハルシネーション抑制**: Synthesizerのプロンプトで「引用にない情報を生成しない」と明示

```python
class Researcher:
    def search(self, info_needs: list[str]) -> list[Citation]:
        all_citations = []
        for need in info_needs:
            results = self.hybrid_search.search(need)
            results = self.metadata_filter.apply(results)
            results = self.cross_encoder.rerank(need, results)

            for result in results:
                all_citations.append(Citation(
                    source_file=result.chunk.source_file,
                    page_number=result.chunk.page_number,
                    original_text=result.chunk.content,  # 原文そのまま
                    relevance_score=result.score
                ))

        # 重複排除、スコア順ソート
        return self._deduplicate_and_sort(all_citations)
```

---

## CLAUDE.md 骨子

```markdown
# Agentic RAG System - Project Rules

## Architecture
LangGraphで制御する4エージェント構成:
Facilitator → Synthesizer → Researcher → Validator (ループ最大3回)
バックエンド: src/agents/graph.py (LangGraph StateGraph)
共有状態: src/agents/state.py (RAGState TypedDict)

## Coding Conventions
- 全関数: 型ヒント必須 (パラメータ + 戻り値)
- 公開関数: Google-style docstring必須
- データ構造: src/models.pyのPydantic BaseModelを使用
- 設定値: src/config.py経由 (ハードコード禁止)
- プロンプト: src/generation/prompts.pyに定数として管理
- LLM呼出: src/generation/llm_client.py経由のみ (直接openai禁止)
- エラー: try/except + logging (silent pass禁止)
- import順: stdlib → third-party → local

## File Patterns
- agents/はBaseAgent ABCを継承しない。各エージェントは独立した関数ベースモジュール
- chunkers/はBaseChunker ABCを継承。chunk(doc) -> list[Chunk]メソッド統一
- 検索モジュールはsearch(query, top_k, **kwargs) -> list[SearchResult]を公開

## Testing
pytest tests/ -v. conftest.pyに共通フィクスチャ。
Weaviate/Ollama不要のユニットテスト + 必要な統合テスト。

## Running
streamlit run src/ui/app.py
```

---

## ログ・デバッグ方針

1. **LangGraph debug=True**: 開発中はgraph.compile(debug=True)でステート遷移を全てログ出力
2. **checkpoint保存**: MemorySaver()で全ステップの状態を保存。ループのデバッグ時に任意のステップに戻れる
3. **Python logging**: 全モジュールで`logging.getLogger(__name__)`使用。レベル: DEBUG/INFO/WARNING/ERROR
4. **Streamlit表示**: デバッグモード時、サイドバーにRAGStateの現在値、Validator採点詳細、ループカウントを表示

---

## 参考資料からの提案・訂正

1. **BGE-M3をメインEmbeddingモデルに推奨**: Dense + Sparse + ColBERTの3モードを1モデルで対応
2. **スクリプト事例集のチャンキング**: ユーザのアイディア通り事例ごとにチャンクが正解
3. **Weaviate内蔵BM25**: 外部ライブラリ不要。k1/b変更にはコレクション再作成が必要
4. **メタデータフィルタリング**: 検索後フィルタリングはユーザ指定通り正しいアプローチ
5. **PDF/PPTX**: Vision LLM方式を採用。テキスト抽出 + 視覚要素文章化の二段構え
6. **NotebookLM参考**: Researcherは生成より引用原文ステッチを優先しハルシネーション抑制

---

## 実行時の最初のアクション

プラン内容を `D:\Work_Data\Initiative\RAG\plan.md` に保存する（全実装の参照元）。

---

## Verification (動作確認方法)

1. **Phase 1**: `docker compose up -d` → Python接続テスト → Weaviateにテストデータ投入/取得
2. **Phase 2**: テスト用.md/.html/.pyファイルで取込パイプライン実行 → Weaviateでチャンク数/内容確認
3. **Phase 3**: サンプルPDF/PPTXを取込 → Vision LLM説明文の品質確認 → 画像パス保存確認
4. **Phase 4**: `pytest tests/test_retrieval.py tests/test_reranking.py -v` → 既知ドキュメントの検索精度確認
5. **Phase 5**: Ollamaでモデルロード → LangGraphグラフをPythonで実行 → 4エージェントループ動作確認
   - Facilitator質問生成 → Synthesizer回答 → Validator採点(3点×0-100) → ループ/合格の動作
6. **Phase 6**: `pytest tests/test_metrics.py -v` → 評価データセットでメトリクス算出
7. **Phase 7**: `streamlit run src/ui/app.py` → 全ページ操作確認
   - チャット: 質問→Facilitator質問→お任せ/追加入力→回答+引用番号表示
   - 取込: フォルダ指定→取込進捗→完了
   - チューニング: k1/bスライダー→実行→ヒートマップ
   - 評価: メトリクス表示
