# PhotoMind (CLIPSE-X) Architecture & Methodology Diagrams

## 1. System Architecture Overview

```mermaid
graph TB
    subgraph "Input Layer"
        A[Photo Collections<br/>ZIP/Folders] --> B[ZIP Extractor<br/>Path Validation]
        C[Natural Language<br/>Queries] --> D[Text Tokenizer]
    end
    
    subgraph "Processing Layer"
        B --> E[Image Discovery<br/>& Validation]
        E --> F[Batch Image<br/>Preprocessing]
        F --> G[CLIP Vision<br/>Encoder]
        D --> H[CLIP Text<br/>Encoder]
        
        G --> I[L2 Normalization<br/>Image Embeddings]
        H --> J[L2 Normalization<br/>Text Embeddings]
    end
    
    subgraph "Storage Layer"
        I --> K[Compressed Index<br/>embeddings.npz]
        E --> L[Metadata Store<br/>index.json]
        K --> M[(Vector Index)]
        L --> M
    end
    
    subgraph "Retrieval Layer"
        J --> N[Cosine Similarity<br/>Computation]
        M --> N
        N --> O[Top-K Ranking]
        O --> P[Search Results<br/>with Scores]
    end
    
    subgraph "Explainability Layer (Phase 3)"
        P --> Q[Visual Attention<br/>Grad-CAM/ViT]
        Q --> R[Attention Heatmaps]
        P --> S[Token Importance<br/>Analysis]
        S --> T[Natural Language<br/>Explanations]
    end
    
    style A fill:#e1f5fe
    style C fill:#e1f5fe
    style G fill:#f3e5f5
    style H fill:#f3e5f5
    style M fill:#e8f5e8
    style Q fill:#fff3e0
    style S fill:#fff3e0
```

## 2. Data Flow Architecture

```mermaid
flowchart LR
    subgraph "Phase 1: Foundation"
        A1[Raw Images] --> A2[Path Traversal<br/>Protection]
        A2 --> A3[Extension<br/>Filtering]
        A3 --> A4[Pillow.verify()<br/>Validation]
        A4 --> A5[ImageRecord<br/>Manifest]
    end
    
    subgraph "Phase 2: Embeddings & Index"
        A5 --> B1[Batch Loading<br/>RGB Conversion]
        B1 --> B2[CLIP Preprocessing<br/>Transforms]
        B2 --> B3[ViT-B/32 Vision<br/>Encoder Forward Pass]
        B3 --> B4[Feature Extraction<br/>CLS Token]
        B4 --> B5[L2 Normalization<br/>Unit Vectors]
        B5 --> B6[Compressed Storage<br/>NPZ + JSON]
    end
    
    subgraph "Phase 3: Explainability"
        B6 --> C1[Query Processing]
        C1 --> C2[Similarity Matching]
        C2 --> C3[Attention Map<br/>Generation]
        C3 --> C4[Visual Overlay<br/>Heatmaps]
        C2 --> C5[Token Analysis<br/>Importance Weights]
        C5 --> C6[Natural Language<br/>Rationale]
    end
    
    style A1 fill:#ffebee
    style B1 fill:#e3f2fd
    style C1 fill:#f1f8e9
```

## 3. Module Architecture & Dependencies

```mermaid
graph TB
    subgraph "Core Modules"
        A[io_utils.py<br/>ZIP handling, validation<br/>ImageRecord management]
        B[model.py<br/>CLIP model loading<br/>Device management]
        C[embedder.py<br/>Batch encoding<br/>L2 normalization]
        D[index_store.py<br/>Persistence layer<br/>NPZ + JSON I/O]
    end
    
    subgraph "CLI Scripts"
        E[build_index.py<br/>Index creation pipeline]
        F[search_index.py<br/>Query processing]
        G[phase1_check_zip.py<br/>Validation testing]
        H[phase2_check_query.py<br/>Search testing]
    end
    
    subgraph "Test Suite"
        I[test_io_utils.py<br/>Foundation tests]
        J[test_embed_and_index.py<br/>Integration tests]
        K[conftest.py<br/>Test configuration]
    end
    
    subgraph "Phase 3 (Planned)"
        L[explainer.py<br/>Visual attention<br/>Natural language]
        M[test_explainer.py<br/>Explainability tests]
    end
    
    subgraph "External Dependencies"
        N[open-clip-torch<br/>CLIP implementation]
        O[torch<br/>Deep learning framework]
        P[PIL/Pillow<br/>Image processing]
        Q[numpy<br/>Numerical computing]
    end
    
    A --> E
    A --> F
    B --> C
    C --> D
    B --> E
    C --> E
    D --> E
    B --> F
    D --> F
    
    B --> N
    C --> N
    C --> O
    A --> P
    C --> Q
    D --> Q
    
    A --> I
    C --> J
    D --> J
    
    L --> M
    
    style A fill:#e8f5e8
    style B fill:#e8f5e8
    style C fill:#e8f5e8
    style D fill:#e8f5e8
    style L fill:#fff3e0
    style M fill:#fff3e0
```

## 4. Phase-by-Phase Methodology

```mermaid
timeline
    title PhotoMind Development Methodology
    
    section Phase 1: Foundation ✅
        Safe ZIP Extraction    : Path traversal protection
                              : Recursive directory scanning
        Image Discovery       : Extension-based filtering
                              : Pillow validation pipeline
        Manifest Generation   : ImageRecord data structures
                              : Clean/rejected categorization
        Testing Framework     : Pytest configuration
                              : Unit test coverage
    
    section Phase 2: Core AI ✅
        CLIP Integration      : ViT-B/32 model loading
                              : Auto CPU/GPU selection
        Embedding Pipeline    : Batch image processing
                              : L2-normalized features
        Index Persistence     : Compressed NPZ storage
                              : JSON metadata format
        Search Functionality  : Cosine similarity retrieval
                              : Top-k ranking system
    
    section Phase 3: Explainability 🎯
        Visual Attention      : Grad-CAM adaptation for ViT
                              : Attention map extraction
        Query Analysis        : Token importance scoring
                              : Semantic contribution weights
        Natural Explanations  : Template-based rationale
                              : "Why?" result annotation
        UI Integration        : Overlay heatmap rendering
                              : Interactive explanation views
    
    section Phase 4: UX & Feedback 📋
        Web Interface         : Flask responsive design
                              : Real-time search results
        User Feedback         : Thumbs up/down collection
                              : Result relevance scoring
        Analytics Dashboard   : Query performance metrics
                              : Usage pattern analysis
        History & Refinement  : Search session tracking
                              : Query suggestion system
    
    section Phase 5: Production 📋
        Performance Scaling   : Optional FAISS integration
                              : Large collection support
        Incremental Updates   : New photo detection
                              : Delta embedding computation
        Deployment Package    : Docker containerization
                              : Installation documentation
        Academic Presentation : XAI course deliverables
                              : Dr. Anand M evaluation
```

## 5. Technical Implementation Flow

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as CLI Scripts
    participant IO as io_utils
    participant M as CLIP Model
    participant E as Embedder
    participant IS as IndexStore
    participant DB as Vector Index
    
    Note over U,DB: Index Building Phase
    U->>CLI: python build_index.py photos/ index_out/
    CLI->>IO: build_manifest(photos_path)
    IO->>IO: validate_images() & filter_extensions()
    IO-->>CLI: valid_records, rejected_paths
    CLI->>M: CLIPModel().load()
    M-->>CLI: loaded_clip_model
    CLI->>E: encode_images(valid_records, clip)
    E->>E: batch_process() & l2_normalize()
    E-->>CLI: normalized_embeddings
    CLI->>IS: save_index(embeddings, records, meta)
    IS->>DB: store compressed NPZ + JSON
    
    Note over U,DB: Search Phase
    U->>CLI: python search_index.py index_out/ "query" k
    CLI->>IS: load_index(index_path)
    IS->>DB: retrieve embeddings & metadata
    DB-->>CLI: image_embeddings, paths, meta
    CLI->>E: encode_text(query, clip)
    E-->>CLI: query_embedding
    CLI->>E: topk_similar(query_emb, image_embs, k)
    E-->>CLI: top_indices, similarity_scores
    CLI->>U: display results with scores
    
    Note over U,DB: Explainability Phase (Phase 3)
    U->>CLI: explain_result(image_id, query)
    CLI->>M: extract_attention_maps(image, query)
    M-->>CLI: visual_attention_weights
    CLI->>E: analyze_token_importance(query)
    E-->>CLI: token_contributions
    CLI->>U: heatmap_overlay + natural_explanation
```

## 6. Quality Assurance & Testing Strategy

```mermaid
mindmap
  root((Testing Strategy))
    Unit Tests
      io_utils validation
      Model loading/unloading
      Embedding normalization
      Index serialization
    Integration Tests
      End-to-end pipeline
      Real image processing
      Search result accuracy
      Performance benchmarks
    Quality Gates
      All tests pass (pytest -q)
      Slow tests with --runslow flag
      Code coverage metrics
      Performance regression checks
    Validation Data
      Known corrupt images
      Various image formats
      Edge case queries
      Large collection stress tests
    Academic Standards
      Reproducible results
      Documented methodology
      Transparent AI decisions
      XAI course requirements
```

## 7. Deployment & Scaling Architecture

```mermaid
C4Context
    title PhotoMind Deployment Context
    
    Person(user, "End User", "Searches personal photo collections using natural language")
    Person(researcher, "AI Researcher", "Analyzes explainable AI performance and accuracy")
    
    System_Boundary(photomind, "PhotoMind System") {
        System(webapp, "Web Interface", "Flask-based responsive UI for search and explanation")
        System(api, "Search API", "RESTful endpoints for embedding and retrieval")
        System(indexer, "Index Builder", "CLI tools for processing photo collections")
        System(explainer, "Explanation Engine", "Visual attention and natural language rationale")
    }
    
    System_Ext(clip, "CLIP Model", "OpenAI's pretrained vision-language model")
    System_Ext(storage, "Vector Storage", "NPZ compressed embeddings + JSON metadata")
    System_Ext(faiss, "FAISS Index", "Optional: High-performance similarity search for large collections")
    
    Rel(user, webapp, "Searches photos")
    Rel(researcher, api, "Evaluates explanations")
    Rel(webapp, api, "Query processing")
    Rel(api, explainer, "Generate explanations")
    Rel(indexer, storage, "Store embeddings")
    Rel(api, storage, "Retrieve vectors")
    Rel(api, clip, "Encode queries")
    Rel(explainer, clip, "Extract attention")
    Rel(api, faiss, "Scale to large collections")
```

---

## Usage Instructions

1. **Copy any diagram section** and paste into Mermaid Live Editor: https://mermaid.live/
2. **Customize styling** by modifying the `style` directives at the end of each diagram
3. **Export formats**: PNG, SVG, PDF available from Mermaid Live Editor
4. **Integration**: These diagrams work in GitHub README, GitLab docs, and most markdown renderers

## Academic Presentation Tips

- Use **Architecture Overview** for system design explanation
- Use **Phase Methodology** timeline for project progression
- Use **Technical Flow** sequence diagram for implementation details
- Use **Testing Strategy** mindmap for quality assurance discussion
- Use **Deployment Context** for scalability and real-world application

Each diagram emphasizes the **explainable AI** aspects that Dr. Anand M expects for the XAI course evaluation.