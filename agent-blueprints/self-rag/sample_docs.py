"""
Sample documents about AI and ML topics for indexing at startup.
These 20 documents seed the FAISS vector store used by the Self-RAG pipeline.
"""

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001",
        "title": "Introduction to Transformers",
        "content": (
            "The Transformer architecture, introduced in 'Attention is All You Need' (Vaswani et al., 2017), "
            "revolutionized natural language processing. Unlike RNNs, Transformers use self-attention mechanisms "
            "to process all tokens in parallel, capturing long-range dependencies efficiently. The key innovation "
            "is scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V, where Q, K, V are "
            "query, key, and value matrices. Multi-head attention runs this operation in parallel across h heads, "
            "allowing the model to attend to information from different representation subspaces."
        ),
        "category": "deep_learning",
    },
    {
        "id": "doc_002",
        "title": "Large Language Models Overview",
        "content": (
            "Large Language Models (LLMs) are neural networks trained on massive text corpora to predict the next "
            "token in a sequence. GPT-3 (175B parameters), PaLM (540B), and Claude are prominent examples. "
            "LLMs exhibit emergent capabilities at scale: few-shot learning, chain-of-thought reasoning, and "
            "instruction following. Training requires enormous compute. Inference can be optimized via quantization "
            "(INT8, INT4), distillation, and speculative decoding. Fine-tuning techniques include RLHF, LoRA, "
            "and QLoRA for adapting pre-trained models to specific tasks."
        ),
        "category": "llm",
    },
    {
        "id": "doc_003",
        "title": "Retrieval-Augmented Generation (RAG)",
        "content": (
            "RAG combines retrieval systems with generative models to ground LLM outputs in retrieved documents. "
            "The basic RAG pipeline: (1) embed the query using a dense retriever, (2) search a vector database "
            "for top-k relevant documents, (3) concatenate retrieved docs with the query as context, (4) generate "
            "an answer. Vector databases like FAISS, Pinecone, Weaviate, and Chroma store document embeddings. "
            "Advanced variants include HyDE (Hypothetical Document Embeddings), multi-hop RAG, and Self-RAG, "
            "which grades retrieved documents for relevance before generating."
        ),
        "category": "rag",
    },
    {
        "id": "doc_004",
        "title": "Self-RAG Architecture",
        "content": (
            "Self-RAG (Asai et al., 2023) is an adaptive RAG framework that trains LLMs to reflect on their "
            "own generation process. Unlike standard RAG, Self-RAG decides WHEN to retrieve (not always), "
            "grades retrieved passages for relevance (ISREL), checks if the generation is grounded in evidence "
            "(ISSUP), and evaluates if the response answers the question (ISUSE). This produces more accurate, "
            "hallucination-free outputs. The model uses special reflection tokens: [Retrieve], [ISREL], "
            "[ISSUP], [ISUSE]. Self-RAG outperforms GPT-4 on several knowledge-intensive tasks."
        ),
        "category": "rag",
    },
    {
        "id": "doc_005",
        "title": "Vector Embeddings and Similarity Search",
        "content": (
            "Vector embeddings map text to dense numerical representations in high-dimensional space. Similar "
            "texts cluster together, enabling semantic search. Popular embedding models include sentence-transformers "
            "(all-MiniLM-L6-v2, all-mpnet-base-v2), OpenAI text-embedding-3-small, and Cohere embed-english-v3.0. "
            "FAISS (Facebook AI Similarity Search) supports billion-scale ANN (Approximate Nearest Neighbor) search "
            "using IVF (Inverted File Index), HNSW (Hierarchical Navigable Small World graphs), and PQ "
            "(Product Quantization). Cosine similarity and inner product are common distance metrics."
        ),
        "category": "retrieval",
    },
    {
        "id": "doc_006",
        "title": "Reinforcement Learning from Human Feedback (RLHF)",
        "content": (
            "RLHF aligns language models with human preferences through three phases: (1) supervised fine-tuning "
            "on demonstration data, (2) training a reward model on human preference comparisons, (3) optimizing "
            "the policy with PPO (Proximal Policy Optimization) using the reward model. InstructGPT and ChatGPT "
            "used RLHF to dramatically improve helpfulness and reduce harmful outputs. Direct Preference Optimization "
            "(DPO) is a simpler alternative that skips explicit reward modeling. Constitutional AI (Anthropic) uses "
            "AI feedback instead of human feedback for scalability."
        ),
        "category": "alignment",
    },
    {
        "id": "doc_007",
        "title": "Chain-of-Thought Prompting",
        "content": (
            "Chain-of-Thought (CoT) prompting enables LLMs to solve complex reasoning tasks by generating "
            "intermediate reasoning steps. Few-shot CoT provides examples with step-by-step solutions; "
            "zero-shot CoT uses 'Let's think step by step.' CoT dramatically improves performance on arithmetic, "
            "commonsense, and symbolic reasoning. Tree-of-Thought (ToT) extends CoT by exploring multiple "
            "reasoning branches. Self-consistency samples multiple CoT paths and takes majority vote to boost "
            "accuracy. CoT is most effective for models with 100B+ parameters."
        ),
        "category": "prompting",
    },
    {
        "id": "doc_008",
        "title": "Parameter-Efficient Fine-Tuning: LoRA",
        "content": (
            "LoRA (Low-Rank Adaptation) adds trainable rank-decomposition matrices to frozen pre-trained weights, "
            "reducing trainable parameters by 10,000x while matching full fine-tuning performance. For weight "
            "matrix W, LoRA learns delta_W = BA where B and A are low-rank matrices. "
            "QLoRA combines 4-bit quantization with LoRA, enabling fine-tuning of 65B models on a single 48GB GPU. "
            "Adapter layers, prefix tuning, and prompt tuning are alternative PEFT methods. LoRA is widely used "
            "for domain adaptation and instruction tuning of open-source models like LLaMA and Mistral."
        ),
        "category": "fine_tuning",
    },
    {
        "id": "doc_009",
        "title": "Agentic AI Systems",
        "content": (
            "AI agents are systems that perceive their environment, plan actions, use tools, and execute tasks "
            "autonomously toward a goal. Key components: (1) LLM backbone for reasoning, (2) tools/APIs for "
            "taking actions, (3) memory (in-context, external, episodic), (4) planning (ReAct, Plan-and-Execute). "
            "ReAct (Reason+Act) interleaves reasoning traces with tool use. LangGraph enables stateful multi-agent "
            "workflows as directed graphs. AutoGPT, BabyAGI, and Claude Computer Use are prominent examples. "
            "Key challenges: long-horizon planning, error recovery, and reliable tool use."
        ),
        "category": "agents",
    },
    {
        "id": "doc_010",
        "title": "Neural Network Training Fundamentals",
        "content": (
            "Neural networks learn via backpropagation and gradient descent. Loss functions measure prediction "
            "error: cross-entropy for classification, MSE for regression. Optimizers: SGD with momentum, Adam "
            "(adaptive moment estimation), AdamW (Adam with weight decay), Lion. Learning rate scheduling "
            "(cosine annealing, warmup) is crucial for stable training. Batch normalization, layer normalization, "
            "dropout, and weight decay regularize networks to prevent overfitting. Gradient clipping prevents "
            "exploding gradients. Mixed precision training (FP16/BF16) reduces memory usage."
        ),
        "category": "deep_learning",
    },
    {
        "id": "doc_011",
        "title": "Convolutional Neural Networks",
        "content": (
            "CNNs use convolutional layers to extract spatial features from images through learned filters. "
            "Key operations: convolution (feature extraction), pooling (spatial downsampling), and fully connected "
            "layers (classification). Landmark architectures: AlexNet (2012), VGG, ResNet (residual connections), "
            "EfficientNet, Vision Transformer (ViT). ResNets solve vanishing gradients with skip connections: "
            "output = F(x) + x. Transfer learning from ImageNet pre-trained models is standard practice. "
            "Modern vision models like CLIP combine vision and language understanding via contrastive learning."
        ),
        "category": "computer_vision",
    },
    {
        "id": "doc_012",
        "title": "Graph Neural Networks",
        "content": (
            "Graph Neural Networks (GNNs) extend deep learning to graph-structured data. Message passing: "
            "each node aggregates features from neighbors. Key architectures: GCN (Graph Convolutional Network), "
            "GAT (Graph Attention Network), GraphSAGE, and Graph Transformers. Applications: molecular property "
            "prediction (drug discovery), social network analysis, recommendation systems, and knowledge graphs. "
            "PyTorch Geometric and DGL are popular libraries. GNNs struggle with over-smoothing (deep GNNs) "
            "and heterophily (connected nodes having different labels)."
        ),
        "category": "deep_learning",
    },
    {
        "id": "doc_013",
        "title": "Diffusion Models for Image Generation",
        "content": (
            "Diffusion models learn to reverse a gradual noising process, generating images by iteratively "
            "denoising Gaussian noise. The forward process adds noise over T timesteps; the reverse process "
            "learns to predict the noise. DDPM established the framework. Stable Diffusion uses Latent Diffusion "
            "Models (LDM) to operate in compressed latent space for efficiency. DALL-E 3, Midjourney, and Imagen "
            "are text-to-image diffusion models. Classifier-free guidance scales conditioning strength. "
            "ControlNet adds spatial control. Diffusion models outperform GANs on diversity."
        ),
        "category": "generative_ai",
    },
    {
        "id": "doc_014",
        "title": "Evaluation Metrics for LLMs",
        "content": (
            "LLM evaluation uses both automatic metrics and human evaluation. Text generation metrics: BLEU "
            "(n-gram precision), ROUGE (recall-oriented), BERTScore (semantic similarity), and perplexity. "
            "Benchmarks: MMLU (multitask language understanding), HumanEval (code generation), HellaSwag "
            "(commonsense), GSM8K (math), TruthfulQA (truthfulness). LLM-as-judge uses stronger models to "
            "evaluate outputs. MT-Bench and Chatbot Arena measure instruction following via pairwise comparisons. "
            "RAG-specific metrics: faithfulness, answer relevancy, and context precision (RAGAS framework)."
        ),
        "category": "evaluation",
    },
    {
        "id": "doc_015",
        "title": "Prompt Engineering Techniques",
        "content": (
            "Effective prompt engineering significantly improves LLM outputs. Core techniques: (1) Clear task "
            "description with specific instructions, (2) Few-shot examples demonstrating desired output format, "
            "(3) Chain-of-thought for complex reasoning, (4) Role prompting ('You are an expert...'), "
            "(5) Structured output via XML/JSON schemas, (6) Negative examples showing what NOT to do. "
            "Advanced: ReAct prompts for tool-using agents, self-consistency sampling, meta-prompting, "
            "and automatic prompt optimization (DSPy). Prompt injection and jailbreak attacks are security "
            "concerns requiring robust system prompts and output filtering."
        ),
        "category": "prompting",
    },
    {
        "id": "doc_016",
        "title": "Federated Learning and Privacy-Preserving ML",
        "content": (
            "Federated learning trains models across distributed devices without centralizing raw data. "
            "FedAvg algorithm: each client trains locally on private data, sends model updates (not data) "
            "to a central server, which aggregates updates via weighted averaging. Applications: keyboard "
            "prediction (Google Gboard), medical imaging across hospitals. Challenges: non-IID data, "
            "communication overhead, and adversarial clients. Differential privacy (DP-SGD) adds calibrated "
            "noise to gradients, providing mathematical privacy guarantees."
        ),
        "category": "privacy",
    },
    {
        "id": "doc_017",
        "title": "Mixture of Experts (MoE) Architecture",
        "content": (
            "MoE models use sparse activation: only k of N expert sub-networks process each token, reducing "
            "compute while maintaining large model capacity. A learned router (gating network) selects experts "
            "per token. Mixtral 8x7B routes to 2 of 8 experts, giving 46.7B total but only 12.9B active "
            "parameters. GPT-4 reportedly uses MoE. Switch Transformer and GShard pioneered MoE scaling. "
            "Challenges: load balancing (ensuring experts are used equally), routing instability, and "
            "communication costs in distributed training."
        ),
        "category": "architecture",
    },
    {
        "id": "doc_018",
        "title": "Multimodal AI Models",
        "content": (
            "Multimodal models process and generate multiple data types: text, images, audio, and video. "
            "GPT-4V, Claude 3, and Gemini are prominent vision-language models. Architectures commonly "
            "use a vision encoder (CLIP ViT), a projection layer, and an LLM decoder. LLaVA and InternVL "
            "are open-source alternatives. Audio models: Whisper (speech recognition), MusicGen (music). "
            "Video-language: VideoLLaMA, GPT-4o. Key capabilities: visual question answering, image captioning, "
            "chart understanding, OCR, and visual reasoning."
        ),
        "category": "multimodal",
    },
    {
        "id": "doc_019",
        "title": "AI Safety and Alignment Research",
        "content": (
            "AI safety research addresses risks from increasingly capable AI systems. Key challenges: "
            "specification gaming (Goodhart's Law), reward hacking, distributional shift, and deceptive alignment. "
            "Alignment techniques: RLHF, Constitutional AI, debate, amplification, and interpretability research. "
            "Mechanistic interpretability reverse-engineers neural networks to understand circuits and features. "
            "Scalable oversight uses AI assistance to supervise more capable AI. Red-teaming and adversarial "
            "testing identify failure modes. Anthropic's safety team leads alignment research."
        ),
        "category": "safety",
    },
    {
        "id": "doc_020",
        "title": "Quantization and Model Compression",
        "content": (
            "Model quantization reduces precision of weights and activations to decrease memory and increase "
            "inference speed. Post-training quantization (PTQ): INT8 quantization halves memory vs FP16 with "
            "minimal accuracy loss; INT4 reduces further but requires calibration. GPTQ and AWQ are popular "
            "PTQ methods for LLMs. Quantization-aware training (QAT) simulates low precision during training. "
            "bitsandbytes library enables 4-bit NF4 quantization used in QLoRA. Knowledge distillation "
            "transfers knowledge from large teacher to small student model. Pruning removes unimportant weights."
        ),
        "category": "optimization",
    },
]
