import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder

DB_DIR = "chroma_db"
THRESHOLD = 0.5

# 全局加载模型，避免每次调用都重新加载
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectorstore = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embeddings,
    collection_metadata={"hnsw:space": "cosine"} # 【新增】强制使用余弦距离
)

# 加载 Rerank 模型（首次运行会自动从 HF 镜像站下载约 1.1GB 的权重）
_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        print("正在加载 Rerank 模型（懒加载）...")
        _reranker = CrossEncoder('BAAI/bge-reranker-base', max_length=512)
    return _reranker


def get_relevant_docs(query: str, top_k: int = 3, use_rerank: bool = True, source_file: str = None) -> str:
    """检索相关文档并拼接成上下文"""
    # 构建检索的 filter 条件
    search_kwargs = {"k": 5}
    if source_file:
        search_kwargs["filter"] = {"source": source_file}

    # 【阶段一：向量粗排】先多召回一些（Top 5），保证召回率
    docs_with_score = vectorstore.similarity_search_with_score(query, **search_kwargs)

    valid_docs = []
    for doc, score in docs_with_score:
        if score <= THRESHOLD:
            valid_docs.append(doc)

    if not valid_docs:
        return ""

    # 【阶段二：Rerank精排】使用交叉编码器重新打分
    if use_rerank and len(valid_docs) > 1:
        print("🎯 正在进行 Rerank 重排...")
        reranker = get_reranker()
        # 构建 问题-文档 对
        pairs = [[query, doc.page_content] for doc in valid_docs]
        # 计算交叉匹配得分
        scores = reranker.predict(pairs)

        # 把文档和得分绑在一起，按得分降序排序
        doc_score_pairs = list(zip(valid_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # 截取最终要送给大模型的Top K
        final_docs = [pair for pair, _ in doc_score_pairs[:top_k]]
    else:
        final_docs = valid_docs[:top_k]

    return final_docs

def get_relevant_context(query: str, top_k: int = 3, use_rerank: bool = True, source_file: str = None) -> str:
    docs = get_relevant_docs(query, top_k=top_k, use_rerank=use_rerank, source_file=source_file)
    print(f"✅️ 最终命中 {len(docs)} 个块进入 Prompt")
    if not docs:
        return ""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)