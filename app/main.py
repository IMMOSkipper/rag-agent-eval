import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from app.generate import ask_rag # 导入写好的问答函数
from app.ingest import async_ingest_document
from app.celery_app import celery_app
from app.agent import async_run_agent
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import uuid
import aiofiles
import hashlib
import redis
from app.retrieve import vectorstore

load_dotenv()
app = FastAPI(title="RAG Eval Pipeline API")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 定义请求体格式
class QueryRequest(BaseModel):
    query: str
    model_name: str = os.getenv("DEFAULT_MODEL", "deepseek-v4.1-flash-1017") # 允许通过接口指定模型

class AgentRequest(BaseModel):
    query: str
    # 【修改】去掉默认值，强制要求传入，增加长度限制（1 到 64 个字符），且不可为空白字符
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="用户会话唯一标识，不可为空"
    )

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".md", ".txt"}
def safe_doc_name(raw_name: str) -> str:
    name = os.path.basename((raw_name or "").replace("\\", "/")).strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="文件名无效")
    if os.path.splitext(name)[1].lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="只接收Markdown文件或TXT文件")
    if len(name) > 100:
        raise HTTPException(status_code=400, detail="文件名过长")
    return name

    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat")
def chat(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result, context = ask_rag(request.query, model_name=request.model_name)
        return {
            "code": 200,
            "data": {
                "query": request.query,
                "answer": result,
                "context": context
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并异步解析入库"""
    filename = safe_doc_name(file.filename or "")
    await asyncio.to_thread(os.makedirs, "data/docs", exist_ok=True)
    file_id = uuid.uuid4().hex[:8]
    tmp_path = os.path.join("data/docs", f".{file_id}.uploading")
    final_path = os.path.join("data/docs", f"{file_id}_{filename}")
    

    hasher = hashlib.md5()
    total = 0
    file_hash = ""
    try:
        async with aiofiles.open(tmp_path, "wb") as out_file:
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=400, detail="文件超过 10MB")
                hasher.update(block)
                await out_file.write(block)
        if total == 0:
            raise HTTPException(status_code=400, detail="文件是空的")
        file_hash = hasher.hexdigest()
        if not r.set(f"upload:{file_hash}", 1, nx=True, ex=3600):
            raise HTTPException(status_code=400, detail="该文件正在上传，请勿重复上传")
        if vectorstore.get(where={"file_hash": file_hash}, include=[], limit=1)["ids"]:
            raise HTTPException(status_code=400, detail="该文件已经存在于知识库中")
        os.replace(tmp_path, final_path)
        # 丢给 Celery 异步处理
        task = async_ingest_document.delay(final_path, file_hash)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if os.path.exists(final_path):
            os.remove(final_path)
        raise

    return {
        "code": 200,
        "message": "文件已上传，正在后台解析入库",
        "task_id": task.id,
        "filename": os.path.basename(final_path)
    }

@app.get("/api/task/{task_id}")
def get_task_status(task_id: str):
    """查询异步任务状态"""
    task = celery_app.AsyncResult(task_id)
    if task.state == 'PENDING':
        return {"status": "PENDING", "message": "任务等待中或未找到"}
    elif task.state == 'SUCCESS':
        return {"status": "SUCCESS", "result": task.result}
    elif task.state == 'FAILURE':
        return {"status": "FAILURE", "error": str(task.info)}
    else:
        return {"status": task.state}

# 发起 Agent 任务
@app.post("/api/agent")
def run_agent_endpoint(request: AgentRequest):
    # 【新增】防御性判空
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id 不能为空白字符")

    task = async_run_agent.delay(request.query, request.session_id)
    return {"code": 200, "task_id": task.id}

# SSE 实时推送状态流
@app.get("/api/agent/stream/{task_id}")
async def stream_agent_status(task_id: str):
    async def event_generator():
        last_meta = None
        while True:
            # 用 asyncio.to_thread 把同步阻塞的 Celery 查询放到后台线程
            # 这样即使有 1000 个并发，也不会卡死 FastAPI 的主线程
            task = await asyncio.to_thread(celery_app.AsyncResult, task_id)
            state = await asyncio.to_thread(lambda: task.state)

            # 如果状态变了，推送给前端
            # 【核心修改】判断具体的 info（内容）是否变化，而不是只看 state（大状态）
            if state == 'PROGRESS':
                info = await asyncio.to_thread(lambda: task.info)
                if info != last_meta:
                    last_meta = info
                    yield f"data: {json.dumps(info, ensure_ascii=False)}\n\n"
            elif state == 'SUCCESS':
                result = await asyncio.to_thread(lambda: task.result)
                yield f"data: {json.dumps(result, ensure_ascii=False)}\n\n"
                break
            elif state == 'FAILURE':
                yield f"data: {json.dumps({'status': 'failed'}, ensure_ascii=False)}\n\n"
                break

            await asyncio.sleep(1) # 每秒轮询一次 Redis 状态

    return StreamingResponse(event_generator(), media_type="text/event-stream")