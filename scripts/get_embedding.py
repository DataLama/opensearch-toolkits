import os
from typing import List, Dict, Union, Tuple, Optional
from pathlib import Path
import orjson
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import asyncio
import openai
import numpy as np
import argparse
import transformers
import datasets
from omegaconf import OmegaConf
from tqdm.asyncio import tqdm_asyncio

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# EMBEDDING_API_URL = "http://192.168.219.104:7997"
EMBEDDING_API_URL = "http://127.0.0.1:7997"

## TODO: max_seq_len을 넘는 경우 버리는게 아니라 split하는 로직 추가하자.

def write_jsonlines(path: Union[str, Path], docs: List[Dict], mode: str = 'ab') -> None:
    """Load batch data from jsonline files."""
    if isinstance(path, str):
        path = Path(path)

    path.unlink(missing_ok=True)

    with path.open(mode) as f:
        for doc in docs:
            f.write(orjson.dumps(doc, option=orjson.OPT_APPEND_NEWLINE))


async def generate_embeddings_async(docs: List[Dict[str, str]], 
                                    model_name: str = "BAAI/bge-m3", 
                                    batch_size: int = 4,
                                    max_concurrent_batches: int = 4) -> List[List[float]]:
    """
    텍스트에 대한 임베딩 벡터를 비동기적으로 생성합니다.
    
    Args:
        docs: 임베딩할 문서 목록, 각 문서는 'text' 키를 포함하는 딕셔너리
        model_name: 사용할 임베딩 모델 이름
        batch_size: 각 배치의 최대 문서 수
        max_concurrent_batches: 동시에 처리할 최대 배치 수
        
    Returns:
        생성된 임베딩 벡터 목록
    """
    texts = [doc['text'] for doc in docs]
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    # 세마포어를 사용하여 동시 요청 수 제한
    semaphore = asyncio.Semaphore(max_concurrent_batches)
    client = openai.AsyncOpenAI(base_url=EMBEDDING_API_URL)
    
    async def process_batch(batch_texts: List[str]) -> List[List[float]]:
        """단일 배치에 대한 임베딩을 처리합니다."""
        async with semaphore:
            try:
                embedding_outputs = await client.embeddings.create(
                    model=model_name,
                    input=batch_texts,
                )
                return [encoded_embedding.embedding for encoded_embedding in embedding_outputs.data]
            except Exception as e:
                print(f"Error processing batch: {e}")
                # 에러가 발생하면 재시도 로직을 구현할 수 있습니다
                # 이 예제에서는 간단히 빈 리스트 반환
                return [[] for _ in range(len(batch_texts))]
    
    # 배치 작업 준비
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    
    # 모든 배치를 동시에 처리
    results = await tqdm_asyncio.gather(
        *[process_batch(batch) for batch in batches],
        desc="Processing embedding batches"
    )
    
    # 결과 병합
    full_embeddings = []
    for batch_result in results:
        full_embeddings.extend(batch_result)
    
    return full_embeddings

# 동기 래퍼 함수
def generate_embeddings(docs: List[Dict[str, str]], model_name: str = "BAAI/bge-m3") -> List[List[float]]:
    """
    텍스트에 대한 임베딩 벡터를 생성하는 동기 래퍼 함수.
    내부적으로는 비동기 처리를 사용합니다.
    """
    return asyncio.run(generate_embeddings_async(docs, model_name))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Get embeddings from a model')
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    ds = datasets.load_dataset(config.dataset_name, config.default, split=config.split)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name.tokenizer)
    print(f"Model max length: {tokenizer.model_max_length}")

    # 필터링된 텍스트 목록 생성
    filtered_docs = []
    for i in range(len(ds)):
        # text = ds[i]['description']
        # text = f"{ds[i]['doc_name']}\n\n{ds[i]['title']}\n\n{ds[i]['description']}"
        text = f"{ds[i]["question"]}"
        length = tokenizer(text, return_length=True)['length'][0]
        if length > tokenizer.model_max_length:
            continue
        filtered_docs.append({"_index": i, "text": text})

    dataset_name_str = config.dataset_name.replace("/", "_")
    model_name_str = config.model_name.tokenizer.replace("/", "_")
    os.makedirs(config.cache_dir, exist_ok=True)
    output_path = os.path.join(config.cache_dir, f"{dataset_name_str}+{config.default}+{config.split}+{model_name_str}_embeddings.jsonl")

    # 임베딩 생성
    print(f"Generating embeddings for {len(filtered_docs)} texts...")
    embeddings = generate_embeddings(filtered_docs, config.model_name.api)
    print(f"Generated {len(embeddings)} embeddings.")
    # 결과를 JSONL 파일로 저장
    write_jsonlines(output_path, [{"_index": doc["_index"], "vector_field": embedding} for doc, embedding in zip(filtered_docs, embeddings)])