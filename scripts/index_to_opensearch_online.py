import os
import logging
from typing import List, Dict, Any, Union
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from pathlib import Path
from langchain_core.documents import Document
from tqdm.auto import tqdm
import orjson
from uuid_extensions import uuid7
import json
import datasets
import numpy as np
from omegaconf import OmegaConf
from argparse import ArgumentParser

import openai
from langchain_openai import OpenAIEmbeddings

from opensearch_toolkits.core import SyncOpenSearchManagerIndexer, OpenSearchConfig

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

EMBEDDING_API_URL = "http://192.168.219.104:7997"


def read_jsonlines(path: Union[str, Path]) -> List[Dict]:
    """Load batch data from jsonline files."""
    if isinstance(path, str):
        path = Path(path)

    with path.open('r') as f:
        data: List[Dict] = [orjson.loads(line) for line in f]
    return data

def convert_korquadv1_to_langchain_docs(dict_records: List[Dict[str, Any]]) -> List[Document]:
    """
    파이썬 딕셔너리 형태의 데이터셋 레코드를 Langchain Document 객체 리스트로 변환합니다.
    
    Args:
        dict_records: 딕셔너리 형태의 레코드 리스트
        
    Returns:
        List[Document]: Langchain Document 객체 리스트
    """
    documents = []
    
    for record in tqdm(dict_records, desc="Converting records to documents"):
        # description을 page_content로 사용
        question = record.get('question', '')

        if (question != "nan") and question:
            page_content = question
        
        # 메타데이터 구성
        metadata = {k: v for k, v in record.items() if k in ['title']}
        
        # Document 객체 생성 및 리스트에 추가
        doc = Document(page_content=page_content, metadata=metadata, id=str(uuid7()))
        documents.append(doc)
    
    return documents

def get_cached_embeddings_path(config):
    dataset_name = config.dataset_name.replace("/", "_")
    embedding_model_name = config.model_name.tokenizer.replace("/", "_")
    cache_name = f"{dataset_name}+{config.default}+{config.split}+{embedding_model_name}_embeddings.jsonl"
    return f"{config.cache_dir}/{cache_name}"

def main():
    # argparse 설정
    parser = ArgumentParser(description="OpenSearch 인덱스 생성 및 문서 인덱싱")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/config.yaml",
    )
    args = parser.parse_args()
    
    # load config
    config = OmegaConf.load(args.config_path)
    index_manager = SyncOpenSearchManagerIndexer(
        OpenSearchConfig(
            host=os.environ["OPENSEARCH_HOST"],  # OpenSearch 서버 주소
            port=int(os.environ["OPENSEARCH_PORT"]),  # OpenSearch 서버 포트
            user=os.environ.get("OPENSEARCH_USER"),  # OpenSearch 사용자 이름
            password=os.environ.get("OPENSEARCH_PASSWORD"),  # OpenSearch 비밀번호
            use_ssl=os.environ["OPENSEARCH_USE_SSL"].lower() == "true",  # SSL 사용 여부
        )
    )
    preprocessor_fn = convert_korquadv1_to_langchain_docs

    # OpenSearch 인덱스 생성
    with open(config.mapping_file) as f:
        index_config = json.load(f)
    index_manager.create_index_if_not_exists(index_name=config.index_name, index_config=index_config)

    # 데이터셋 로드
    ds = datasets.load_dataset(
        config.dataset_name, config.default, split=config.split
            ).select(range(0, 1000))
    # cached_embeddings = read_jsonlines(get_cached_embeddings_path(config))

    docs = preprocessor_fn(ds.to_list())

    # embedding.
    client = openai.OpenAI(base_url=EMBEDDING_API_URL)

    def process_batch(batch_texts: List[str]) -> List[List[float]]:
        """단일 배치에 대한 임베딩을 처리합니다."""
        try:
            embedding_outputs = client.embeddings.create(
                model='kure-v1',
                input=batch_texts,
            )
            return [encoded_embedding.embedding for encoded_embedding in embedding_outputs.data]
        except Exception as e:
            print(f"Error processing batch: {e}")
            # 에러가 발생하면 재시도 로직을 구현할 수 있습니다
            # 이 예제에서는 간단히 빈 리스트 반환
            return [[] for _ in range(len(batch_texts))]

    # 문서 인덱싱
    result = index_manager.index_documents_for_langchain(
        config.index_name,
        docs,
        embedding_fn=process_batch,
        text_field="text",
        embedding_field="vector_field",
        id_field="id",
        refresh=True,
        stats_only=False
    )

    if result['failed'] > 0:
        print(f"인덱싱 실패한 문서 케이스: {result['failed_items'][0]}")
    else:
        print(f"인덱싱 결과: {result}")

if __name__ == "__main__":
    main()