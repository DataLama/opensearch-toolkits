import os
import logging
import asyncio
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

from opensearch_toolkits.core import AsyncOpenSearchManagerIndexer, OpenSearchConfig

logging.basicConfig(
    level=logging.INFO,  # 디버깅을 위해 INFO 레벨로 설정
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


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


async def async_main():
    # argparse 설정
    parser = ArgumentParser(description="OpenSearch 인덱스 생성 및 문서 인덱싱 (비동기)")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config/config.yaml",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)

    # 데이터셋 로드
    ds = datasets.load_dataset(
        config.dataset_name, config.default, split=config.split
    )
    cached_embeddings = read_jsonlines(get_cached_embeddings_path(config))
    preprocessor_fn = convert_korquadv1_to_langchain_docs
    docs = preprocessor_fn(ds.to_list())
    
    # OpenSearch 인덱스 매니저 생성
    async with AsyncOpenSearchManagerIndexer(
        OpenSearchConfig(
            host=os.environ["OPENSEARCH_HOST"],  # OpenSearch 서버 주소
            port=int(os.environ["OPENSEARCH_PORT"]),  # OpenSearch 서버 포트
            user=os.environ.get("OPENSEARCH_USER"),  # OpenSearch 사용자 이름
            password=os.environ.get("OPENSEARCH_PASSWORD"),  # OpenSearch 비밀번호
            use_ssl=os.environ["OPENSEARCH_USE_SSL"].lower() == "true",  # SSL 사용 여부
        )
    ) as index_manager:
        # OpenSearch 인덱스 생성
        index_name = f"async-{config.index_name}"
        index_config = json.load(open(config.mapping_file))
        
        index_created = await index_manager.create_index_if_not_exists(
            index_name=index_name, 
            index_config=index_config
        )
        
        if not index_created:
            logging.error(f'Failed to create or confirm index {index_name}')
            return

        # 문서 인덱싱 (비동기)
        result = await index_manager.index_documents_for_langchain(
            index_name,
            docs,
            cached_embeddings,
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
        


def main():
    """비동기 메인 함수 실행을 위한 엔트리 포인트"""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()