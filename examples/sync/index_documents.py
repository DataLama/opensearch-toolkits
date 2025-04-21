"""
OpenSearch Document Indexing Example
===================================

이 예제는 Corpus-42 데이터셋을 불러와 OpenSearch에 인덱싱하는 방법을 보여줍니다.

필요한 라이브러리:
- opensearch_toolkits (OpenSearch 연결 및 관리)
- pydantic (데이터 검증)
- python-dotenv (환경 변수 관리)
- datasets (Hugging Face 데이터셋 접근)
- langchain_core (Document 객체 사용)
- uuid_extensions (UUID 생성)
- tqdm (진행 상황 시각화)

시작하기 전에 .env 파일에 다음 환경 변수를 설정해야 합니다:
```
OPENSEARCH_HOST=your-opensearch-host
OPENSEARCH_PORT=9200
OPENSEARCH_USER=your-username
OPENSEARCH_PASSWORD=your-password
OPENSEARCH_USE_SSL=true
```
"""

import os
import json
from typing import Dict, List, Any
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv
from opensearch_toolkits.core import OpenSearchConfig, SyncOpenSearchManagerIndexer
from langchain_core.documents import Document
from uuid_extensions import uuid7  # time-based UUID
from tqdm.auto import tqdm
import datasets


def main():
    # 1. 환경 변수 로드 및 OpenSearch 연결 설정
    load_dotenv(find_dotenv())  # .env 파일에서 환경 변수 로드
    
    config = OpenSearchConfig(
        host=os.environ["OPENSEARCH_HOST"],  # OpenSearch 서버 주소
        port=int(os.environ["OPENSEARCH_PORT"]),  # OpenSearch 서버 포트
        user=os.environ["OPENSEARCH_USER"],  # OpenSearch 사용자 이름
        password=os.environ["OPENSEARCH_PASSWORD"],  # OpenSearch 비밀번호
        use_ssl=os.environ["OPENSEARCH_USE_SSL"].lower() == "true",  # SSL 사용 여부
    )
    
    osm = SyncOpenSearchManagerIndexer(config)
    index_name = "test-corpus42"
    
    # 2. 데이터셋 로드
    print("데이터셋 로드 중...")
    ds = datasets.load_dataset(
        "datalama/corpus-42", 
        "default", 
        split="BIZ_Overall_240715164857"
    )
    
    # 3. 데이터를 LangChain Document 형식으로 변환
    print("Document 형식으로 변환 중...")
    documents = convert_corpus42_to_langchain_docs(ds)
    
    # 4. OpenSearch 인덱싱을 위한 요청 형식으로 변환
    print("인덱싱 요청 형식으로 변환 중...")
    requests = prepare_index_requests(documents)
    
    # 5. 인덱스 생성 및 문서 인덱싱
    print("인덱스 매핑 로드 중...")
    with open("corpus42_simple.json") as f:
        mapping = json.load(f)
    
    print(f"'{index_name}' 인덱스 삭제 및 재생성 중...")
    osm.delete_index(index_name)
    osm.create_index_if_not_exists(index_name, index_config=mapping)
    
    print("문서 인덱싱 중...")
    osm.index_documents(
        index_name=index_name,
        documents=requests,
        id_field='id',
        refresh=True
    )
    
    print(f"인덱싱 완료: {len(documents)}개 문서가 '{index_name}' 인덱스에 저장되었습니다.")


def convert_corpus42_to_langchain_docs(dict_records: List[Dict[str, Any]]) -> List[Document]:
    """
    파이썬 딕셔너리 형태의 데이터셋 레코드를 Langchain Document 객체 리스트로 변환합니다.
    
    Args:
        dict_records: 딕셔너리 형태의 레코드 리스트
        
    Returns:
        List[Document]: Langchain Document 객체 리스트
    """
    documents = []
    
    for record in tqdm(dict_records, desc="Converting records to documents"):
        # 'nan' 값을 None으로 변환
        record = {k:v if v != 'nan' else None for k, v in record.items()}
        
        # page_content 구성
        doc_name = record.get('doc_name', '')
        title = record.get('title', '')
        description = record.get('description', '')
        
        texts = []
        if doc_name and doc_name != "nan":
            texts.append(doc_name)
        
        if title and title != "nan":
            texts.append(title)
        
        if description and description != "nan":
            texts.append(description)
        
        page_content = "\n\n".join(texts)
        
        # 메타데이터 구성 (page_content에 사용된 필드 제외)
        metadata = {k: v for k, v in record.items() if k not in ['description', 'doc_name', 'title']}
        
        # Document 객체 생성 및 리스트에 추가
        doc = Document(page_content=page_content, metadata=metadata, id=str(uuid7()))
        documents.append(doc)
    
    return documents


def prepare_index_requests(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    LangChain Document 객체 리스트를 OpenSearch 인덱싱 요청 형식으로 변환합니다.
    
    Args:
        documents: LangChain Document 객체 리스트
        
    Returns:
        List[Dict[str, Any]]: OpenSearch 인덱싱 요청 리스트
    """
    requests = []
    
    for doc in documents:
        request = doc.model_dump()
        request['text'] = request.pop("page_content")
        requests.append(request)
    
    return requests


if __name__ == "__main__":
    main()