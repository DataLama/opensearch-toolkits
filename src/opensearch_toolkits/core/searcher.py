from typing import Dict, Any, Optional, List, Union, Iterator, AsyncIterator
import logging
import time
from pydantic import BaseModel, Field, model_validator

from opensearchpy.helpers import scan as opensearch_scan
from opensearchpy.helpers import async_scan as opensearch_async_scan

from .client import OpenSearchConfig
from .manager import SyncOpenSearchManager, AsyncOpenSearchManager

# 먼저 SyncOpenSearchManager를 상속받는 클래스 구현
class SyncOpenSearchManagerSearcher(SyncOpenSearchManager):
    """Synchronous OpenSearch manager with search capabilities."""
    
    def search(self, 
               index: str, 
               query: Dict[str, Any], 
               size: int = 10, 
               from_: int = 0, 
               sort: Optional[List[Dict[str, Any]]] = None, 
               source: Optional[Union[List[str], bool]] = None,
               timeout: int = 1) -> Dict[str, Any]:
        """
        Search documents in an OpenSearch index.
        
        Args:
            index: Name of the index to search
            query: OpenSearch query in dictionary format
            size: Maximum number of documents to return
            from_: Starting offset for results
            sort: Optional sorting criteria
            source: Fields to include in the response (_source filtering)
            timeout: Query timeout duration
            
        Returns:
            Dict[str, Any]: Search results
        """
        if not self.index_exists(index):
            logging.warning(f"Index '{index}' does not exist")
            return {"error": "Index not found"}
        
        try:
            body = {
                "query": query,
                "size": size,
                "from": from_
            }
            
            if sort:
                body["sort"] = sort
            
            search_params = {
                "index": index,
                "body": body,
                "timeout": timeout
            }
            
            if source is not None:
                search_params["_source"] = source
                
            start_time = time.time()
            response = self.client.search(**search_params)
            end_time = time.time()
            
            result = {
                "took": response["took"],
                "total_hits": response["hits"]["total"]["value"],
                "max_score": response["hits"]["max_score"],
                "hits": response["hits"]["hits"],
                "timed_out": response["timed_out"],
                "client_took_ms": int((end_time - start_time) * 1000)
            }
            
            return result
        except Exception as e:
            logging.error(f"Error searching in index '{index}': {str(e)}")
            return {"error": str(e)}
    
    def multi_search(self, 
                    searches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform multiple search requests in a single API call.
        
        Args:
            searches: List of search request bodies with metadata
                     Each element should have 'index' and 'search_body' keys
                     
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        if not searches:
            return []
        
        try:
            body = []
            
            # Create msearch body format: header line followed by query line
            for search in searches:
                index = search.get("index")
                search_body = search.get("search_body", {})
                
                if not index:
                    raise ValueError("Each search must specify an index")
                
                # Add header
                header = {"index": index}
                if "_source" in search:
                    header["_source"] = search["_source"]
                    
                body.append(header)
                # Add query body
                body.append(search_body)
            
            response = self.client.msearch(body=body)
            
            results = []
            for i, resp in enumerate(response["responses"]):
                if "error" in resp:
                    results.append({"error": resp["error"]})
                else:
                    results.append({
                        "took": resp["took"],
                        "total_hits": resp["hits"]["total"]["value"],
                        "max_score": resp["hits"]["max_score"],
                        "hits": resp["hits"]["hits"],
                        "timed_out": resp.get("timed_out", False)
                    })
            
            return results
        except Exception as e:
            logging.error(f"Error performing multi-search: {str(e)}")
            return [{"error": str(e)}]
        
    def scan_documents(self, 
                     index: str, 
                     query: Dict[str, Any] = None, 
                     scroll: str = "5m", 
                     size: int = 1000,
                     preserve_order: bool = False,
                     source: Optional[Union[List[str], bool]] = None,
                     raise_on_error: bool = True) -> Iterator[Dict[str, Any]]:
        """
        스크롤 API를 사용하여 대량의 문서를 효율적으로 검색합니다.
        
        Args:
            index: 검색할 인덱스 이름
            query: 검색 쿼리 (기본값은 match_all)
            scroll: 스크롤 컨텍스트의 유지 시간 (예: '5m', '1h')
            size: 각 배치당 반환할 문서 수
            preserve_order: 결과 정렬 순서 유지 여부 (성능에 영향을 줄 수 있음)
            source: 결과에 포함할 필드 목록 (_source 필터링)
            raise_on_error: 검색 중 오류 발생 시 예외를 발생시킬지 여부
            
        Returns:
            Iterator[Dict[str, Any]]: 검색된 문서를 순회할 수 있는 이터레이터
            
        Yields:
            Dict[str, Any]: 각 검색 결과 문서
        """
        if not self.index_exists(index):
            logging.warning(f"Index '{index}' does not exist")
            raise ValueError(f"Index '{index}' does not exist")
        
        try:
            if query is None:
                query = {"match_all": {}}
            
            search_body = {"query": query}
            
            scan_params = {
                "client": self.client,
                "index": index,
                "query": search_body,
                "scroll": scroll,
                "size": size,
                "preserve_order": preserve_order,
                "raise_on_error": raise_on_error
            }
            
            if source is not None:
                scan_params["_source"] = source
                
            logging.info(f"Starting scan of index '{index}' with size {size}")
            return opensearch_scan(**scan_params)
            
        except Exception as e:
            logging.error(f"Error scanning documents in index '{index}': {str(e)}")
            raise
    
    def bulk_scan_process(self, 
                         index: str,
                         processor_func,
                         query: Dict[str, Any] = None,
                         scroll: str = "5m",
                         size: int = 1000,
                         source: Optional[Union[List[str], bool]] = None,
                         chunk_size: int = 100) -> Dict[str, Any]:
        """
        스캔으로 문서를 검색하고 일괄 처리하는 유틸리티 메서드입니다.
        
        Args:
            index: 검색할 인덱스 이름
            processor_func: 각 문서 청크를 처리할 콜백 함수
                           함수 시그니처: func(chunk_docs) -> None
            query: 검색 쿼리 (기본값은 match_all)
            scroll: 스크롤 컨텍스트의 유지 시간
            size: 각 배치당 반환할 문서 수
            source: 결과에 포함할 필드 목록
            chunk_size: 콜백 함수에 전달할 문서 청크의 크기
            
        Returns:
            Dict[str, Any]: 처리 결과 통계
        """
        if not callable(processor_func):
            raise ValueError("processor_func must be a callable function")
        
        try:
            start_time = time.time()
            processed_count = 0
            chunk = []
            
            # 문서 스캔 실행
            for doc in self.scan_documents(
                index=index,
                query=query,
                scroll=scroll,
                size=size,
                source=source
            ):
                chunk.append(doc)
                
                # 청크가 지정된 크기에 도달하면 처리
                if len(chunk) >= chunk_size:
                    processor_func(chunk)
                    processed_count += len(chunk)
                    chunk = []
            
            # 남은 문서가 있으면 처리
            if chunk:
                processor_func(chunk)
                processed_count += len(chunk)
            
            end_time = time.time()
            duration = end_time - start_time
            
            result = {
                "processed_documents": processed_count,
                "duration_seconds": duration,
                "documents_per_second": processed_count / duration if duration > 0 else 0,
                "index": index
            }
            
            logging.info(f"Bulk scan processing completed: {processed_count} documents in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            logging.error(f"Error in bulk scan processing: {str(e)}")
            raise


# 다음으로 AsyncOpenSearchManager를 상속받는 클래스 구현
class AsyncOpenSearchManagerSearcher(AsyncOpenSearchManager):
    """Asynchronous OpenSearch manager with search capabilities."""
    
    async def search(self, 
                   index: str, 
                   query: Dict[str, Any], 
                   size: int = 10, 
                   from_: int = 0, 
                   sort: Optional[List[Dict[str, Any]]] = None, 
                   source: Optional[Union[List[str], bool]] = None,
                   timeout: int = 1) -> Dict[str, Any]:
        """
        Search documents in an OpenSearch index asynchronously.
        
        Args:
            index: Name of the index to search
            query: OpenSearch query in dictionary format
            size: Maximum number of documents to return
            from_: Starting offset for results
            sort: Optional sorting criteria
            source: Fields to include in the response (_source filtering)
            timeout: Query timeout duration
            
        Returns:
            Dict[str, Any]: Search results
        """
        if not await self.index_exists(index):
            logging.warning(f"Index '{index}' does not exist")
            return {"error": "Index not found"}
        
        try:
            client = await self.init_client()
            
            body = {
                "query": query,
                "size": size,
                "from": from_
            }
            
            if sort:
                body["sort"] = sort
            
            search_params = {
                "index": index,
                "body": body,
                "timeout": timeout
            }
            
            if source is not None:
                search_params["_source"] = source
                
            start_time = time.time()
            response = await client.search(**search_params)
            end_time = time.time()
            
            result = {
                "took": response["took"],
                "total_hits": response["hits"]["total"]["value"],
                "max_score": response["hits"]["max_score"],
                "hits": response["hits"]["hits"],
                "timed_out": response["timed_out"],
                "client_took_ms": int((end_time - start_time) * 1000)
            }
            
            return result
        except Exception as e:
            logging.error(f"Error searching in index '{index}' asynchronously: {str(e)}")
            return {"error": str(e)}
    
    async def multi_search(self, 
                         searches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Perform multiple search requests in a single API call asynchronously.
        
        Args:
            searches: List of search request bodies with metadata
                     Each element should have 'index' and 'search_body' keys
                     
        Returns:
            List[Dict[str, Any]]: List of search results
        """
        if not searches:
            return []
        
        try:
            client = await self.init_client()
            
            body = []
            
            # Create msearch body format: header line followed by query line
            for search in searches:
                index = search.get("index")
                search_body = search.get("search_body", {})
                
                if not index:
                    raise ValueError("Each search must specify an index")
                
                # Add header
                header = {"index": index}
                if "_source" in search:
                    header["_source"] = search["_source"]
                    
                body.append(header)
                # Add query body
                body.append(search_body)
            
            response = await client.msearch(body=body)
            
            results = []
            for i, resp in enumerate(response["responses"]):
                if "error" in resp:
                    results.append({"error": resp["error"]})
                else:
                    results.append({
                        "took": resp["took"],
                        "total_hits": resp["hits"]["total"]["value"],
                        "max_score": resp["hits"]["max_score"],
                        "hits": resp["hits"]["hits"],
                        "timed_out": resp.get("timed_out", False)
                    })
            
            return results
        except Exception as e:
            logging.error(f"Error performing multi-search asynchronously: {str(e)}")
            return [{"error": str(e)}]

    async def scan_documents(self, 
                           index: str, 
                           query: Dict[str, Any] = None, 
                           scroll: str = "5m", 
                           size: int = 1000,
                           preserve_order: bool = False,
                           source: Optional[Union[List[str], bool]] = None,
                           raise_on_error: bool = True) -> AsyncIterator[Dict[str, Any]]:
        """
        스크롤 API를 사용하여 대량의 문서를 비동기적으로 효율적으로 검색합니다.
        
        Args:
            index: 검색할 인덱스 이름
            query: 검색 쿼리 (기본값은 match_all)
            scroll: 스크롤 컨텍스트의 유지 시간 (예: '5m', '1h')
            size: 각 배치당 반환할 문서 수
            preserve_order: 결과 정렬 순서 유지 여부 (성능에 영향을 줄 수 있음)
            source: 결과에 포함할 필드 목록 (_source 필터링)
            raise_on_error: 검색 중 오류 발생 시 예외를 발생시킬지 여부
            
        Returns:
            AsyncIterator[Dict[str, Any]]: 검색된 문서를 비동기적으로 순회할 수 있는 이터레이터
            
        Yields:
            Dict[str, Any]: 각 검색 결과 문서
        """
        if not await self.index_exists(index):
            logging.warning(f"Index '{index}' does not exist")
            raise ValueError(f"Index '{index}' does not exist")
        
        try:
            client = await self.init_client()
            
            if query is None:
                query = {"match_all": {}}
            
            search_body = {"query": query}
            
            scan_params = {
                "client": client,
                "index": index,
                "query": search_body,
                "scroll": scroll,
                "size": size,
                "preserve_order": preserve_order,
                "raise_on_error": raise_on_error
            }
            
            if source is not None:
                scan_params["_source"] = source
                
            logging.info(f"Starting async scan of index '{index}' with size {size}")
            return opensearch_async_scan(**scan_params)
            
        except Exception as e:
            logging.error(f"Error scanning documents in index '{index}' asynchronously: {str(e)}")
            raise
    
    async def bulk_scan_process(self, 
                              index: str,
                              processor_func,
                              query: Dict[str, Any] = None,
                              scroll: str = "5m",
                              size: int = 1000,
                              source: Optional[Union[List[str], bool]] = None,
                              chunk_size: int = 100) -> Dict[str, Any]:
        """
        스캔으로 문서를 비동기적으로 검색하고 일괄 처리하는 유틸리티 메서드입니다.
        
        Args:
            index: 검색할 인덱스 이름
            processor_func: 각 문서 청크를 처리할 비동기 콜백 함수
                           함수 시그니처: async func(chunk_docs) -> None
            query: 검색 쿼리 (기본값은 match_all)
            scroll: 스크롤 컨텍스트의 유지 시간
            size: 각 배치당 반환할 문서 수
            source: 결과에 포함할 필드 목록
            chunk_size: 콜백 함수에 전달할 문서 청크의 크기
            
        Returns:
            Dict[str, Any]: 처리 결과 통계
        """
        import inspect
        
        if not callable(processor_func):
            raise ValueError("processor_func must be a callable function")
        
        if not inspect.iscoroutinefunction(processor_func):
            raise ValueError("processor_func must be an async function (coroutine)")
        
        try:
            start_time = time.time()
            processed_count = 0
            chunk = []
            
            # 문서 스캔 실행
            async for doc in await self.scan_documents(
                index=index,
                query=query,
                scroll=scroll,
                size=size,
                source=source
            ):
                chunk.append(doc)
                
                # 청크가 지정된 크기에 도달하면 처리
                if len(chunk) >= chunk_size:
                    await processor_func(chunk)
                    processed_count += len(chunk)
                    chunk = []
            
            # 남은 문서가 있으면 처리
            if chunk:
                await processor_func(chunk)
                processed_count += len(chunk)
            
            end_time = time.time()
            duration = end_time - start_time
            
            result = {
                "processed_documents": processed_count,
                "duration_seconds": duration,
                "documents_per_second": processed_count / duration if duration > 0 else 0,
                "index": index
            }
            
            logging.info(f"Async bulk scan processing completed: {processed_count} documents in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            logging.error(f"Error in async bulk scan processing: {str(e)}")
            raise
