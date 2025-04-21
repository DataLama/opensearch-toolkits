import os
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from opensearchpy.helpers import bulk, async_bulk
from opensearchpy import NotFoundError, RequestError
from tqdm import tqdm
from langchain_core.documents import Document

from .client import OpenSearchConfig
from .manager import SyncOpenSearchManager, AsyncOpenSearchManager


class SyncOpenSearchManagerIndexer(SyncOpenSearchManager):
    """Extended class for synchronous OpenSearch operations with indexing capabilities."""
    
    def __init__(self, config: OpenSearchConfig):
        """Initialize the synchronous OpenSearch manager with configuration."""
        super().__init__(config)
    
    def create_index_if_not_exists(
        self, 
        index_name: str, 
        index_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create an index if it doesn't already exist.
        
        Args:
            index_name: Name of the index to create
            settings: Index settings
            index_config: Index configuration (settings and mappings)
            
        Returns:
            bool: True if created or already exists, False on error
        """
        if self.index_exists(index_name):
            logging.info(f"Index '{index_name}' already exists")
            return True
        
        if "mappings" not in index_config or "settings" not in index_config:
            logging.error(f"Invalid index configuration for '{index_name}'")
            return False
        
        try:
            self.client.indices.create(index=index_name, body=index_config)
            logging.info(f"Successfully created index '{index_name}'")
            return True
        except Exception as e:
            logging.error(f"Error creating index '{index_name}': {str(e)}")
            return False
    
    def delete_index(self, index_name: str) -> bool:
        """
        Delete an index.
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return True  # Consider this a success since the end state is as expected
        
        try:
            self.client.indices.delete(index=index_name)
            logging.info(f"Successfully deleted index '{index_name}'")
            return True
        except Exception as e:
            logging.error(f"Error deleting index '{index_name}': {str(e)}")
            return False
    
    def index_document(
        self,
        index_name: str,
        document: Dict[str, Any],
        doc_id: Optional[str] = None,
        refresh: bool = False
    ) -> bool:
        """
        Index a single document.
        
        Args:
            index_name: Name of the index
            document: Document data to index
            doc_id: Optional document ID (will be auto-generated if not provided)
            refresh: Whether to refresh the index immediately
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return False
        
        try:
            if doc_id:
                self.client.index(
                    index=index_name,
                    body=document,
                    id=doc_id,
                    refresh=refresh
                )
            else:
                self.client.index(
                    index=index_name,
                    body=document,
                    refresh=refresh
                )
            return True
        except Exception as e:
            logging.error(f"Error indexing document to '{index_name}': {str(e)}")
            return False
    
    def index_documents(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        id_field: Optional[str] = None,
        chunk_size: int = 500,
        refresh: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, int]:
        """
        Bulk index multiple documents.
        
        Args:
            index_name: Name of the index
            documents: List of documents to index
            id_field: Field to use as document ID
            chunk_size: Number of documents to index in each batch
            refresh: Whether to refresh the index after each batch
            progress_bar: Whether to show a progress bar
            
        Returns:
            Dict[str, int]: Stats about the indexing operation
        """
        if not self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return {"failed": len(documents), "succeeded": 0, "total": len(documents)}
        
        # Prepare actions for bulk indexing
        actions = []
        for doc in documents:

            action = {"_index": index_name}
            
            # Use specified field as ID if provided
            if id_field and id_field in doc:
                _id = doc.pop(id_field)
                action["_id"] = _id

            action.update(doc)
            
            
            actions.append(action)
        
        # Set up progress bar if requested
        if progress_bar:
            pbar = tqdm(total=len(actions), desc=f"Indexing to {index_name}")
        
        # Index in chunks
        stats = {"succeeded": 0, "failed": 0}
        
        for i in range(0, len(actions), chunk_size):
            chunk = actions[i:i + chunk_size]
            try:
                success, failed_items = bulk(
                    self.client,
                    chunk,
                    chunk_size=chunk_size,
                    refresh=refresh,
                    raise_on_error=False
                )
                stats["succeeded"] += success
                stats["failed"] += len(failed_items)
                
                if progress_bar:
                    pbar.update(len(chunk))
                    
            except Exception as e:
                logging.error(f"Error during bulk indexing: {str(e)}")
                stats["failed"] += len(chunk)
                if progress_bar:
                    pbar.update(len(chunk))
        
        if progress_bar:
            pbar.close()
        
        stats["total"] = len(documents)
        return stats
    
    def index_documents_with_embeddings(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        embedding_field: str,
        embedding_fn: Callable[[str], List[float]],
        text_field: str = "text",
        id_field: Optional[str] = None,
        chunk_size: int = 100,
        refresh: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, int]:
        """
        Index documents with real-time embedding generation.
        
        Args:
            index_name: Name of the vector index
            documents: List of documents to index
            embedding_field: Field name to store embeddings
            embedding_fn: Function that takes text and returns an embedding vector
            text_field: Field containing text to embed
            id_field: Field to use as document ID
            chunk_size: Number of documents to process in each batch
            refresh: Whether to refresh the index after each batch
            progress_bar: Whether to show a progress bar
            
        Returns:
            Dict[str, int]: Stats about the indexing operation
        """
        if not self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return {"failed": len(documents), "succeeded": 0, "total": len(documents)}
        
        # Process documents in batches to generate embeddings
        processed_docs = []
        
        # Set up progress bar if requested
        if progress_bar:
            pbar = tqdm(total=len(documents), desc=f"Generating embeddings")
        
        for i in range(0, len(documents), chunk_size):
            batch = documents[i:i + chunk_size]
            
            for doc in batch:
                if text_field not in doc:
                    logging.warning(f"Document missing text field '{text_field}', skipping")
                    continue
                
                try:
                    # Generate embedding for the document
                    embedding = embedding_fn(doc[text_field])
                    doc_copy = doc.copy()
                    doc_copy[embedding_field] = embedding
                    processed_docs.append(doc_copy)
                except Exception as e:
                    logging.error(f"Error generating embedding: {str(e)}")
            
            if progress_bar:
                pbar.update(len(batch))
        
        if progress_bar:
            pbar.close()
        
        # Index the processed documents with embeddings
        return self.index_documents(
            index_name=index_name,
            documents=processed_docs,
            id_field=id_field,
            chunk_size=chunk_size,
            refresh=refresh,
            progress_bar=progress_bar
        )
    
    def index_documents_with_precomputed_embeddings(
        self,
        index_name: str,
        docs:List[Document],
        embeddings:List[Dict[str, Any]],
        text_field: str = "text",
        embedding_field: str = "embedding",
        id_field: Optional[str] = None,
        chunk_size: int = 500,
        refresh: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, int]:
        """
        Index documents with precomputed embeddings.
        
        Args:
            index_name: Name of the vector index
            docs: List of Document objects to index
            embeddings: List of precomputed embeddings
            text_field: Field name containing text
            embedding_field: Field name containing embeddings
            id_field: Field to use as document ID
            chunk_size: Number of documents to index in each batch
            refresh: Whether to refresh the index after each batch
            progress_bar: Whether to show a progress bar
            
        Returns:
            Dict[str, int]: Stats about the indexing operation
        """
        # Preprocess documents and embeddings
        documents = self._filter_and_merge(docs, embeddings, text_field, embedding_field)
        
        # Index the documents with precomputed embeddings
        return self._index_documents_with_precomputed_embeddings(
            index_name=index_name,
            documents=documents,
            embedding_field=embedding_field,
            id_field=id_field,
            chunk_size=chunk_size,
            refresh=refresh,
            progress_bar=progress_bar
        )
    
    def _filter_and_merge(
        self,
        docs:List[Document],
        embeddings:List[Dict[str, Any]],
        text_field: str = "text",
        embedding_field:str = "vector_field",
    ) -> List[Dict[str, Any]]:
        embedding_indices = set([item["_index"] for item in embeddings])
        filtered_docs = [doc for i, doc in enumerate(docs) if i in embedding_indices]
        
        documents = []
        for i, doc in enumerate(filtered_docs):
            document = doc.model_dump()
            document[text_field] = document.pop("page_content")
            document[embedding_field] = embeddings[i]
            documents.append(document)

        return documents
    
    def _index_documents_with_precomputed_embeddings(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        embedding_field: str = "vector_field",
        id_field: Optional[str] = None,
        chunk_size: int = 500,
        refresh: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, int]:
        """
        Index documents with precomputed embeddings.
        
        Args:
            index_name: Name of the vector index
            documents: List of documents with embeddings already included
            embedding_field: Field name containing embeddings
            id_field: Field to use as document ID
            chunk_size: Number of documents to index in each batch
            refresh: Whether to refresh the index after each batch
            progress_bar: Whether to show a progress bar
            
        Returns:
            Dict[str, int]: Stats about the indexing operation
        """
        # Validate that documents have embeddings
        invalid_docs = []
        valid_docs = []
        
        # Index the valid documents
        return self.index_documents(
            index_name=index_name,
            documents=valid_docs,
            id_field=id_field,
            chunk_size=chunk_size,
            refresh=refresh,
            progress_bar=progress_bar
        )
    
    def reindex(
        self,
        source_index: str,
        target_index: str,
        query: Optional[Dict[str, Any]] = None,
        script: Optional[Dict[str, Any]] = None,
        wait_for_completion: bool = True,
        refresh: bool = True,
        timeout: str = "1m"
    ) -> Dict[str, Any]:
        """
        Reindex documents from one index to another.
        
        Args:
            source_index: Source index name
            target_index: Target index name
            query: Optional query to filter documents
            script: Optional script to transform documents during reindexing
            wait_for_completion: Wait for the operation to complete
            refresh: Refresh the target index after reindexing
            timeout: Timeout for the operation
            
        Returns:
            Dict[str, Any]: Result of the reindex operation
        """
        if not self.index_exists(source_index):
            error_msg = f"Source index '{source_index}' does not exist"
            logging.error(error_msg)
            return {"error": error_msg}
        
        # Create the reindex body
        body = {
            "source": {
                "index": source_index
            },
            "dest": {
                "index": target_index
            }
        }
        
        # Add query if provided
        if query:
            body["source"]["query"] = query
        
        # Add script if provided
        if script:
            body["script"] = script
        
        try:
            result = self.client.reindex(
                body=body,
                refresh=refresh,
                wait_for_completion=wait_for_completion,
                timeout=timeout
            )
            return result
        except Exception as e:
            error_msg = f"Error during reindex operation: {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg}


class AsyncOpenSearchManagerIndexer(AsyncOpenSearchManager):
    """Extended class for asynchronous OpenSearch operations with indexing capabilities."""
    
    def __init__(self, config: OpenSearchConfig):
        """Initialize the asynchronous OpenSearch manager with configuration."""
        super().__init__(config)
    
    async def create_index_if_not_exists(
        self,
        index_name: str,
        index_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create an index if it doesn't already exist (async).
        
        Args:
            index_name: Name of the index to create
            settings: Index settings
            index_config: Index configuration (settings and mappings)
            
        Returns:
            bool: True if created or already exists, False on error
        """
        if await self.index_exists(index_name):
            logging.info(f"Index '{index_name}' already exists")
            return True
        
        if "mappings" not in index_config or "settings" not in index_config:
            logging.error(f"Invalid index configuration for '{index_name}'")
            return False
        
        try:
            client = await self.init_client()
            await client.indices.create(index=index_name, body=index_config)
            logging.info(f"Successfully created index '{index_name}' asynchronously")
            return True
        except Exception as e:
            logging.error(f"Error creating index '{index_name}' asynchronously: {str(e)}")
            return False
    
    async def delete_index(self, index_name: str) -> bool:
        """
        Delete an index (async).
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not await self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return True  # Consider this a success since the end state is as expected
        
        try:
            client = await self.init_client()
            await client.indices.delete(index=index_name)
            logging.info(f"Successfully deleted index '{index_name}' asynchronously")
            return True
        except Exception as e:
            logging.error(f"Error deleting index '{index_name}' asynchronously: {str(e)}")
            return False
    
    async def index_document(
        self,
        index_name: str,
        document: Dict[str, Any],
        doc_id: Optional[str] = None,
        refresh: bool = False
    ) -> bool:
        """
        Index a single document (async).
        
        Args:
            index_name: Name of the index
            document: Document data to index
            doc_id: Optional document ID (will be auto-generated if not provided)
            refresh: Whether to refresh the index immediately
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not await self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return False
        
        try:
            client = await self.init_client()
            if doc_id:
                await client.index(
                    index=index_name,
                    body=document,
                    id=doc_id,
                    refresh=refresh
                )
            else:
                await client.index(
                    index=index_name,
                    body=document,
                    refresh=refresh
                )
            return True
        except Exception as e:
            logging.error(f"Error indexing document to '{index_name}' asynchronously: {str(e)}")
            return False
    
    async def index_documents(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        id_field: Optional[str] = None,
        chunk_size: int = 500,
        refresh: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, int]:
        """
        Bulk index multiple documents (async).
        
        Args:
            index_name: Name of the index
            documents: List of documents to index
            id_field: Field to use as document ID
            chunk_size: Number of documents to index in each batch
            refresh: Whether to refresh the index after each batch
            progress_bar: Whether to show a progress bar
            
        Returns:
            Dict[str, int]: Stats about the indexing operation
        """
        if not await self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return {"failed": len(documents), "succeeded": 0, "total": len(documents)}
        
        # Prepare actions for bulk indexing
        actions = []
        for doc in documents:
            action = {
                "_index": index_name,
                "_source": doc
            }
            
            # Use specified field as ID if provided
            if id_field and id_field in doc:
                action["_id"] = doc[id_field]
            
            actions.append(action)
        
        # Set up progress bar if requested
        if progress_bar:
            pbar = tqdm(total=len(actions), desc=f"Indexing to {index_name}")
        
        # Index in chunks
        stats = {"succeeded": 0, "failed": 0}
        client = await self.init_client()
        
        for i in range(0, len(actions), chunk_size):
            chunk = actions[i:i + chunk_size]
            try:
                success, failed_items = await async_bulk(
                    client,
                    chunk,
                    chunk_size=chunk_size,
                    refresh=refresh,
                    raise_on_error=False
                )
                stats["succeeded"] += success
                stats["failed"] += len(failed_items)
                
                if progress_bar:
                    pbar.update(len(chunk))
                    
            except Exception as e:
                logging.error(f"Error during bulk indexing (async): {str(e)}")
                stats["failed"] += len(chunk)
                if progress_bar:
                    pbar.update(len(chunk))
        
        if progress_bar:
            pbar.close()
        
        stats["total"] = len(documents)
        return stats
    
    async def index_documents_with_embeddings(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        embedding_field: str,
        embedding_fn: Callable[[str], List[float]],
        text_field: str = "text",
        id_field: Optional[str] = None,
        chunk_size: int = 100,
        refresh: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, int]:
        """
        Index documents with real-time embedding generation (async).
        
        Args:
            index_name: Name of the vector index
            documents: List of documents to index
            embedding_field: Field name to store embeddings
            embedding_fn: Function that takes text and returns an embedding vector
            text_field: Field containing text to embed
            id_field: Field to use as document ID
            chunk_size: Number of documents to process in each batch
            refresh: Whether to refresh the index after each batch
            progress_bar: Whether to show a progress bar
            
        Returns:
            Dict[str, int]: Stats about the indexing operation
        """
        if not await self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return {"failed": len(documents), "succeeded": 0, "total": len(documents)}
        
        # Process documents in batches to generate embeddings
        processed_docs = []
        
        # Set up progress bar if requested
        if progress_bar:
            pbar = tqdm(total=len(documents), desc=f"Generating embeddings")
        
        for i in range(0, len(documents), chunk_size):
            batch = documents[i:i + chunk_size]
            
            for doc in batch:
                if text_field not in doc:
                    logging.warning(f"Document missing text field '{text_field}', skipping")
                    continue
                
                try:
                    # Generate embedding for the document
                    embedding = embedding_fn(doc[text_field])
                    doc_copy = doc.copy()
                    doc_copy[embedding_field] = embedding
                    processed_docs.append(doc_copy)
                except Exception as e:
                    logging.error(f"Error generating embedding: {str(e)}")
            
            if progress_bar:
                pbar.update(len(batch))
        
        if progress_bar:
            pbar.close()
        
        # Index the processed documents with embeddings
        return await self.index_documents(
            index_name=index_name,
            documents=processed_docs,
            id_field=id_field,
            chunk_size=chunk_size,
            refresh=refresh,
            progress_bar=progress_bar
        )
    
    async def index_documents_with_precomputed_embeddings(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        embedding_field: str = "embedding",
        id_field: Optional[str] = None,
        chunk_size: int = 500,
        refresh: bool = False,
        progress_bar: bool = True
    ) -> Dict[str, int]:
        """
        Index documents with precomputed embeddings (async).
        
        Args:
            index_name: Name of the vector index
            documents: List of documents with embeddings already included
            embedding_field: Field name containing embeddings
            id_field: Field to use as document ID
            chunk_size: Number of documents to index in each batch
            refresh: Whether to refresh the index after each batch
            progress_bar: Whether to show a progress bar
            
        Returns:
            Dict[str, int]: Stats about the indexing operation
        """
        # Validate that documents have embeddings
        invalid_docs = []
        valid_docs = []
        
        for i, doc in enumerate(documents):
            if embedding_field not in doc:
                invalid_docs.append(i)
                continue
                
            # Check that embedding is a list/array of numbers
            embedding = doc[embedding_field]
            if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                invalid_docs.append(i)
                continue
                
            valid_docs.append(doc)
        
        if invalid_docs:
            logging.warning(f"Found {len(invalid_docs)} documents missing valid embeddings, skipping these")
        
        # Index the valid documents
        return await self.index_documents(
            index_name=index_name,
            documents=valid_docs,
            id_field=id_field,
            chunk_size=chunk_size,
            refresh=refresh,
            progress_bar=progress_bar
        )
    
    async def reindex(
        self,
        source_index: str,
        target_index: str,
        query: Optional[Dict[str, Any]] = None,
        script: Optional[Dict[str, Any]] = None,
        wait_for_completion: bool = True,
        refresh: bool = True,
        timeout: str = "1m"
    ) -> Dict[str, Any]:
        """
        Reindex documents from one index to another (async).
        
        Args:
            source_index: Source index name
            target_index: Target index name
            query: Optional query to filter documents
            script: Optional script to transform documents during reindexing
            wait_for_completion: Wait for the operation to complete
            refresh: Refresh the target index after reindexing
            timeout: Timeout for the operation
            
        Returns:
            Dict[str, Any]: Result of the reindex operation
        """
        if not await self.index_exists(source_index):
            error_msg = f"Source index '{source_index}' does not exist"
            logging.error(error_msg)
            return {"error": error_msg}
        
        # Create the reindex body
        body = {
            "source": {
                "index": source_index
            },
            "dest": {
                "index": target_index
            }
        }
        
        # Add query if provided
        if query:
            body["source"]["query"] = query
        
        # Add script if provided
        if script:
            body["script"] = script
        
        try:
            client = await self.init_client()
            result = await client.reindex(
                body=body,
                refresh=refresh,
                wait_for_completion=wait_for_completion,
                timeout=timeout
            )
            return result
        except Exception as e:
            error_msg = f"Error during reindex operation (async): {str(e)}"
            logging.error(error_msg)
            return {"error": error_msg}