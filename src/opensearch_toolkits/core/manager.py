import os
import logging
from typing import Dict, Any, Optional, List, Union
import time
from pydantic import BaseModel, Field, model_validator
from abc import ABC, abstractmethod
from .client import create_opensearch_client, create_async_opensearch_client, OpenSearchConfig


class BaseOpenSearchManager(ABC):
    """Base abstract class for managing OpenSearch connections and operations."""
    
    def __init__(self, config: OpenSearchConfig):
        """Initialize the OpenSearch manager with configuration."""
        self.config = config


class SyncOpenSearchManager(BaseOpenSearchManager):
    """Class for managing synchronous OpenSearch operations."""
    
    def __init__(self, config: OpenSearchConfig):
        """Initialize the synchronous OpenSearch manager with configuration."""
        super().__init__(config)
        self.client = create_opensearch_client(config)
    
    def check_connection(self) -> bool:
        """Check if connection to OpenSearch is healthy."""
        try:
            return self.client.ping()
        except Exception as e:
            logging.error(f"Connection error: {e}")
            return False
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get basic information about the OpenSearch cluster."""
        return self.client.info()
    
    def get_cluster_health(self) -> Dict[str, Any]:
        """Get health information about the OpenSearch cluster."""
        return self.client.cluster.health()
    
    def list_indices(self, pattern: str = "*") -> List[str]:
        """List all available OpenSearch indices, optionally filtered by pattern."""
        indices = self.client.indices.get(index=pattern)
        return list(indices.keys())
    
    def get_all_indices(self) -> List[str]:
        """
        Get all OpenSearch indices using the get_alias method.
        This method provides a more comprehensive list including aliases.

        Returns:
            List[str]: List of index names
        """
        try:
            indices = self.client.indices.get_alias(index="*")
            return list(indices.keys())
        except Exception as e:
            logging.error(f"Error listing indices: {str(e)}")
            return []
    
    def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific index.

        Args:
            index_name: Name of the index to get statistics for

        Returns:
            Dict[str, Any]: Index statistics information
        """
        if not self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return {"error": "Index not found"}

        try:
            stats = self.client.indices.stats(index=index_name)
            result = {
                "doc_count": stats["_all"]["primaries"]["docs"]["count"],
                "size_bytes": stats["_all"]["primaries"]["store"]["size_in_bytes"],
                "index_name": index_name,
                "shards": {
                    "total": stats["_shards"]["total"],
                    "successful": stats["_shards"]["successful"],
                    "failed": stats["_shards"]["failed"]
                }
            }
            return result
        except Exception as e:
            logging.error(f"Error retrieving stats for index '{index_name}': {str(e)}")
            return {"error": str(e)}
    
    def get_mappings(self, index: str) -> Dict[str, Any]:
        """Get field mappings for a specific OpenSearch index."""
        return self.client.indices.get_mapping(index=index)
    
    def get_shard_info(self, index: Optional[str] = None) -> Dict[str, Any]:
        """Get shard information for all or specific indices."""
        return self.client.cat.shards(index=index, format="json")
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get detailed cluster statistics."""
        return self.client.cluster.stats()
    
    def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists.

        Args:
            index_name: Name of the index to check

        Returns:
            bool: True if the index exists, False otherwise
        """
        return self.client.indices.exists(index=index_name)
    
    def update_index_settings(self, index_name: str, settings: Dict[str, Any]) -> bool:
        """
        Update settings for a specific index.

        Args:
            index_name: Name of the index to update
            settings: Settings to apply to the index

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return False

        try:
            self.client.indices.put_settings(index=index_name, body=settings)
            logging.info(f"Successfully updated settings for index '{index_name}'")
            return True
        except Exception as e:
            logging.error(f"Error updating settings for index '{index_name}': {str(e)}")
            return False
    
    def clone_index(self, source_index: str, target_index: str) -> bool:
        """
        Clone an index.

        Args:
            source_index: Name of the source index
            target_index: Name of the target index

        Returns:
            bool: True if successful, False otherwise
        """
        if not self.index_exists(source_index):
            logging.warning(f"Source index '{source_index}' does not exist")
            return False

        if self.index_exists(target_index):
            logging.warning(f"Target index '{target_index}' already exists")
            return False

        try:
            # Make source index read-only
            self.client.indices.put_settings(
                index=source_index,
                body={"settings": {"index.blocks.write": True}}
            )

            # Clone the index
            self.client.indices.clone(index=source_index, target=target_index)

            # Make source index writable again
            self.client.indices.put_settings(
                index=source_index,
                body={"settings": {"index.blocks.write": None}}
            )

            logging.info(f"Successfully cloned index from '{source_index}' to '{target_index}'")
            return True
        except Exception as e:
            logging.error(f"Error cloning index from '{source_index}' to '{target_index}': {str(e)}")
            # Try to make source index writable again in case of error
            try:
                self.client.indices.put_settings(
                    index=source_index,
                    body={"settings": {"index.blocks.write": None}}
                )
            except:
                pass
            return False


class AsyncOpenSearchManager(BaseOpenSearchManager):
    """Class for managing asynchronous OpenSearch operations."""
    
    def __init__(self, config: OpenSearchConfig):
        """Initialize the asynchronous OpenSearch manager with configuration."""
        super().__init__(config)
        self.client = None
        self._is_closing = False
    
    async def init_client(self):
        """Initialize the async client lazily."""
        if self.client is None:
            self.client = create_async_opensearch_client(self.config)
        return self.client
    
    async def close(self):
        """Close client session and clean up resources properly."""
        if self.client is not None and not self._is_closing:
            self._is_closing = True
            try:
                await self.client.close()
                # Access and close the underlying HTTP session if available
                if hasattr(self.client, 'transport'):
                    transport = self.client.transport
                    if hasattr(transport, 'connection_pool'):
                        connection_pool = transport.connection_pool
                        for conn in connection_pool.connections:
                            if hasattr(conn, 'session') and hasattr(conn.session, 'close'):
                                await conn.session.close()
            except Exception as e:
                logging.error(f"Error closing async client: {e}")
            finally:
                self.client = None
                self._is_closing = False
    
    async def __aenter__(self):
        """Support for async context manager."""
        await self.init_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are cleaned up when exiting context."""
        await self.close()
    
    async def check_connection(self) -> bool:
        """Check if async connection to OpenSearch is healthy."""
        try:
            client = await self.init_client()
            return await client.ping()
        except Exception as e:
            logging.error(f"Async connection error: {e}")
            return False
    
    async def get_cluster_info(self) -> Dict[str, Any]:
        """Get basic information about the OpenSearch cluster (async)."""
        client = await self.init_client()
        return await client.info()
    
    async def get_cluster_health(self) -> Dict[str, Any]:
        """Get health information about the OpenSearch cluster (async)."""
        client = await self.init_client()
        return await client.cluster.health()
    
    async def list_indices(self, pattern: str = "*") -> List[str]:
        """List all available OpenSearch indices, optionally filtered by pattern (async)."""
        client = await self.init_client()
        indices = await client.indices.get(index=pattern)
        return list(indices.keys())
    
    async def get_all_indices(self) -> List[str]:
        """
        Get all OpenSearch indices using the get_alias method (async).

        Returns:
            List[str]: List of index names
        """
        try:
            client = await self.init_client()
            indices = await client.indices.get_alias(index="*")
            return list(indices.keys())
        except Exception as e:
            logging.error(f"Error listing indices asynchronously: {str(e)}")
            return []
    
    async def get_index_stats(self, index_name: str) -> Dict[str, Any]:
        """
        Get statistics for a specific index (async).

        Args:
            index_name: Name of the index to get statistics for

        Returns:
            Dict[str, Any]: Index statistics information
        """
        if not await self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return {"error": "Index not found"}

        try:
            client = await self.init_client()
            stats = await client.indices.stats(index=index_name)
            result = {
                "doc_count": stats["_all"]["primaries"]["docs"]["count"],
                "size_bytes": stats["_all"]["primaries"]["store"]["size_in_bytes"],
                "index_name": index_name,
                "shards": {
                    "total": stats["_shards"]["total"],
                    "successful": stats["_shards"]["successful"],
                    "failed": stats["_shards"]["failed"]
                }
            }
            return result
        except Exception as e:
            logging.error(f"Error retrieving stats for index '{index_name}' asynchronously: {str(e)}")
            return {"error": str(e)}
    
    async def get_mappings(self, index: str) -> Dict[str, Any]:
        """Get field mappings for a specific OpenSearch index (async)."""
        client = await self.init_client()
        return await client.indices.get_mapping(index=index)
    
    async def get_shard_info(self, index: Optional[str] = None) -> Dict[str, Any]:
        """Get shard information for all or specific indices (async)."""
        client = await self.init_client()
        return await client.cat.shards(index=index, format="json")
    
    async def get_cluster_stats(self) -> Dict[str, Any]:
        """Get detailed cluster statistics (async)."""
        client = await self.init_client()
        return await client.cluster.stats()
    
    async def index_exists(self, index_name: str) -> bool:
        """
        Check if an index exists (async).

        Args:
            index_name: Name of the index to check

        Returns:
            bool: True if the index exists, False otherwise
        """
        client = await self.init_client()
        return await client.indices.exists(index=index_name)
    
    async def update_index_settings(self, index_name: str, settings: Dict[str, Any]) -> bool:
        """
        Update settings for a specific index (async).

        Args:
            index_name: Name of the index to update
            settings: Settings to apply to the index

        Returns:
            bool: True if successful, False otherwise
        """
        if not await self.index_exists(index_name):
            logging.warning(f"Index '{index_name}' does not exist")
            return False

        try:
            client = await self.init_client()
            await client.indices.put_settings(index=index_name, body=settings)
            logging.info(f"Successfully updated settings for index '{index_name}' asynchronously")
            return True
        except Exception as e:
            logging.error(f"Error updating settings for index '{index_name}' asynchronously: {str(e)}")
            return False
    
    async def clone_index(self, source_index: str, target_index: str) -> bool:
        """
        Clone an index (async).

        Args:
            source_index: Name of the source index
            target_index: Name of the target index

        Returns:
            bool: True if successful, False otherwise
        """
        if not await self.index_exists(source_index):
            logging.warning(f"Source index '{source_index}' does not exist")
            return False

        if await self.index_exists(target_index):
            logging.warning(f"Target index '{target_index}' already exists")
            return False

        try:
            client = await self.init_client()
            
            # Make source index read-only
            await client.indices.put_settings(
                index=source_index,
                body={"settings": {"index.blocks.write": True}}
            )

            # Clone the index
            await client.indices.clone(index=source_index, target=target_index)

            # Make source index writable again
            await client.indices.put_settings(
                index=source_index,
                body={"settings": {"index.blocks.write": None}}
            )

            logging.info(f"Successfully cloned index from '{source_index}' to '{target_index}' asynchronously")
            return True
        except Exception as e:
            logging.error(f"Error cloning index from '{source_index}' to '{target_index}' asynchronously: {str(e)}")
            # Try to make source index writable again in case of error
            try:
                client = await self.init_client()
                await client.indices.put_settings(
                    index=source_index,
                    body={"settings": {"index.blocks.write": None}}
                )
            except:
                pass
            return False