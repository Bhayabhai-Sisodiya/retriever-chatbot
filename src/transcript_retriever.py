#!/usr/bin/env python3
"""
Transcript Retriever Module

This module provides functions to retrieve and summarize call transcripts from Qdrant.
"""

import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, OrderBy, Direction, Distance, VectorParams
from templates import get_output_format, get_error_message

# Load environment variables
load_dotenv()

from FlagEmbedding import BGEM3FlagModel

# Initialize the embedding model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device="cpu")

class TranscriptRetriever:
    """Class for retrieving call transcripts from Qdrant"""
    
    def __init__(
        self, 
        qdrant_url: str = "127.0.0.1",
        qdrant_port: int = 6333,
        collection_name: str = "call_transcriptions"
    ):
        """
        Initialize the transcript retriever
        
        Args:
            qdrant_url: Qdrant server URL
            qdrant_port: Qdrant server port
            collection_name: Name of the Qdrant collection
        """
        self.qdrant_client = QdrantClient(host=qdrant_url, port=qdrant_port)
        self.collection_name = collection_name

        # Check if collection exists and create if it doesn't
        # self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """
        Check if the collection exists and create it if it doesn't
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_exists = any(
                col.name == self.collection_name
                for col in collections.collections
            )

            if not collection_exists:
                print(f"📦 Collection '{self.collection_name}' does not exist. Creating...")
                self._create_collection()
                print(f"✅ Collection '{self.collection_name}' created successfully")
            else:
                print(f"✅ Collection '{self.collection_name}' already exists")

        except Exception as e:
            print(f"❌ Error checking/creating collection: {e}")
            # Continue anyway - the error will be caught when trying to use the collection

    def _create_collection(self):
        """
        Create the Qdrant collection with proper schema
        """
        # Create collection with BGE-M3 vector configuration
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=1024,  # BGE-M3 embedding size
                distance=Distance.COSINE
            )
        )

    def check_transcript_exists(self, filename: str) -> Dict[str, Any]:
        """
        Check if a transcript with the given filename already exists in Qdrant

        Args:
            filename: Name of the transcript file to check

        Returns:
            Dictionary with 'exists' boolean and metadata if found
        """
        try:
            # Search for any chunks with this filename
            search_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="filename",
                            match=MatchValue(value=filename)
                        )
                    ]
                ),
                limit=1,  # We only need to know if at least one exists
                with_payload=True,
                with_vectors=False
            )

            points = search_result[0]  # Get the points from the scroll result

            if points:
                # Extract metadata from the first point
                first_point = points[0]
                payload = first_point.payload

                return {
                    "exists": True,
                    "filename": payload.get("filename"),
                    "processed_at": payload.get("processed_at"),
                    "source_file": payload.get("source_file"),
                    "total_chunks": len(points) if len(points) < 1000 else "1000+",  # Approximate
                    "message": f"Transcript '{filename}' already exists in the collection"
                }
            else:
                return {
                    "exists": False,
                    "message": f"Transcript '{filename}' not found in collection"
                }

        except Exception as e:
            return {
                "exists": False,
                "error": True,
                "message": f"Error checking transcript existence: {str(e)}"
            }

    def list_all_filenames(self) -> Dict[str, Any]:
        """
        Get a list of all unique filenames (call IDs) in the collection

        Returns:
            Dictionary with success status and list of filenames with metadata
        """
        try:
            # Scroll through all points to get unique filenames
            all_filenames = {}
            offset = None

            while True:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    limit=100,  # Process in batches
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = scroll_result

                if not points:
                    break

                # Extract unique filenames and their metadata
                for point in points:
                    payload = point.payload
                    filename = payload.get("filename")

                    if filename and filename not in all_filenames:
                        all_filenames[filename] = {
                            "filename": filename,
                            "source_file": payload.get("source_file", ""),
                            "processed_at": payload.get("processed_at", ""),
                            "chunk_count": 1
                        }
                    elif filename:
                        # Increment chunk count for existing filename
                        all_filenames[filename]["chunk_count"] += 1

                # Check if we have more data
                if next_offset is None:
                    break
                offset = next_offset

            # Convert to sorted list
            filename_list = list(all_filenames.values())
            filename_list.sort(key=lambda x: x.get("processed_at", ""), reverse=True)

            return {
                "success": True,
                "total_files": len(filename_list),
                "filenames": filename_list,
                "message": f"Found {len(filename_list)} unique transcript files"
            }

        except Exception as e:
            return {
                "success": False,
                "error": True,
                "message": f"Error listing filenames: {str(e)}"
            }

    def search_transcripts(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """
        Perform semantic search on transcript chunks using the query

        Args:
            query: Search query/question
            limit: Number of top results to return

        Returns:
            Dictionary with search results and metadata
        """
        try:

            # Generate dense embedding for the query
            query_embedding = model.encode([query],
                                         batch_size=1,
                                         max_length=8192)['dense_vecs'][0]

            # Perform vector search in Qdrant
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            # Process search results
            results = []
            for result in search_results:
                payload = result.payload
                results.append({
                    "score": result.score,
                    "text": payload.get("text", ""),
                    "filename": payload.get("filename", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "processed_at": payload.get("processed_at", ""),
                    "source_file": payload.get("source_file", "")
                })

            return {
                "success": True,
                "query": query,
                "total_results": len(results),
                "results": results,
                "message": f"Found {len(results)} relevant chunks"
            }

        except Exception as e:
            return {
                "success": False,
                "error": True,
                "message": f"Error searching transcripts: {str(e)}"
            }
    
    def get_last_call_transcript(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get the most recent call transcript chunks from Qdrant

        Args:
            limit: Maximum number of chunks to retrieve

        Returns:
            Dictionary containing transcript data and metadata
        """
        try:
            # Scroll through collection ordered by processed_at timestamp (descending)
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                order_by=OrderBy(
                    key="processed_at",
                    direction=Direction.DESC
                ),
                with_payload=True,
                with_vectors=False
            )

            if not scroll_result[0]:  # No points found
                return {
                    "success": False,
                    "message": "No transcripts found in the collection",
                    "chunks": [],
                    "filename": None,
                    "full_text": ""
                }

            # Get the most recent filename
            latest_point = scroll_result[0][0]
            latest_filename = latest_point.payload.get('filename', 'unknown')

            # Filter all chunks with the same filename
            same_file_chunks = [
                point for point in scroll_result[0]
                if point.payload.get('filename') == latest_filename
            ]

            # Sort chunks by chunk_index to maintain order
            same_file_chunks.sort(
                key=lambda x: x.payload.get('chunk_index', 0)
            )
            
            # Extract text and metadata
            chunks_data = []
            full_text = ""
            
            for chunk in same_file_chunks:
                chunk_info = {
                    "chunk_index": chunk.payload.get('chunk_index', 0),
                    "text": chunk.payload.get('text', ''),
                    "chunk_size": chunk.payload.get('chunk_size', 0),
                    "processed_at": chunk.payload.get('processed_at', ''),
                }
                chunks_data.append(chunk_info)
                full_text += chunk.payload.get('text', '') + "\n\n"
            
            # Get metadata from the first chunk
            first_chunk = same_file_chunks[0].payload
            metadata = {
                "filename": latest_filename,
                "source_file": first_chunk.get('source_file', ''),
                "file_type": first_chunk.get('file_type', ''),
                "processed_at": first_chunk.get('processed_at', ''),
                "total_chunks": len(same_file_chunks),
                "chunk_count": first_chunk.get('chunk_count', 0)
            }
            
            return {
                "success": True,
                "message": f"Retrieved {len(same_file_chunks)} chunks from latest transcript",
                "metadata": metadata,
                "chunks": chunks_data,
                "full_text": full_text.strip()
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error retrieving transcript: {str(e)}",
                "chunks": [],
                "filename": None,
                "full_text": ""
            }
    
    def get_transcript_by_filename(self, filename: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get all chunks for a specific transcript file
        
        Args:
            filename: Name of the transcript file
            limit: Maximum number of chunks to retrieve
            
        Returns:
            Dictionary containing transcript data and metadata
        """
        try:
            # Search for chunks with specific filename
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="filename",
                            match=MatchValue(value=filename)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            
            if not scroll_result[0]:
                return {
                    "success": False,
                    "message": f"No transcript found with filename: {filename}",
                    "chunks": [],
                    "filename": filename,
                    "full_text": ""
                }
            
            # Sort chunks by chunk_index
            chunks = sorted(
                scroll_result[0],
                key=lambda x: x.payload.get('chunk_index', 0)
            )
            
            # Extract text and metadata
            chunks_data = []
            full_text = ""
            
            for chunk in chunks:
                chunk_info = {
                    "chunk_index": chunk.payload.get('chunk_index', 0),
                    "text": chunk.payload.get('text', ''),
                    "chunk_size": chunk.payload.get('chunk_size', 0),
                    "processed_at": chunk.payload.get('processed_at', ''),
                }
                chunks_data.append(chunk_info)
                full_text += chunk.payload.get('text', '') + "\n\n"
            
            # Get metadata
            first_chunk = chunks[0].payload
            metadata = {
                "filename": filename,
                "source_file": first_chunk.get('source_file', ''),
                "file_type": first_chunk.get('file_type', ''),
                "processed_at": first_chunk.get('processed_at', ''),
                "total_chunks": len(chunks),
                "chunk_count": first_chunk.get('chunk_count', 0)
            }
            
            return {
                "success": True,
                "message": f"Retrieved {len(chunks)} chunks for {filename}",
                "metadata": metadata,
                "chunks": chunks_data,
                "full_text": full_text.strip()
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error retrieving transcript: {str(e)}",
                "chunks": [],
                "filename": filename,
                "full_text": ""
            }
    
    def list_available_transcripts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List all available transcripts in the collection
        
        Args:
            limit: Maximum number of transcripts to list
            
        Returns:
            List of transcript metadata
        """
        try:
            # Get all points and group by filename
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Get more points to find unique filenames
                with_payload=True,
                with_vectors=False
            )
            
            if not scroll_result[0]:
                return []
            
            # Group by filename and get metadata
            transcripts = {}
            for point in scroll_result[0]:
                filename = point.payload.get('filename', 'unknown')
                if filename not in transcripts:
                    transcripts[filename] = {
                        "filename": filename,
                        "source_file": point.payload.get('source_file', ''),
                        "file_type": point.payload.get('file_type', ''),
                        "processed_at": point.payload.get('processed_at', ''),
                        "chunk_count": 0
                    }
                transcripts[filename]["chunk_count"] += 1
            
            # Sort by processed_at (most recent first)
            transcript_list = list(transcripts.values())
            transcript_list.sort(
                key=lambda x: x.get('processed_at', ''),
                reverse=True
            )
            
            return transcript_list[:limit]
            
        except Exception as e:
            print(f"Error listing transcripts: {e}")
            return []


# Utility functions for easy usage
def get_last_call_summary(
    qdrant_url: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "call_transcriptions"
) -> str:
    """
    Get a formatted summary of the last call transcript
    
    Args:
        qdrant_url: Qdrant server URL
        qdrant_port: Qdrant server port
        collection_name: Name of the Qdrant collection
        
    Returns:
        Formatted string with transcript summary
    """
    retriever = TranscriptRetriever(qdrant_url, qdrant_port, collection_name)
    result = retriever.get_last_call_transcript()
    
    if not result["success"]:
        return f"❌ {result['message']}"
    
    metadata = result["metadata"]
    full_text = result["full_text"]
    
    # Use template for formatting
    summary = get_output_format(
        "full_transcript_format",
        filename=metadata['filename'],
        source_file=metadata['source_file'],
        processed_at=metadata['processed_at'],
        total_chunks=metadata['total_chunks'],
        full_text=full_text
    )

    return summary.strip()


def get_transcript_by_name(
    filename: str,
    qdrant_url: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "call_transcriptions"
) -> str:
    """
    Get a formatted transcript by filename
    
    Args:
        filename: Name of the transcript file
        qdrant_url: Qdrant server URL
        qdrant_port: Qdrant server port
        collection_name: Name of the Qdrant collection
        
    Returns:
        Formatted string with transcript content
    """
    retriever = TranscriptRetriever(qdrant_url, qdrant_port, collection_name)
    result = retriever.get_transcript_by_filename(filename)
    
    if not result["success"]:
        return f"❌ {result['message']}"
    
    metadata = result["metadata"]
    full_text = result["full_text"]
    
    # Use template for formatting
    summary = get_output_format(
        "full_transcript_format",
        filename=metadata['filename'],
        source_file=metadata['source_file'],
        processed_at=metadata['processed_at'],
        total_chunks=metadata['total_chunks'],
        full_text=full_text
    )

    return summary.strip()


def check_transcript_exists_by_filename(
    filename: str,
    qdrant_url: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "call_transcriptions"
) -> Dict[str, Any]:
    """
    Check if a transcript with the given filename already exists in Qdrant

    Args:
        filename: Name of the transcript file to check
        qdrant_url: Qdrant server URL
        qdrant_port: Qdrant server port
        collection_name: Name of the Qdrant collection

    Returns:
        Dictionary with existence status and metadata
    """
    retriever = TranscriptRetriever(qdrant_url, qdrant_port, collection_name)
    return retriever.check_transcript_exists(filename)


def list_all_transcript_filenames(
    qdrant_url: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "call_transcriptions"
) -> str:
    """
    Get a formatted list of all transcript filenames (call IDs) in the collection

    Args:
        qdrant_url: Qdrant server URL
        qdrant_port: Qdrant server port
        collection_name: Name of the Qdrant collection

    Returns:
        Formatted string with all filenames and metadata
    """
    retriever = TranscriptRetriever(qdrant_url, qdrant_port, collection_name)
    result = retriever.list_all_filenames()

    if not result["success"]:
        return get_error_message("retrieval_failed", error=result["message"])

    filenames = result["filenames"]
    total_files = result["total_files"]

    if total_files == 0:
        return get_error_message("no_transcripts")

    # Format the output
    output_lines = [
        f"📋 **All Call Transcripts ({total_files} files)**",
        "",
        "**File List:**"
    ]

    for i, file_info in enumerate(filenames, 1):
        filename = file_info["filename"]
        processed_at = file_info["processed_at"]
        chunk_count = file_info["chunk_count"]

        # Format the date for better readability
        try:
            from datetime import datetime
            if processed_at:
                dt = datetime.fromisoformat(processed_at.replace('Z', '+00:00'))
                formatted_date = dt.strftime("%Y-%m-%d %H:%M")
            else:
                formatted_date = "Unknown"
        except:
            formatted_date = processed_at or "Unknown"

        output_lines.append(f"{i}. **{filename}**")
        output_lines.append(f"   - Processed: {formatted_date}")
        output_lines.append(f"   - Chunks: {chunk_count}")
        output_lines.append("")

    output_lines.append("---")
    output_lines.append(f"✅ Total: {total_files} transcript files in collection '{collection_name}'")

    return "\n".join(output_lines)


def search_transcripts_rag(
    query: str,
    qdrant_url: str = "localhost",
    qdrant_port: int = 6333,
    collection_name: str = "call_transcriptions",
    limit: int = 10
) -> dict:
    """
    Perform RAG search on transcripts and return structured results

    Args:
        query: Question or search query
        qdrant_url: Qdrant server URL
        qdrant_port: Qdrant server port
        collection_name: Name of the Qdrant collection
        limit: Number of top results to retrieve

    Returns:
        Dictionary with search results and formatted context
    """
    retriever = TranscriptRetriever(qdrant_url, qdrant_port, collection_name)
    result = retriever.search_transcripts(query, limit)

    if not result["success"]:
        return get_error_message("retrieval_failed", error=result["message"])

    search_results = result["results"]
    total_results = result["total_results"]

    if total_results == 0:
        return f"❌ No relevant transcript chunks found for query: '{query}'"

    # Format the search results for context
    context_chunks = []
    for i, chunk in enumerate(search_results, 1):
        score = chunk["score"]
        text = chunk["text"]
        filename = chunk["filename"]
        chunk_index = chunk["chunk_index"]

        context_chunks.append(f"**Chunk {i}** (Score: {score:.3f}, File: {filename}, Index: {chunk_index})")
        context_chunks.append(f"{text}")
        context_chunks.append("")  # Empty line for separation

    # Combine all context
    full_context = "\n".join(context_chunks)

    return {
        "error": False,
        "query": query,
        "total_results": total_results,
        "context": full_context,
        "raw_results": search_results
    }
