"""
Call Transcription Processing and Qdrant Storage Module

This module provides functions to:
1. Process call transcription files
2. Perform recursive chunking
3. Generate BGE-M3 embeddings
4. Create Qdrant collections
5. Store embeddings in Qdrant
"""

# import os
import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, 
    # CreateCollection, CollectionInfo
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize the embedding model
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, device="cpu")

@dataclass
class TranscriptionChunk:
    """Data class for transcription chunks"""
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


class TranscriptionProcessor:
    """Main class for processing call transcriptions and storing in Qdrant"""
    
    def __init__(
        self, 
        qdrant_host: str = "127.0.0.1",
        qdrant_port: int = 6333,
        embedding_model: str = "BAAI/bge-m3"
    ):
        """
        Initialize the transcription processor
        
        Args:
            qdrant_url: Qdrant server URL
            qdrant_port: Qdrant server port
            embedding_model: BGE-M3 model name
        """
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedding_model = model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_transcription_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load transcription file (supports JSON, TXT, CSV)
        
        Args:
            file_path: Path to the transcription file
            
        Returns:
            Dictionary containing transcription data
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return {
                    'text': content,
                    'filename': file_path.name,
                    'file_path': str(file_path)
                }

        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def recursive_chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[TranscriptionChunk]:
        """
        Perform recursive chunking on transcription text
        
        Args:
            text: Transcription text to chunk
            metadata: Metadata to attach to chunks
            
        Returns:
            List of TranscriptionChunk objects
        """
        chunks = self.text_splitter.split_text(text)
        
        transcription_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'chunk_count': len(chunks),
                'chunk_size': len(chunk_text)
            })
            
            chunk = TranscriptionChunk(
                id=str(uuid.uuid4()),
                text=chunk_text,
                metadata=chunk_metadata
            )
            transcription_chunks.append(chunk)
        
        return transcription_chunks
    
    def generate_embeddings(self, chunks: List[TranscriptionChunk]) -> List[TranscriptionChunk]:
        """
        Generate BGE-M3 embeddings for chunks
        
        Args:
            chunks: List of TranscriptionChunk objects
            
        Returns:
            List of chunks with embeddings added
        """
        texts = [chunk.text for chunk in chunks]
        
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts, 
        )
        for chunk, embedding in zip(chunks, embeddings['dense_vecs']):
            # print(f"Embedding: {embedding}, type : {len(embedding)}")
            chunk.embedding = embedding.tolist()
        
        return chunks

    def create_qdrant_collection(
        self,
        vector_size: int = 1024,
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ) -> bool:
        """
        Create Qdrant collection with proper schema

        Args:
            vector_size: Size of the embedding vectors (BGE-M3 is 1024)
            distance: Distance metric for similarity search
            recreate: Whether to recreate if collection exists

        Returns:
            True if collection created/exists, False otherwise
        """
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            collection_exists = any(
                col.name == "call_transcriptions"
                for col in collections.collections
            )

            if collection_exists:
                if recreate:
                    print(f"Deleting existing collection: call_transcriptions")
                    self.qdrant_client.delete_collection("call_transcriptions")
                else:
                    print(f"Collection call_transcriptions already exists")
                    return True

            # Create collection
            print(f"Creating collection: call_transcriptions")
            self.qdrant_client.create_collection(
                collection_name="call_transcriptions",
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )

            print(f"Collection call_transcriptions created successfully")
            return True

        except Exception as e:
            print(f"Error creating collection: {e}")
            return False

    def upsert_chunks_to_qdrant(
        self,
        chunks: List[TranscriptionChunk],
        collection_name: str,
        batch_size: int = 100
    ) -> bool:
        """
        Upsert transcription chunks to Qdrant collection

        Args:
            chunks: List of TranscriptionChunk objects with embeddings
            collection_name: Name of the Qdrant collection
            batch_size: Number of points to upsert in each batch

        Returns:
            True if successful, False otherwise
        """
        try:
            points = []
            for chunk in chunks:
                if chunk.embedding is None:
                    raise ValueError(f"Chunk {chunk.id} has no embedding")

                point = PointStruct(
                    id=chunk.id,
                    vector=chunk.embedding,
                    payload={
                        'text': chunk.text,
                        **chunk.metadata
                    }
                )
                points.append(point)

            # Upsert in batches
            print(f"Upserting {len(points)} points to collection call_transcriptions...")

            for i in tqdm(range(0, len(points), batch_size)):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name="call_transcriptions",
                    points=batch
                )

            print(f"Successfully upserted {len(points)} points")
            return True

        except Exception as e:
            print(f"Error upserting to Qdrant: {e}")
            return False

    def process_transcription_file(
        self,
        file_path: str,
        collection_name: str,
        recreate_collection: bool = False
    ) -> bool:
        """
        Complete pipeline: load file, chunk, embed, and store in Qdrant

        Args:
            file_path: Path to transcription file
            collection_name: Name of Qdrant collection
            recreate_collection: Whether to recreate collection if exists

        Returns:
            True if successful, False otherwise
        """
        # try:
        print(f"Processing transcription file: {file_path}")

        # 1. Load transcription file
        transcription_data = self.load_transcription_file(file_path)

        # 2. Prepare metadata
        metadata = {
            'source_file': file_path,
            'filename': Path(file_path).name,
            'file_type': Path(file_path).suffix.lower(),
            'processed_at': pd.Timestamp.now().isoformat()
        }

        # Add any additional metadata from the file
        if isinstance(transcription_data, dict):
            for key, value in transcription_data.items():
                if key != 'text' and not isinstance(value, (list, dict)):
                    metadata[key] = value

        # 3. Recursive chunking
        print("Performing recursive chunking...")
        chunks = self.recursive_chunk_text(
            transcription_data.get('text', str(transcription_data)),
            metadata
        )
        print(f"Created {len(chunks)} chunks")

        # 4. Generate embeddings
        chunks_with_embeddings = self.generate_embeddings(chunks)

        # 5. Create Qdrant collection
        success = self.create_qdrant_collection(
            vector_size=1024,
            recreate=recreate_collection
        )

        if not success:
            return False

        # 6. Upsert to Qdrant
        success = self.upsert_chunks_to_qdrant(
            chunks_with_embeddings,
            "call_transcriptions"
        )

        if success:
            print(f"Successfully processed {file_path}")
            print(f"Collection: call_transcriptions")
            print(f"Total chunks: {len(chunks)}")

        return success

        # except Exception as e:
        #     print(f"Error processing transcription file: {e}")
        #     return False


# Utility functions for easy usage
def process_single_transcription(
    file_path: str,
    collection_name: str = "call_transcriptions",
    qdrant_url: str = "127.0.0.1",
    qdrant_port: int = 6333
) -> bool:
    """
    Convenience function to process a single transcription file and store vectorize embeddings to Qdrant

    Args:
        file_path: Path to transcription file
        collection_name: Name of the Qdrant collection
        qdrant_url: Qdrant server URL
        qdrant_port: Qdrant server port

    Returns:
        True if successful, False otherwise
    """
    processor = TranscriptionProcessor(qdrant_url, qdrant_port)
    return processor.process_transcription_file(
        file_path,
        collection_name
    )

