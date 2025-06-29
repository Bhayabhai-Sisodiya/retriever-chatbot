#!/usr/bin/env python3
"""
MCP Server for Call Transcription Tools

This module creates an MCP (Model Context Protocol) server that exposes
the call transcription processing tools using FastMCP.
"""

import os
import logging
# from typing import Any, Dict, List
from dotenv import load_dotenv

# Configure logging to suppress verbose INFO logs
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("qdrant_client").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)
logging.getLogger("mcp.server").setLevel(logging.WARNING)
logging.getLogger("server").setLevel(logging.WARNING)

# Set root logger to WARNING to suppress all INFO logs
logging.basicConfig(level=logging.WARNING)

from mcp.server.fastmcp import FastMCP

# Import our existing tools
from transcription_processor import process_single_transcription
from transcript_retriever import (
    get_last_call_summary,
    get_transcript_by_name,
    # check_transcript_exists_by_filename,
    list_all_transcript_filenames,
    search_transcripts_rag
)
from templates import (
    get_success_message,
    get_error_message,
    # get_output_format,
    get_rag_prompt
)

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("Call Transcription Server")

@mcp.tool()
def ingest_transcription(file_path: str) -> str:
    """
    INGESTION TOOL: Process and store a NEW call transcription file in Qdrant.

    Use this tool ONLY when the user wants to:
    - Ingest, process, load, import, or store a NEW transcription file
    - Add a new file to the database
    - Process a file from a given file path

    Args:
        file_path: Path to the transcription file to process

    Returns:
        Success or error message
    """
    try:
        # Configuration
        qdrant_url = os.getenv('QDRANT_URL', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        collection_name = "call_transcriptions"

        # Check if file exists
        if not os.path.exists(file_path):
            return get_error_message("file_path_not_found", filename=file_path)


        # Process the transcription
        success = process_single_transcription(
            file_path=file_path,
            collection_name=collection_name,
            qdrant_url=qdrant_url,
            qdrant_port=qdrant_port
        )

        if success:
            return get_success_message("ingestion", file_path=file_path, collection_name=collection_name)
        else:
            return get_error_message("ingestion_failed", file_path=file_path)

    except Exception as e:
        return get_error_message("processing_error", error=str(e))


@mcp.tool()
def get_last_transcript_summary() -> str:
    """
    RETRIEVAL TOOL: Get the MOST RECENT call transcript from the database.

    Use this tool ONLY when the user wants:
    - The LAST, LATEST, MOST RECENT, or NEWEST call transcript
    - Recent call summary without specifying a filename
    - To see what was discussed in the latest call

    DO NOT use this tool if a specific filename is mentioned.

    Returns:
        Formatted string with the complete last call transcript and metadata
    """
    try:
        # Configuration
        qdrant_url = os.getenv('QDRANT_URL', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        collection_name = "call_transcriptions"

        # Get the last call transcript
        result = get_last_call_summary(qdrant_url, qdrant_port, collection_name)
        return result

    except Exception as e:
        return get_error_message("retrieval_failed", error=str(e))


@mcp.tool()
def get_transcript_by_filename(filename: str) -> str:
    """
    SPECIFIC RETRIEVAL TOOL: Get a PARTICULAR transcript by its filename.

    Use this tool ONLY when the user:
    - Mentions a SPECIFIC filename or transcript name
    - Wants a PARTICULAR transcript (not the latest one)
    - Uses words like "named", "called", "file", or provides an actual filename

    DO NOT use this tool for "last" or "latest" requests.

    Args:
        filename: Name of the transcript file to retrieve

    Returns:
        Formatted string with the complete transcript and metadata
    """
    try:
        # Configuration
        qdrant_url = os.getenv('QDRANT_URL', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        collection_name = "call_transcriptions"

        # Get the specific transcript
        result = get_transcript_by_name(filename, qdrant_url, qdrant_port, collection_name)
        return result

    except Exception as e:
        return get_error_message("retrieval_failed", error=str(e))


@mcp.tool()
def list_all_transcripts() -> str:
    """
    LIST TOOL: Get a list of all transcript filenames (call IDs) in the collection.

    Use this tool ONLY when the user wants:
    - To see ALL available transcripts/calls
    - A list of call IDs or filenames
    - To browse available transcripts
    - To see what calls are stored in the system

    Keywords: "list", "all", "show all", "available", "calls", "transcripts", "filenames"

    Returns:
        Formatted string with all transcript filenames and metadata
    """
    try:
        # Configuration
        qdrant_url = os.getenv('QDRANT_URL', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        collection_name = "call_transcriptions"

        # Get all transcript filenames
        result = list_all_transcript_filenames(qdrant_url, qdrant_port, collection_name)
        return result

    except Exception as e:
        return get_error_message("retrieval_failed", error=str(e))


@mcp.tool()
def answer_question_from_transcripts(question: str) -> str:
    """
    RAG TOOL: Answer questions by searching through transcript chunks and generating precise answers.

    Use this tool ONLY when the user:
    - Asks a SPECIFIC QUESTION about transcript content
    - Wants to find information across multiple transcripts
    - Asks about pricing, costs, rates, or financial details
    - Needs to search for specific topics or keywords
    - Wants detailed answers from transcript content

    Keywords: "what", "how", "when", "where", "why", "price", "cost", "rate", "pricing", "find", "search", "tell me about"

    Args:
        question: The question to answer using transcript content

    Returns:
        Formatted string with the answer and relevant context from transcripts
    """
    try:
        # Configuration
        qdrant_url = os.getenv('QDRANT_URL', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        collection_name = "call_transcriptions"

        # Perform RAG search to get context
        search_result = search_transcripts_rag(question, qdrant_url, qdrant_port, collection_name, limit=20)

        # Check if search was successful
        if search_result.get("error", False):
            return get_error_message("retrieval_failed", error=search_result["message"])

        # Extract context and generate answer using LLM
        context = search_result["context"]
        total_results = search_result["total_results"]

        if total_results == 0:
            return f"❌ No relevant information found in transcripts for: '{question}'"

        # Create RAG prompt for answer generation using template
        rag_prompt = get_rag_prompt(question, context)

        # Use the LLM to generate the answer
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1  # Low temperature for factual accuracy
        )

        answer_response = llm.invoke([HumanMessage(content=rag_prompt)])

        # Format the final response
        final_answer = f"""**Question:** {question}

**Answer:**
{answer_response.content}

**Sources:** {total_results} relevant transcript chunks from collection '{collection_name}'"""

        return final_answer.strip()

    except Exception as e:
        return get_error_message("retrieval_failed", error=str(e))


# Add a resource for server information
@mcp.resource("server://info")
def get_server_info() -> str:
    """Get information about the MCP server and available tools."""
    return """
# Call Transcription MCP Server

This server provides access to call transcription processing and retrieval tools:

## Available Tools:
1. **ingest_transcription** - Process and store new transcription files
2. **get_last_transcript_summary** - Get the most recent transcript
3. **get_transcript_by_filename** - Get specific transcript by filename
4. **list_all_transcripts** - List all available transcripts
5. **answer_question_from_transcripts** - RAG-based question answering

## Configuration:
- Qdrant URL: {qdrant_url}
- Qdrant Port: {qdrant_port}
- Collection: call_transcriptions

## Usage:
Connect to this server using an MCP client to access transcription tools.
""".format(
        qdrant_url=os.getenv('QDRANT_URL', 'localhost'),
        qdrant_port=os.getenv('QDRANT_PORT', 6333)
    )


def main():
    """Run the MCP server"""
    print("🚀 Starting Call Transcription MCP Server...")
    print("Available tools:")
    print("  - ingest_transcription")
    print("  - get_last_transcript_summary") 
    print("  - get_transcript_by_filename")
    print("  - list_all_transcripts")
    print("  - answer_question_from_transcripts")
    print("\nServer running on stdio...")
    
    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()
