"""
Prompt Templates for Call Transcription Processing Agent

This module contains all the prompts used throughout the application,
centralized for easy management and consistency.
"""

# System prompt for the LangGraph agent
AGENT_SYSTEM_PROMPT = """
You are a specialized call transcription processing agent. You have access to exactly 5 tools and must choose EXACTLY ONE tool based on the user's request.

**PRIORITY RULE: If the user asks ANY QUESTION (starts with what/how/when/where/why/tell me), use answer_question_from_transcripts**

**TOOL SELECTION RULES:**

1. **Use ingest_transcription ONLY when:**
   - User wants to "ingest", "process", "load", "import", or "store" a NEW transcription file
   - User provides a file path to process
   - Keywords: "ingest", "process", "load", "import", "add", "store"
   - Example: "ingest transcript from /path/to/file.txt"

2. **Use get_last_transcript_summary ONLY when:**
   - User wants the MOST RECENT or LATEST call transcript or summary
   - User asks for "last call", "latest transcript", "recent call", "most recent", "summary"
   - NO specific filename is mentioned
   - Keywords: "last", "latest", "recent", "most recent", "newest", "summary"
   - Example: "show me the last call transcript" or "give me a summary of the latest call"

3. **Use get_transcript_by_filename ONLY when:**
   - User specifies a PARTICULAR filename or transcript name WITH FILE EXTENSION
   - User wants a SPECIFIC transcript file (not content search)
   - A complete filename with extension is explicitly mentioned (.txt, .json, .csv)
   - Keywords: "named", "called", "file", "transcript named", "get file"
   - Example: "get transcript named 'call_001.txt'" or "retrieve file demo_call.json"
   - DO NOT use for content questions or when no file extension is mentioned

4. **Use list_all_transcripts ONLY when:**
   - User wants to see ALL available transcripts or call IDs
   - User asks for a list of files, calls, or transcripts
   - User wants to browse what's available in the system
   - Keywords: "list", "all", "show all", "available", "calls", "transcripts", "filenames", "what calls", "browse"
   - Example: "list all transcripts" or "show me all available calls"

5. **Use answer_question_from_transcripts WHEN:**
   - User asks ANY QUESTION about transcript content (starts with what/how/when/where/why/tell me)
   - User wants to find information across transcripts
   - User asks about pricing, costs, rates, or financial details
   - User mentions people, roles, or specific topics WITHOUT file extensions
   - Keywords: "what", "how", "when", "where", "why", "tell me", "price", "cost", "rate", "pricing", "find", "search", "about", "said", "discussed"
   - Example: "What was the pricing discussed?", "Tell me what CRO said about pricing", "How much does it cost?"

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**
- CALL EXACTLY ONE TOOL per user request
- NEVER call multiple tools in sequence
- After calling a tool, STOP and return the result
- The tool result is complete - do NOT call additional tools
- If user asks for "summary" or "last call" → use get_last_transcript_summary (it generates a summary automatically)
- If user asks for "list" or "all calls" → use list_all_transcripts
- If user asks a QUESTION about content → use answer_question_from_transcripts
- DO NOT call multiple tools for one request
- DO NOT call any tool after another tool has already been called

**Examples:**
❌ WRONG: Call get_last_transcript_summary, then call get_transcript_by_filename
✅ CORRECT: Call get_last_transcript_summary and STOP

❌ WRONG: User says "show last call" → calling ingest_transcription
✅ CORRECT: User says "show last call" → calling get_last_transcript_summary and STOP

❌ WRONG: User says "list all calls" → calling get_last_transcript_summary
✅ CORRECT: User says "list all calls" → calling list_all_transcripts and STOP

❌ WRONG: User says "get file.txt" → calling get_last_transcript_summary
✅ CORRECT: User says "get file.txt" → calling get_transcript_by_filename and STOP

❌ WRONG: User says "what was the pricing?" → calling get_last_transcript_summary
✅ CORRECT: User says "what was the pricing?" → calling answer_question_from_transcripts and STOP

❌ WRONG: User says "tell me what CRO said about pricing" → calling get_transcript_by_filename
✅ CORRECT: User says "tell me what CRO said about pricing" → calling answer_question_from_transcripts and STOP
"""

# Summarization prompt for generating call summaries
SUMMARIZATION_PROMPT_TEMPLATE = """
Please provide a concise summary of this call transcript:

**Call Information:**
- File: {filename}
- Processed: {processed_at}
- Total chunks: {total_chunks}

**Transcript:**
{full_text}

**Instructions:**
1. Summarize the main topics discussed
2. Identify key participants and their roles
3. Highlight any important decisions or outcomes
4. Keep the summary concise (3-5 sentences)
5. Focus on actionable items or resolutions

**Summary:**
"""

# Tool descriptions
INGEST_TOOL_DESCRIPTION = """
INGESTION TOOL: Process and store a NEW call transcription file in Qdrant.

Use this tool ONLY when the user wants to:
- Ingest, process, load, import, or store a NEW transcription file
- Add a new file to the database
- Process a file from a given file path

Args:
    file_path: Path to the transcription file to process
    
Returns:
    Success message or error details
"""

LAST_SUMMARY_TOOL_DESCRIPTION = """
RETRIEVAL TOOL: Get the MOST RECENT call transcript from the database.

Use this tool ONLY when the user wants:
- The LAST, LATEST, MOST RECENT, or NEWEST call transcript
- Recent call summary without specifying a filename
- To see what was discussed in the latest call

DO NOT use this tool if a specific filename is mentioned.

Returns:
    Formatted string with the complete last call transcript and metadata
"""

FILENAME_TOOL_DESCRIPTION = """
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

# Success/Error message templates
SUCCESS_MESSAGES = {
    "ingestion": "✅ Successfully ingested transcription from {file_path} into collection '{collection_name}'",
    "retrieval": "✅ Retrieved and summarized {chunk_count} chunks successfully",
    "processing": "✅ Processing completed successfully"
}

ERROR_MESSAGES = {
    "ingestion_failed": "❌ Failed to ingest transcription from {file_path}",
    "retrieval_failed": "❌ Error retrieving transcript: {error}",
    "no_transcripts": "❌ No transcripts found in the collection",
    "file_not_found": "❌ No transcript found with filename: {filename}",
    "file_path_not_found": "❌ File not found at path: {filename}",
    "processing_error": "❌ Error processing transcription: {error}",
    "api_key_missing": "❌ {api_key_name} not found in environment variables",
    "duplicate_transcript": "⚠️ Transcript '{filename}' already exists in collection '{collection_name}'\nPreviously processed at: {processed_at}\nNo need to ingest again. Use retrieval tools to access the existing transcript."
}

# Output format templates
OUTPUT_FORMATS = {
    "file_info": """**File Information:**
- Filename: {filename}
- Processed: {processed_at}
- Total Chunks: {total_chunks}""",
    
    "summary_format": """**File Information:**
- Filename: {filename}
- Processed: {processed_at}
- Total Chunks: {total_chunks}

**Summary:**
{summary_content}""",
    
    "full_transcript_format": """**File Information:**
- Filename: {filename}
- Source: {source_file}
- Processed: {processed_at}
- Total Chunks: {total_chunks}

**Transcript Content:**
{full_text}

---
✅ Retrieved {total_chunks} chunks successfully"""
}

# Configuration messages
CONFIG_MESSAGES = {
    "qdrant_connection": "🗄️ Qdrant: {qdrant_url}:{qdrant_port}",
    "collection_target": "📊 Target collection: {collection_name}",
    "processing_file": "🔄 Processing transcription: {file_path}",
    "retrieving_last": "🔍 Retrieving last call transcript from collection: {collection_name}",
    "retrieving_specific": "🔍 Retrieving transcript: {filename}"
}

# Help and instruction messages
HELP_MESSAGES = {
    "capabilities": """I can help you with call transcription processing:

**Available Commands:**
1. **Ingest new transcripts:** "ingest transcript from /path/to/file.txt"
2. **Get last call summary:** "show me the last call transcript"
3. **Get specific transcript:** "retrieve transcript named 'filename.txt'"

**Supported file formats:** JSON, TXT
**Features:** Recursive chunking, BGE-M3 embeddings, Qdrant storage""",
    
    "usage_examples": """**Usage Examples:**
- "ingest transcript from /path/file.txt" → Process new file
- "show me the last call" → Get recent call summary  
- "get transcript named 'call.txt'" → Get specific file
- "what was discussed in the latest call?" → Get recent summary""",
    
    "tool_selection_tips": """**Tool Selection Tips:**
- Use specific keywords: 'ingest', 'last', 'named'
- Be explicit about file paths for ingestion
- Mention 'last' or 'latest' for recent transcripts
- Include filename for specific transcript retrieval"""
}

# Validation messages
VALIDATION_MESSAGES = {
    "file_path_required": "Please provide a valid file path for ingestion",
    "filename_required": "Please specify the filename you want to retrieve",
    "invalid_request": "I couldn't understand your request. Please try again with a clearer instruction.",
    "multiple_tools_warning": "Please make one request at a time"
}

def get_summarization_prompt(filename: str, processed_at: str, total_chunks: int, full_text: str) -> str:
    """
    Generate a summarization prompt with the provided parameters
    
    Args:
        filename: Name of the transcript file
        processed_at: Processing timestamp
        total_chunks: Number of chunks
        full_text: Full transcript text
        
    Returns:
        Formatted summarization prompt
    """
    return SUMMARIZATION_PROMPT_TEMPLATE.format(
        filename=filename,
        processed_at=processed_at,
        total_chunks=total_chunks,
        full_text=full_text
    )

def get_success_message(message_type: str, **kwargs) -> str:
    """
    Get a formatted success message
    
    Args:
        message_type: Type of success message
        **kwargs: Parameters for formatting
        
    Returns:
        Formatted success message
    """
    return SUCCESS_MESSAGES.get(message_type, "✅ Operation completed successfully").format(**kwargs)

def get_error_message(message_type: str, **kwargs) -> str:
    """
    Get a formatted error message
    
    Args:
        message_type: Type of error message
        **kwargs: Parameters for formatting
        
    Returns:
        Formatted error message
    """
    return ERROR_MESSAGES.get(message_type, "❌ An error occurred").format(**kwargs)

def get_output_format(format_type: str, **kwargs) -> str:
    """
    Get a formatted output template
    
    Args:
        format_type: Type of output format
        **kwargs: Parameters for formatting
        
    Returns:
        Formatted output string
    """
    return OUTPUT_FORMATS.get(format_type, "").format(**kwargs)


# RAG (Retrieval-Augmented Generation) prompt template
RAG_PROMPT_TEMPLATE = """
Based on the following transcript chunks, provide a precise and detailed answer to the user's question.

**User Question:** {question}

**Relevant Transcript Context:**
{context}

**Instructions:**
1. Answer the question directly and precisely
2. If the question is about pricing, costs, or rates, provide specific numbers and details
3. Quote relevant parts from the transcripts to support your answer
4. If information is incomplete, mention what's available and what's missing
5. Be concise but comprehensive
6. If no relevant information is found, clearly state that you can't answer, don't try to make things up

**Answer:**
"""


def get_rag_prompt(question: str, context: str) -> str:
    """
    Get the RAG prompt for answer generation

    Args:
        question: User's question
        context: Relevant transcript context

    Returns:
        Formatted RAG prompt
    """
    return RAG_PROMPT_TEMPLATE.format(question=question, context=context).strip()
