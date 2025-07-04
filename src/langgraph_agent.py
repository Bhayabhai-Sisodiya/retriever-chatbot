#!/usr/bin/env python3
"""
LangGraph Agent for Call Transcription Processing

This module creates a LangGraph-based agent that can ingest call transcripts
using the transcription processing functions.
"""

import os
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

from transcription_processor import process_single_transcription
from transcript_retriever import get_transcript_by_name, check_transcript_exists_by_filename, list_all_transcript_filenames, search_transcripts_rag
from templates import (
    AGENT_SYSTEM_PROMPT,
    get_summarization_prompt,
    get_success_message,
    get_error_message,
    get_output_format,
    get_rag_prompt
)

# Load environment variables
load_dotenv()

# Define the state for our agent
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Create the tool for processing transcriptions
@tool
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
        Success message or error details
    """
    try:
        # Configuration
        collection_name = "call_transcriptions"
        qdrant_url = os.getenv('QDRANT_URL', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))

        # Extract filename from file path for checking
        filename = os.path.basename(file_path)

        # Check if transcript already exists
        existence_check = check_transcript_exists_by_filename(
            filename=filename,
            qdrant_url=qdrant_url,
            qdrant_port=qdrant_port,
            collection_name=collection_name
        )

        # If transcript already exists, return early with message
        if existence_check.get("exists", False):
            return get_error_message(
                "duplicate_transcript",
                filename=filename,
                collection_name=collection_name,
                processed_at=existence_check.get("processed_at", "unknown time")
            )

        # If there was an error checking existence, log it but continue with ingestion
        if existence_check.get("error", False):
            print(f"Warning: Could not check transcript existence: {existence_check.get('message', 'Unknown error')}")
            print("Proceeding with ingestion...")

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

@tool
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

        # Configuration loaded silently

        # Get the last call transcript chunks
        from transcript_retriever import TranscriptRetriever
        retriever = TranscriptRetriever(qdrant_url, qdrant_port, collection_name)
        result = retriever.get_last_call_transcript()

        if not result["success"]:
            return f"❌ {result['message']}"

        # Generate summary using LLM
        full_text = result["full_text"]
        metadata = result["metadata"]

        # Create summary prompt using template
        summary_prompt = get_summarization_prompt(
            filename=metadata['filename'],
            processed_at=metadata['processed_at'],
            total_chunks=metadata['total_chunks'],
            full_text=full_text
        )

        # Use the LLM to generate summary
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.3
        )

        summary_response = llm.invoke([HumanMessage(content=summary_prompt)])

        # Format the final response using template
        final_summary = get_output_format(
            "summary_format",
            filename=metadata['filename'],
            processed_at=metadata['processed_at'],
            total_chunks=metadata['total_chunks'],
            summary_content=summary_response.content
        )

        return final_summary.strip()

    except Exception as e:
        return get_error_message("retrieval_failed", error=str(e))

@tool
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

        # Configuration loaded silently

        # Get the transcript by filename
        transcript = get_transcript_by_name(
            filename=filename,
            qdrant_url=qdrant_url,
            qdrant_port=qdrant_port,
            collection_name=collection_name
        )

        return transcript

    except Exception as e:
        return get_error_message("retrieval_failed", error=str(e))

@tool
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
        Formatted list of all transcript filenames with metadata
    """
    try:
        # Configuration
        qdrant_url = os.getenv('QDRANT_URL', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        collection_name = "call_transcriptions"

        # Get the list of all transcripts
        result = list_all_transcript_filenames(
            qdrant_url=qdrant_url,
            qdrant_port=qdrant_port,
            collection_name=collection_name
        )

        return result

    except Exception as e:
        return get_error_message("retrieval_failed", error=str(e))

@tool
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
        Precise answer generated from relevant transcript chunks
    """
    try:
        # Configuration
        qdrant_url = os.getenv('QDRANT_URL', 'localhost')
        qdrant_port = int(os.getenv('QDRANT_PORT', 6333))
        collection_name = "call_transcriptions"

        # Search for relevant chunks
        search_result = search_transcripts_rag(
            query=question,
            qdrant_url=qdrant_url,
            qdrant_port=qdrant_port,
            collection_name=collection_name,
            limit=20
        )

        # Check if search was successful
        if isinstance(search_result, str):
            # Error case
            return search_result

        # Extract context and generate answer
        context = search_result["context"]
        total_results = search_result["total_results"]

        if total_results == 0:
            return f"❌ No relevant information found in transcripts for: '{question}'"

        # Create RAG prompt for answer generation using template
        rag_prompt = get_rag_prompt(question, context)

        # Use the LLM to generate the answer
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

# Initialize the LLM
def get_llm():
    """Initialize and return the LLM with tools"""
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.0  # Zero temperature for consistent tool selection
    )
    return llm.bind_tools([
        ingest_transcription,
        get_last_transcript_summary,
        get_transcript_by_filename,
        list_all_transcripts,
        answer_question_from_transcripts
    ])

# Define the agent node
def agent_node(state: AgentState):
    """Main agent node that processes messages and decides on actions"""
    messages = state["messages"]
    llm_with_tools = get_llm()
    
    # Add system message if this is the first interaction
    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        system_message = HumanMessage(content=AGENT_SYSTEM_PROMPT)
        messages = [system_message] + messages
    
    # Call the LLM with tools
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

# Define the tool execution node
def tool_node(state: AgentState):
    """Execute tools called by the agent"""
    messages = state["messages"]
    last_message = messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_results = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Execute the appropriate tool
            if tool_name == "ingest_transcription":
                result = ingest_transcription.invoke({"file_path": tool_args["file_path"]})

            elif tool_name == "get_last_transcript_summary":
                result = get_last_transcript_summary.invoke({})

            elif tool_name == "get_transcript_by_filename":
                result = get_transcript_by_filename.invoke({"filename": tool_args["filename"]})

            elif tool_name == "list_all_transcripts":
                result = list_all_transcripts.invoke({})

            elif tool_name == "answer_question_from_transcripts":
                result = answer_question_from_transcripts.invoke({"question": tool_args["question"]})

            else:
                result = f"❌ Unknown tool: {tool_name}"

            # Create tool message
            tool_message = ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
            tool_results.append(tool_message)

        return {"messages": tool_results}

    return {"messages": []}

# Define routing logic
def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Determine if we should continue or end"""
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message has tool calls, go to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # Otherwise, end the conversation
    return "__end__"

def should_continue_after_tools(state: AgentState) -> Literal["__end__"]:
    """After tools are executed, always end (no more loops)"""
    return "__end__"

# Create the graph
def create_agent():
    """Create and compile the LangGraph agent"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "__end__": END
        }
    )
    # After tools execute, always end (prevent infinite loops)
    workflow.add_edge("tools", END)
    
    # Compile the graph
    return workflow.compile()

# Create a function to run the agent
def run_transcription_agent(user_input: str, verbose: bool = False):
    """
    Run the LangGraph agent with user input

    Args:
        user_input: User's instruction/query
        verbose: Whether to print detailed output (default: False for clean output)
    """
    # Create the agent
    app = create_agent()
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    # Run the agent with recursion limit
    result = app.invoke(
        initial_state,
        config={"recursion_limit": 10}  # Limit recursion to prevent infinite loops
    )

    # Extract and print only the final tool result (summary/transcript)
    final_result = None
    for message in reversed(result["messages"]):
        if isinstance(message, ToolMessage):
            final_result = message.content
            break

    if final_result:
        print(final_result)
    else:
        # If no tool was called, show the AI response
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and not hasattr(message, 'tool_calls'):
                print(message.content)
                break
    
    return result

# Main execution
if __name__ == "__main__":
    # Example usage
    test_queries = [
        "ingest new call transcript from /home/ad.rapidops.com/bhayabhai.sisodiya/Documents/retriever-chatbot/1_demo_call.txt",
        "show me the last call transcript",
        "get the most recent call summary",
        "retrieve transcript named '1_demo_call.txt'"
    ]
    
    print("🚀 LangGraph Call Transcription Agent Demo")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test {i}:")
        try:
            run_transcription_agent(query)
        except Exception as e:
            print(f"❌ Error: {e}")
        
        if i < len(test_queries):
            print("\n" + "="*60)
