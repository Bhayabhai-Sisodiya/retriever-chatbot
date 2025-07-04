#!/usr/bin/env python3
"""
LangGraph Agent with MCP Tools Integration

This module creates a LangGraph agent that uses MCP (Model Context Protocol) 
to access call transcription tools through the MCP server.
"""

import os
import asyncio
import sys
import logging
from typing import Dict, Any, List
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

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()


class MCPLangGraphAgent:
    """LangGraph agent with MCP tools integration"""

    def __init__(self, server_script_path: str = "src/mcp_server.py"):
        """
        Initialize the MCP LangGraph agent

        Args:
            server_script_path: Path to the MCP server script
        """
        self.server_script_path = server_script_path
        self.llm = None
        self._llm_initialized = False

    def _initialize_llm(self):
        """Initialize the LLM once"""
        if not self._llm_initialized:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.0
            )
            self._llm_initialized = True

    async def chat(self):
        """
        Interactive chat interface with persistent MCP connection
        """
        # Initialize LLM if not done
        self._initialize_llm()

        print("\n" + "="*60)
        print("🤖 MCP LangGraph Agent - Interactive Chat")
        print("="*60)
        print("You can ask me to:")
        print("  • Process transcription files: 'ingest the file at /path/to/file.txt'")
        print("  • Get recent transcripts: 'show me the last call'")
        print("  • Find specific transcripts: 'get transcript named demo_call.txt'")
        print("  • List all transcripts: 'list all available calls'")
        print("  • Answer questions: 'what was discussed about pricing?'")
        print("  • Type 'quit' to exit")
        print("="*60)

        try:
            # Create server parameters for stdio connection
            server_params = StdioServerParameters(
                command="python",
                args=[self.server_script_path],
                env=None
            )

            # Use persistent connection for the entire chat session
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the connection
                    await session.initialize()

                    # Get tools
                    tools = await load_mcp_tools(session)

                    # Create agent with reused LLM
                    agent = create_react_agent(self.llm, tools)

                    # Create system message with instructions
                    system_message = SystemMessage(content="""You are a helpful assistant for call transcription processing and analysis.

You have access to tools for working with call transcripts:
1. ingest_transcription - Process and store new transcription files
2. get_last_transcript_summary - Get the most recent transcript
3. get_transcript_by_filename - Get specific transcript by filename
4. list_all_transcripts - List all available transcripts
5. answer_question_from_transcripts - Answer questions using RAG on transcripts

Guidelines:
- Use ingest_transcription when users want to process/load/import new files
- Use get_last_transcript_summary for "last", "latest", "most recent" requests
- Use get_transcript_by_filename when a specific filename is mentioned
- Use list_all_transcripts when users want to see all available transcripts
- Use answer_question_from_transcripts for specific questions about transcript content
- Always provide clear, helpful responses with relevant information

Be concise but informative in your responses.""")

                    # Interactive chat loop
                    while True:
                        try:
                            user_input = input("\n🗣️ You: ").strip()

                            if user_input.lower() in ['quit', 'exit', 'q']:
                                print("\n👋 Goodbye!")
                                break

                            if not user_input:
                                continue

                            print("🤖 Processing... ✨ ", end="", flush=True)

                            # Run the agent with the user input
                            agent_response = await agent.ainvoke({
                                "messages": [
                                    system_message,
                                    HumanMessage(content=user_input)
                                ]
                            })

                            # Extract and print the response
                            if agent_response and "messages" in agent_response:
                                last_message = agent_response["messages"][-1]
                                print(last_message.content)
                            else:
                                print("❌ No response from agent")

                        except KeyboardInterrupt:
                            print("\n👋 Goodbye!")
                            break
                        except Exception as e:
                            print(f"❌ Error processing message: {e}")

        except Exception as e:
            print(f"❌ Error setting up MCP connection: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function"""
    agent = MCPLangGraphAgent()
    asyncio.run(agent.chat())


if __name__ == "__main__":
    main()
