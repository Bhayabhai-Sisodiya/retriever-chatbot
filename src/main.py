#!/usr/bin/env python3
"""
Enhanced Interactive CLI for Call Transcription Processing Agent

Usage: 
    python src/main.py
    python -m src.main
"""

import os
import sys
import signal
import time
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TranscriptionCLI:
    """Enhanced CLI interface for the Call Transcription Agent"""
    
    def __init__(self):
        self.session_start = datetime.now()
        self.command_history: List[str] = []
        self.agent_available = False
        
    def check_requirements(self) -> bool:
        """Check if all required dependencies and configurations are available"""
        try:
            # Check if langgraph_agent module is available
            from langgraph_agent import run_transcription_agent
            
            # Check if OpenAI API key is set
            if not os.getenv("OPENAI_API_KEY"):
                print("❌ OPENAI_API_KEY not found in environment variables")
                print("Please set your OpenAI API key in the .env file")
                return False
            
            self.agent_available = True
            return True
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please make sure all dependencies are installed: pip install -r requirement.txt")
            return False
        except Exception as e:
            print(f"❌ Setup error: {e}")
            return False
    
    def print_header(self):
        """Print application header"""
        print("🤖 Call Transcription Processing Agent")
        print("=" * 50)
        print(f"Session started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Powered by LangGraph + OpenAI + Qdrant + BGE-M3")
        print("=" * 50)
    
    def print_welcome(self):
        """Print welcome message and quick start guide"""
        self.print_header()
        print()
        print("🚀 Welcome! I'm your AI assistant for call transcription processing.")
        print()
        print("🎯 Quick Start:")
        print("  Type your question naturally, like:")
        print("  • 'What was discussed about pricing?'")
        print("  • 'Show me the last call transcript summary'")
        print("  • 'List all available calls'")
        print()
        print("⚡ Commands:")
        print("  help    - Show detailed help")
        print("  examples- Show usage examples")
        print("  status  - Check system status")
        print("  history - Show command history")
        print("  clear   - Clear screen")
        print("  quit    - Exit")
        print()
        print("💡 Tip: I can process files, answer questions, and search transcripts!")
        print("-" * 50)
    
    def print_help(self):
        """Print detailed help information"""
        print("\n📖 Detailed Help - Call Transcription Agent")
        print("=" * 55)
        print()
        print("🔧 5 Main Capabilities:")
        print()
        print("1️⃣  FILE INGESTION")
        print("   • 'ingest transcript from /path/to/file.txt'")
        print("   • 'process the file at /home/user/call.json'")
        print("   • Supports: .txt, .json, .csv files")
        print("   • Automatically detects duplicates")
        print()
        print("2️⃣  LATEST TRANSCRIPT")
        print("   • 'show me the last call transcript'")
        print("   • 'get the most recent call summary'")
        print("   • 'what was discussed in the latest call?'")
        print()
        print("3️⃣  SPECIFIC TRANSCRIPT")
        print("   • 'get transcript named demo_call.txt'")
        print("   • 'retrieve file customer_meeting.txt'")
        print("   • Must include filename with extension")
        print()
        print("4️⃣  LIST ALL TRANSCRIPTS")
        print("   • 'list all transcripts'")
        print("   • 'show me all available calls'")
        print("   • 'what calls do we have?'")
        print()
        print("5️⃣  QUESTION ANSWERING (RAG)")
        print("   • 'what was the pricing discussed?'")
        print("   • 'tell me what CRO said about our pricing?'")
        print("   • 'how much does the service cost?'")
        print("   • 'what security concerns were raised?'")
        print("   • 'who were the participants?'")
        print("   • 'when is the go-live date?'")
        print()
        print("🎯 The agent automatically selects the right tool based on your question!")
        print()
    
    def print_examples(self):
        """Print usage examples by category"""
        print("\n📚 Usage Examples by Category")
        print("=" * 40)
        print()
        print("💰 PRICING QUESTIONS:")
        print("  • 'What was the pricing discussed in the calls?'")
        print("  • 'How much does the AI Copilot cost?'")
        print("  • 'Tell me about the pricing model'")
        print("  • 'What rates were mentioned?'")
        print()
        print("👥 PEOPLE & ROLES:")
        print("  • 'Who was involved in the discussion?'")
        print("  • 'What did the CRO say about pricing?'")
        print("  • 'Tell me about the sales engineer comments'")
        print()
        print("🔒 SECURITY & FEATURES:")
        print("  • 'What security concerns were raised?'")
        print("  • 'Tell me about the AI Copilot features'")
        print("  • 'What was discussed about data security?'")
        print()
        print("📅 TIMELINE & NEXT STEPS:")
        print("  • 'When is the go-live date?'")
        print("  • 'What are the next steps?'")
        print("  • 'What was the timeline discussed?'")
        print()
    
    def check_system_status(self):
        """Check and display comprehensive system status"""
        print("\n🔍 System Status Check")
        print("=" * 30)
        
        # API Configuration
        print("🔑 API Configuration:")
        if os.getenv("OPENAI_API_KEY"):
            key_preview = os.getenv("OPENAI_API_KEY")[:8] + "..."
            print(f"  ✅ OpenAI API: {key_preview}")
        else:
            print("  ❌ OpenAI API: Not configured")
        
        # Qdrant Configuration
        qdrant_url = os.getenv("QDRANT_URL", "localhost")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        print(f"  🗄️ Qdrant: {qdrant_url}:{qdrant_port}")
        
        # Test Qdrant Connection
        print("\n🔌 Connectivity:")
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=qdrant_url, port=int(qdrant_port))
            collections = client.get_collections()
            print(f"  ✅ Qdrant: Connected ({len(collections.collections)} collections)")
            
            # Check transcription collection
            try:
                collection_info = client.get_collection("call_transcriptions")
                print(f"  ✅ Transcripts: {collection_info.points_count} stored")
            except Exception:
                print("  ⚠️  Transcripts: Collection not found (will be created)")
                
        except Exception as e:
            print(f"  ❌ Qdrant: Connection failed ({e})")
        
        # Agent Status
        print("\n🤖 Agent Status:")
        if self.agent_available:
            print("  ✅ LangGraph Agent: Ready")
            print("  ✅ Tools: 5 tools available")
            print("  ✅ Embeddings: BGE-M3 model")
        else:
            print("  ❌ Agent: Not available")
        
        # Session Info
        uptime = datetime.now() - self.session_start
        print(f"\n📊 Session Info:")
        print(f"  ⏱️  Uptime: {uptime}")
        print(f"  📝 Commands: {len(self.command_history)}")
        print()
    
    def show_history(self):
        """Show command history"""
        print("\n📜 Command History")
        print("=" * 25)
        if not self.command_history:
            print("No commands executed yet.")
        else:
            for i, cmd in enumerate(self.command_history[-10:], 1):  # Show last 10
                print(f"{i:2d}. {cmd}")
        print()
    
    def handle_special_commands(self, user_input: str) -> bool:
        """Handle special commands. Returns True if command was handled."""
        command = user_input.lower().strip()
        
        if command in ['help', 'h', '?']:
            self.print_help()
            return True
        
        elif command in ['examples', 'example', 'ex']:
            self.print_examples()
            return True
        
        elif command in ['status', 'stat', 'check']:
            self.check_system_status()
            return True
        
        elif command in ['history', 'hist']:
            self.show_history()
            return True
        
        elif command in ['clear', 'cls']:
            os.system('clear' if os.name == 'posix' else 'cls')
            self.print_welcome()
            return True
        
        elif command in ['quit', 'exit', 'q', 'bye']:
            self.print_goodbye()
            return True
        
        return False
    
    def print_goodbye(self):
        """Print goodbye message with session summary"""
        uptime = datetime.now() - self.session_start
        print(f"\n👋 Session Summary")
        print("=" * 20)
        print(f"Duration: {uptime}")
        print(f"Commands executed: {len(self.command_history)}")
        print()
        print("Thank you for using the Call Transcription Agent!")
        print("🤖 Goodbye!")
    
    def process_user_input(self, user_input: str) -> str:
        """Process user input with the agent"""
        try:
            from langgraph_agent import run_transcription_agent
            
            # Add to history
            self.command_history.append(user_input)
            
            # Show processing indicator
            print("🤖 Processing", end="", flush=True)
            for _ in range(3):
                time.sleep(0.3)
                print(".", end="", flush=True)
            print(" ✨")
            print()
            
            # Process with agent
            result = run_transcription_agent(user_input)
            return result
            
        except Exception as e:
            return f"❌ Error: {e}\n\n💡 Try 'help' for usage guidance or 'status' to check configuration."
    
    def run(self):
        """Main CLI loop"""
        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, lambda sig, frame: (self.print_goodbye(), sys.exit(0)))
        
        # Check requirements
        if not self.check_requirements():
            sys.exit(1)
        
        # Print welcome
        self.print_welcome()
        
        # Main interaction loop
        while True:
            try:
                # Get user input with nice prompt
                user_input = input("\n💬 Ask me anything: ").strip()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Handle special commands
                if self.handle_special_commands(user_input):
                    if user_input.lower().strip() in ['quit', 'exit', 'q', 'bye']:
                        break
                    continue
                
                # Process with agent
                print("-" * 50)
                result = self.process_user_input(user_input)
                print(result)
                print("-" * 50)
                
            except KeyboardInterrupt:
                break
            except EOFError:
                self.print_goodbye()
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                print("Please try again or type 'quit' to exit.")

def main():
    """Entry point for the CLI application"""
    cli = TranscriptionCLI()
    cli.run()

if __name__ == "__main__":
    main()
