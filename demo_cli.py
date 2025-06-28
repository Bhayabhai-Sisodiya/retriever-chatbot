#!/usr/bin/env python3
"""
Demo script to test the CLI functionality

Usage: python demo_cli.py
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_commands():
    """Demonstrate CLI commands programmatically"""
    print("🎬 CLI Demo - Call Transcription Agent")
    print("=" * 50)
    
    try:
        from langgraph_agent import run_transcription_agent
        
        demo_queries = [
            {
                "query": "list all transcripts",
                "description": "List all available transcripts"
            },
            {
                "query": "show me the last call transcript", 
                "description": "Get the most recent call summary"
            },
            {
                "query": "what was the pricing discussed?",
                "description": "Ask a question about pricing (RAG)"
            },
            {
                "query": "tell me what CRO said about our pricing?",
                "description": "Specific question about CRO and pricing"
            }
        ]
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"\n📋 Demo {i}: {demo['description']}")
            print(f"Query: '{demo['query']}'")
            print("-" * 40)
            
            try:
                result = run_transcription_agent(demo['query'])
                print(result)
            except Exception as e:
                print(f"❌ Error: {e}")
            
            print("-" * 40)
            
            # Pause between demos
            input("\nPress Enter to continue to next demo...")
        
        print("\n🎉 Demo completed!")
        print("\nTo start the interactive CLI, run:")
        print("  python chat.py")
        print("  or")
        print("  python src/main.py")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all dependencies are installed")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    demo_commands()
