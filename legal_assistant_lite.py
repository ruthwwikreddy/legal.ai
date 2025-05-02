"""
Legal Assistant Lite - A lightweight AI assistant for Indian legal research
Uses Ollama to process and analyze Indian legal texts like IPC, Constitution, and case law
"""

import os
import sys
import json
import time
import requests
import argparse
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path

# Constants
OLLAMA_API_URL = "http://localhost:11434/api"
# Models optimized for legal tasks
RECOMMENDED_MODELS = {
    "high_quality": ["llama3:latest", "mistral:latest", "deepseek-llm:latest"],  # Best quality, most resource-intensive
    "balanced": ["phi3:latest", "llama2:7b", "mistral:7b"],  # Good balance of quality and performance
    "lightweight": ["tinyllama:latest", "gemma:2b", "phi:latest"]  # Fastest, least resource-intensive
}
DEFAULT_MODEL = "mistral"  # Good balance of quality and performance for legal tasks
DATA_DIR = Path("legal_data")
SUPPORTED_DOCUMENT_TYPES = ["ipc", "constitution", "judgments", "acts", "general"]

class LegalAssistantLite:
    def __init__(self, model: str = DEFAULT_MODEL):
        """
        Initialize the Legal Assistant Lite application
        
        Args:
            model: Name of the Ollama model to use
        """
        self.model = model
        self.data_dir = DATA_DIR
        self.ensure_data_directory()
        self.document_index = self.build_document_index()
        self.verify_ollama_availability()
        
    def ensure_data_directory(self) -> None:
        """Create the data directory structure if it doesn't exist"""
        self.data_dir.mkdir(exist_ok=True)
        for doc_type in SUPPORTED_DOCUMENT_TYPES:
            (self.data_dir / doc_type).mkdir(exist_ok=True)
            
    def verify_ollama_availability(self) -> None:
        """Verify that Ollama is running and the selected model is available"""
        try:
            # Check if Ollama server is running
            response = requests.get(f"{OLLAMA_API_URL}/tags")
            if response.status_code != 200:
                print(f"Error: Ollama server not available. Please make sure Ollama is running.")
                sys.exit(1)
                
            # Check if the selected model is available
            models = response.json().get("models", [])
            available_models = [model["name"] for model in models]
            
            if self.model not in available_models:
                print(f"Warning: Model '{self.model}' not found in Ollama.")
                print(f"Available models: {', '.join(available_models)}")
                
                # Suggest alternative models based on categories
                flat_recommended_models = [model for category in RECOMMENDED_MODELS.values() for model in category]
                available_recommended = [model for model in flat_recommended_models if model in available_models]
                
                if available_recommended:
                    print(f"\nRecommended alternatives that are already installed:")
                    for model in available_recommended:
                        print(f"  - {model}")
                
                user_input = input(f"Do you want to pull '{self.model}' now? (y/n): ")
                
                if user_input.lower() == 'y':
                    print(f"Pulling model '{self.model}'. This may take a while...")
                    self._run_ollama_command("pull", {"name": self.model})
                    print(f"Model '{self.model}' pulled successfully!")
                else:
                    if available_recommended:
                        alt_model = available_recommended[0]
                        use_alt = input(f"Use '{alt_model}' instead? (y/n): ")
                        if use_alt.lower() == 'y':
                            self.model = alt_model
                            print(f"Using '{alt_model}' instead.")
                        else:
                            print("Please select an available model or pull the desired model.")
                            sys.exit(1)
                    else:
                        print("Please select an available model or pull the desired model.")
                        sys.exit(1)
                    
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama. Please make sure Ollama is installed and running.")
            print("You can install Ollama from: https://ollama.ai/")
            sys.exit(1)
            
    def build_document_index(self) -> Dict[str, Dict[str, Path]]:
        """
        Build an index of available legal documents
        
        Returns:
            Dictionary mapping document types to document IDs and their file paths
        """
        index = {doc_type: {} for doc_type in SUPPORTED_DOCUMENT_TYPES}
        
        for doc_type in SUPPORTED_DOCUMENT_TYPES:
            doc_dir = self.data_dir / doc_type
            if doc_dir.exists():
                for file_path in doc_dir.glob("*.txt"):
                    doc_id = file_path.stem
                    index[doc_type][doc_id] = file_path
                    
        return index
    
    def _run_ollama_command(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a command against the Ollama API
        
        Args:
            endpoint: API endpoint to call
            payload: Request payload
            
        Returns:
            API response as dictionary
        """
        try:
            response = requests.post(f"{OLLAMA_API_URL}/{endpoint}", json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with Ollama: {e}")
            return {}
            
    def add_document(self, doc_type: str, doc_id: str, file_path: str) -> bool:
        """
        Add a legal document to the system
        
        Args:
            doc_type: Type of document (ipc, constitution, judgments, etc.)
            doc_id: Identifier for the document
            file_path: Path to the document file
            
        Returns:
            True if document was added successfully, False otherwise
        """
        if doc_type not in SUPPORTED_DOCUMENT_TYPES:
            print(f"Error: Unsupported document type '{doc_type}'. Supported types: {', '.join(SUPPORTED_DOCUMENT_TYPES)}")
            return False
            
        try:
            # Read the document content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Create a clean filename
            clean_doc_id = re.sub(r'[^\w\-]', '_', doc_id)
            target_path = self.data_dir / doc_type / f"{clean_doc_id}.txt"
            
            # Save the document
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
            # Update the document index
            self.document_index[doc_type][clean_doc_id] = target_path
            print(f"Document '{doc_id}' added successfully as {doc_type}")
            return True
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    def list_documents(self) -> None:
        """Display a list of all available legal documents"""
        print("\n=== Available Legal Documents ===")
        
        has_documents = False
        for doc_type, documents in self.document_index.items():
            if documents:
                has_documents = True
                print(f"\n{doc_type.upper()}:")
                for doc_id in sorted(documents.keys()):
                    print(f"  - {doc_id}")
                    
        if not has_documents:
            print("\nNo documents found. Use the 'add' command to add legal documents.")
            print("Example: python legal_assistant_lite.py add ipc IPC_1860 path/to/ipc_text.txt")
    
    def get_document_content(self, doc_type: str, doc_id: str) -> Optional[str]:
        """
        Retrieve the content of a specific document
        
        Args:
            doc_type: Type of document to retrieve
            doc_id: ID of the document
            
        Returns:
            Document content as string, or None if not found
        """
        if doc_type not in self.document_index or doc_id not in self.document_index[doc_type]:
            return None
            
        try:
            with open(self.document_index[doc_type][doc_id], 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading document: {e}")
            return None
    
    def search_documents(self, query: str) -> List[Tuple[str, str, str]]:
        """
        Search for documents containing the query term
        
        Args:
            query: Search term
            
        Returns:
            List of matching (doc_type, doc_id, snippet) tuples
        """
        results = []
        query_lower = query.lower()
        
        for doc_type, documents in self.document_index.items():
            for doc_id, file_path in documents.items():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if query_lower in content.lower():
                            # Find a snippet containing the query
                            pos = content.lower().find(query_lower)
                            start = max(0, pos - 100)
                            end = min(len(content), pos + len(query) + 100)
                            
                            # Adjust to word boundaries
                            while start > 0 and content[start] != ' ':
                                start -= 1
                                
                            while end < len(content) and content[end] != ' ':
                                end += 1
                                
                            snippet = f"...{content[start:end]}..."
                            results.append((doc_type, doc_id, snippet))
                except Exception as e:
                    print(f"Error searching document {doc_id}: {e}")
                    
        return results
    
    def ask(self, question: str, context_docs: List[Tuple[str, str]] = None) -> str:
        """
        Ask a legal question to the AI
        
        Args:
            question: The legal question to ask
            context_docs: Optional list of (doc_type, doc_id) tuples to use as context
            
        Returns:
            AI response to the question
        """
        context = ""
        
        # Add document context if specified
        if context_docs:
            for doc_type, doc_id in context_docs:
                doc_content = self.get_document_content(doc_type, doc_id)
                if doc_content:
                    context += f"\n\n--- {doc_type.upper()}: {doc_id} ---\n{doc_content[:5000]}"  # Limit to first 5000 chars
        
        # Prepare the prompt
        system_prompt = """You are a knowledgeable Indian legal assistant specializing in Indian law.
You provide accurate, concise responses based on the legal texts and contexts provided.
Always cite the specific sections or precedents you reference. If you're unsure of something, admit it rather than speculating.
"""

        prompt = f"{question}"
        if context:
            prompt = f"Based on the following legal texts:\n{context}\n\nQuestion: {question}"
            
        # Call Ollama API for generation
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
        
        print("Thinking...")
        response = self._run_ollama_command("generate", payload)
        return response.get("response", "Sorry, I couldn't generate a response.")
    
    def summarize(self, doc_type: str, doc_id: str) -> str:
        """
        Generate a summary of a legal document
        
        Args:
            doc_type: Type of document to summarize
            doc_id: ID of the document
            
        Returns:
            Summary of the document
        """
        doc_content = self.get_document_content(doc_type, doc_id)
        if not doc_content:
            return f"Document not found: {doc_type}/{doc_id}"
            
        # Truncate if necessary to fit model context
        if len(doc_content) > 15000:
            doc_content = doc_content[:15000] + "...[content truncated]"
            
        system_prompt = """You are a legal summarization expert specializing in Indian law.
Provide a concise, well-structured summary of the provided legal text.
Identify key points, principles, and implications. Use clear headings and bullet points where appropriate.
"""

        payload = {
            "model": self.model,
            "prompt": f"Please summarize the following {doc_type} document ({doc_id}):\n\n{doc_content}",
            "system": system_prompt,
            "stream": False
        }
        
        print("Generating summary...")
        response = self._run_ollama_command("generate", payload)
        return response.get("response", "Sorry, I couldn't generate a summary.")
    
    def create_study_notes(self, topics: List[str], doc_references: List[Tuple[str, str]] = None) -> str:
        """
        Generate study notes on legal topics
        
        Args:
            topics: List of legal topics to cover
            doc_references: Optional list of (doc_type, doc_id) tuples to reference
            
        Returns:
            Generated study notes
        """
        context = ""
        
        # Add document context if specified
        if doc_references:
            for doc_type, doc_id in doc_references:
                doc_content = self.get_document_content(doc_type, doc_id)
                if doc_content:
                    context += f"\n\n--- {doc_type.upper()}: {doc_id} ---\n{doc_content[:3000]}"  # Limit to first 3000 chars
        
        topics_str = ", ".join(topics)
        
        system_prompt = """You are an expert Indian legal educator.
Create comprehensive but concise study notes on the requested legal topics.
Include key principles, relevant sections of law, important case precedents, and practical applications.
Organize the notes with clear headings, bullet points, and examples for better understanding.
"""

        prompt = f"Create detailed study notes on the following Indian legal topics: {topics_str}"
        if context:
            prompt += f"\n\nReference the following legal texts in your notes:\n{context}"
            
        # Call Ollama API for generation
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
        
        print("Creating study notes...")
        response = self._run_ollama_command("generate", payload)
        return response.get("response", "Sorry, I couldn't generate study notes.")

def list_available_models() -> None:
    """List recommended models for legal tasks"""
    print("\n=== Recommended Models for Indian Legal Assistant ===")
    
    print("\nHigh Quality (Best results, most resource-intensive):")
    for model in RECOMMENDED_MODELS["high_quality"]:
        print(f"  - {model}")
    
    print("\nBalanced (Good balance of quality and performance):")
    for model in RECOMMENDED_MODELS["balanced"]:
        print(f"  - {model}")
    
    print("\nLightweight (Fastest, least resource-intensive):")
    for model in RECOMMENDED_MODELS["lightweight"]:
        print(f"  - {model}")
    
    # Check which models are actually available in Ollama
    try:
        response = requests.get(f"{OLLAMA_API_URL}/tags")
        if response.status_code == 200:
            available_models = [model["name"] for model in response.json().get("models", [])]
            print("\nCurrently installed models on your system:")
            if available_models:
                for model in available_models:
                    print(f"  - {model}")
            else:
                print("  No models found. Use 'ollama pull <model>' to download models.")
    except:
        print("\nCould not connect to Ollama to check available models.")
    
    print("\nRecommendation: For legal tasks, 'mistral' provides a good balance of accuracy and performance.")
    print("For optimal results on complex legal reasoning, consider using 'llama3:70b' if your system supports it.")

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description="Legal Assistant Lite - AI-powered Indian legal research assistant")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--model-type", type=str, choices=["high_quality", "balanced", "lightweight"], 
                        help="Use recommended models of specified type")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add document command
    add_parser = subparsers.add_parser("add", help="Add a legal document")
    add_parser.add_argument("doc_type", choices=SUPPORTED_DOCUMENT_TYPES, help="Type of document")
    add_parser.add_argument("doc_id", help="Identifier for the document")
    add_parser.add_argument("file_path", help="Path to the document file")
    
    # List documents command
    subparsers.add_parser("list", help="List available legal documents")
    
    # List models command
    subparsers.add_parser("models", help="List recommended models for legal tasks")
    
    # Search documents command
    search_parser = subparsers.add_parser("search", help="Search for documents containing a term")
    search_parser.add_argument("query", help="Search term")
    
    # Ask command
    ask_parser = subparsers.add_parser("ask", help="Ask a legal question")
    ask_parser.add_argument("question", help="The legal question to ask")
    ask_parser.add_argument("--context", "-c", nargs=2, action="append", metavar=("DOC_TYPE", "DOC_ID"),
                           help="Document to use as context (can be specified multiple times)")
    
    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Generate a summary of a legal document")
    summarize_parser.add_argument("doc_type", choices=SUPPORTED_DOCUMENT_TYPES, help="Type of document")
    summarize_parser.add_argument("doc_id", help="Identifier for the document")
    
    # Study notes command
    study_parser = subparsers.add_parser("study", help="Generate study notes on legal topics")
    study_parser.add_argument("topics", nargs="+", help="Legal topics to cover")
    study_parser.add_argument("--reference", "-r", nargs=2, action="append", metavar=("DOC_TYPE", "DOC_ID"),
                             help="Document to reference (can be specified multiple times)")
    
    # Interactive mode
    subparsers.add_parser("interactive", help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Handle model-type selection if specified
    model = args.model
    if args.model_type:
        if RECOMMENDED_MODELS.get(args.model_type) and RECOMMENDED_MODELS[args.model_type]:
            model = RECOMMENDED_MODELS[args.model_type][0]  # Use first model in the category
            print(f"Using recommended {args.model_type} model: {model}")
    
    # Initialize the legal assistant
    assistant = LegalAssistantLite(model=model)
    
    # Process commands
    if args.command == "add":
        assistant.add_document(args.doc_type, args.doc_id, args.file_path)
        
    elif args.command == "list":
        assistant.list_documents()
        
    elif args.command == "models":
        list_available_models()
        
    elif args.command == "search":
        results = assistant.search_documents(args.query)
        if results:
            print(f"\nFound {len(results)} matching documents:")
            for doc_type, doc_id, snippet in results:
                print(f"\n{doc_type.upper()}: {doc_id}")
                print(f"Snippet: {snippet}")
        else:
            print("No matching documents found.")
            
    elif args.command == "ask":
        context_docs = args.context if args.context else []
        response = assistant.ask(args.question, context_docs)
        print("\nResponse:")
        print(response)
        
    elif args.command == "summarize":
        summary = assistant.summarize(args.doc_type, args.doc_id)
        print("\nSummary:")
        print(summary)
        
    elif args.command == "study":
        references = args.reference if args.reference else []
        notes = assistant.create_study_notes(args.topics, references)
        print("\nStudy Notes:")
        print(notes)
        
    elif args.command == "interactive":
        run_interactive_mode(assistant)
        
    else:
        # If no command is specified, show help
        parser.print_help()

def run_interactive_mode(assistant: LegalAssistantLite) -> None:
    """
    Run an interactive session with the legal assistant
    
    Args:
        assistant: Initialized LegalAssistantLite instance
    """
    # Determine model quality category
    model_category = "unknown"
    for category, models in RECOMMENDED_MODELS.items():
        if assistant.model in models or assistant.model.split(':')[0] in models:
            model_category = category
            break
    
    model_description = ""
    if model_category == "high_quality":
        model_description = "(High quality - optimized for accuracy)"
    elif model_category == "balanced":
        model_description = "(Balanced - good mix of speed and quality)"
    elif model_category == "lightweight":
        model_description = "(Lightweight - optimized for speed)"
    
    print(f"""
=== Legal Assistant Lite Interactive Mode ===
Using model: {assistant.model} {model_description}

Available commands:
- ask <question>: Ask a legal question
- context <doc_type> <doc_id>: Add a document to the current context
- clear: Clear the current context
- list: List available documents
- models: Show recommended models for legal tasks
- search <term>: Search for documents containing a term
- summarize <doc_type> <doc_id>: Summarize a document
- study <topic1> [topic2...]: Generate study notes on topics
- model <model_name>: Switch to a different model
- exit/quit: Exit interactive mode

""")

    context_docs = []
    
    while True:
        if context_docs:
            context_str = ", ".join([f"{t}:{i}" for t, i in context_docs])
            cmd_input = input(f"[Context: {context_str}] > ")
        else:
            cmd_input = input("> ")
            
        if not cmd_input.strip():
            continue
            
        parts = cmd_input.split(maxsplit=1)
        cmd = parts[0].lower()
        
        if cmd in ["exit", "quit"]:
            break
            
        elif cmd == "ask":
            if len(parts) < 2:
                print("Usage: ask <question>")
                continue
                
            question = parts[1]
            response = assistant.ask(question, context_docs)
            print("\nResponse:")
            print(response)
            
        elif cmd == "context":
            if len(parts) < 2:
                print("Usage: context <doc_type> <doc_id>")
                continue
                
            context_parts = parts[1].split()
            if len(context_parts) < 2:
                print("Usage: context <doc_type> <doc_id>")
                continue
                
            doc_type, doc_id = context_parts[0], context_parts[1]
            if doc_type not in assistant.document_index or doc_id not in assistant.document_index[doc_type]:
                print(f"Document not found: {doc_type}/{doc_id}")
                continue
                
            context_docs.append((doc_type, doc_id))
            print(f"Added {doc_type}/{doc_id} to context")
            
        elif cmd == "clear":
            context_docs = []
            print("Context cleared")
            
        elif cmd == "list":
            assistant.list_documents()
            
        elif cmd == "models":
            list_available_models()
            
        elif cmd == "model":
            if len(parts) < 2:
                print("Usage: model <model_name>")
                continue
                
            new_model = parts[1]
            try:
                response = requests.get(f"{OLLAMA_API_URL}/tags")
                if response.status_code == 200:
                    available_models = [model["name"] for model in response.json().get("models", [])]
                    
                    if new_model in available_models:
                        assistant.model = new_model
                        print(f"Switched to model: {new_model}")
                    else:
                        print(f"Model '{new_model}' not found. Available models:")
                        for model in available_models:
                            print(f"  - {model}")
                        pull_model = input(f"Do you want to pull '{new_model}' now? (y/n): ")
                        if pull_model.lower() == 'y':
                            print(f"Pulling model '{new_model}'. This may take a while...")
                            assistant._run_ollama_command("pull", {"name": new_model})
                            assistant.model = new_model
                            print(f"Switched to model: {new_model}")
            except:
                print("Could not connect to Ollama to check available models.")
            
        elif cmd == "search":
            if len(parts) < 2:
                print("Usage: search <term>")
                continue
                
            query = parts[1]
            results = assistant.search_documents(query)
            if results:
                print(f"\nFound {len(results)} matching documents:")
                for doc_type, doc_id, snippet in results:
                    print(f"\n{doc_type.upper()}: {doc_id}")
                    print(f"Snippet: {snippet}")
            else:
                print("No matching documents found.")
                
        elif cmd == "summarize":
            if len(parts) < 2:
                print("Usage: summarize <doc_type> <doc_id>")
                continue
                
            summary_parts = parts[1].split()
            if len(summary_parts) < 2:
                print("Usage: summarize <doc_type> <doc_id>")
                continue
                
            doc_type, doc_id = summary_parts[0], summary_parts[1]
            summary = assistant.summarize(doc_type, doc_id)
            print("\nSummary:")
            print(summary)
            
        elif cmd == "study":
            if len(parts) < 2:
                print("Usage: study <topic1> [topic2...]")
                continue
                
            topics = parts[1].split()
            notes = assistant.create_study_notes(topics, context_docs)
            print("\nStudy Notes:")
            print(notes)
            
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'exit' to quit")
            
if __name__ == "__main__":
    main()