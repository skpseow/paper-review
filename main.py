import os
from dotenv import load_dotenv

# Ensure we have our modules in path if running from root
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator.graph import build_workflow

def main():
    load_dotenv()
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set. Please update your .env file.")
        return

    print("\n" + "="*50)
    print("      ADVANCED AI PEER REVIEW SUITE       ")
    print("="*50 + "\n")
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Using file path from args: {file_path}")
    else:
        file_path = input("Enter the path to the PDF or TXT file to review: ").strip()
    
    if not os.path.exists(file_path):
        print(f"\nError: File '{file_path}' not found.")
        return
        
    print(f"\n[1/3] Ingesting document '{os.path.basename(file_path)}'...")
    
    app = build_workflow()
    
    # Generate and save State Tree Visualization
    try:
        print("[*] Generating State Tree visualization...")
        png_data = app.get_graph().draw_mermaid_png()
        png_path = os.path.join(os.path.dirname(file_path), "state_tree.png") if os.path.dirname(file_path) else "state_tree.png"
        with open(png_path, "wb") as f:
            f.write(png_data)
        print(f"[*] Saved State Tree to: {png_path}\n")
    except Exception as e:
        print(f"[-] Could not generate State Tree. Ensure dependencies are installed or network is connected. ({e})\n")
        
    # Initialize the LangGraph State
    initial_state = {
        "file_path": file_path,
        "document_sections": {},
        "agent_reports": {},
        "verdict_report": "",
        "error": ""
    }
    
    print("[2/3] Running parallel specialized agents (this may take a few moments)...")
    
    try:
        # App.invoke inherently resolves the graph execution, 
        # handling async concurrency for the 4 parallel agent branches.
        final_state = app.invoke(initial_state)
        
        if final_state.get("error"):
            print(f"\n[!] Workflow error: {final_state['error']}")
            return
            
        print("[3/3] Synthesis Join complete. Generating Verdict...\n")
        print("="*50)
        print("                 VERDICT REPORT")
        print("="*50 + "\n")
        print(final_state["verdict_report"])
        
    except Exception as e:
        print(f"\n[!] Critical execution error: {str(e)}")

if __name__ == "__main__":
    main()
