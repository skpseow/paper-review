import os
from typing import Dict, TypedDict, Annotated, Any
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ingestion.parser import DocumentParser
from agents.specialized import create_agents

def dictionary_reducer(left: dict, right: dict) -> dict:
    """Helper to merge the agent reports dictionary during graph execution."""
    if not left: left = {}
    if not right: right = {}
    return {**left, **right}

# Define the state matching our architecture visualization
class PeerReviewState(TypedDict):
    file_path: str
    document_sections: Dict[str, str]
    # Reducer allows parallel nodes to merge their dict updates instead of overwriting
    agent_reports: Annotated[Dict[str, str], dictionary_reducer]
    verdict_report: str
    error: str

def ingest_document(state: PeerReviewState):
    """Parses and segments the document."""
    parser = DocumentParser(state["file_path"])
    try:
        sections = parser.segment_document()
        return {"document_sections": sections}
    except Exception as e:
        return {"error": str(e)}

def run_paper_summarizer(state: PeerReviewState):
    agent = create_agents()["PaperSummarizer"]
    # We pass the literature section as it contains the abstract & introduction
    section_text = state["document_sections"].get("literature_review", "")
    if not section_text:
        return {"agent_reports": {"PaperSummarizer": "No introduction found."}}
        
    report = agent.review(section_text)
    return {"agent_reports": {"PaperSummarizer": report}}

def run_methodology_classifier(state: PeerReviewState):
    agent = create_agents()["MethodologyClassifier"]
    # Provide the methods section, which is typically sufficient to classify methodology
    section_text = state["document_sections"].get("methods", "")
    if not section_text:
        return {"agent_reports": {"MethodologyClassifier": "No Methods section found to classify methodology."}}
        
    report = agent.review(section_text)
    return {"agent_reports": {"MethodologyClassifier": report}}

def run_results_analyst(state: PeerReviewState):
    agent = create_agents()["ResultsAnalyst"]
    section_text = state["document_sections"].get("results", "")
    if not section_text:
        return {"agent_reports": {"ResultsAnalyst": "No Results section found."}}
        
    report = agent.review(section_text)
    return {"agent_reports": {"ResultsAnalyst": report}}

def run_methodologist(state: PeerReviewState):
    agent = create_agents()["Methodologist"]
    section_text = state["document_sections"].get("methods", "")
    if not section_text:
        return {"agent_reports": {"Methodologist": "No Methods section found."}}
        
    report = agent.review(section_text)
    return {"agent_reports": {"Methodologist": report}}

def run_literature_scout(state: PeerReviewState):
    agent = create_agents()["LiteratureScout"]
    section_text = state["document_sections"].get("literature_review", "")
    if not section_text:
        return {"agent_reports": {"LiteratureScout": "No Literature Review found."}}
        
    report = agent.review(section_text)
    return {"agent_reports": {"LiteratureScout": report}}

def run_coherence_editor(state: PeerReviewState):
    agent = create_agents()["CoherenceEditor"]
    section_text = state["document_sections"].get("discussion_conclusion", "")
    if not section_text:
         return {"agent_reports": {"CoherenceEditor": "No Discussion/Conclusion found."}}
         
    report = agent.review(section_text)
    return {"agent_reports": {"CoherenceEditor": report}}

def synthesis_join(state: PeerReviewState):
    """Orchestrator synthesizes the 4 reports into a Verdict Report."""
    if "error" in state and state["error"]:
        return {"verdict_report": f"Pipeline failed during ingestion: {state['error']}"}
        
    reports = state.get("agent_reports", {})
    
    orchestrator_llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    
    system_prompt = (
        "You are the Lead Scientific Orchestrator of an Advanced AI Peer Review Suite. "
        "You will receive 6 specialized reviews of an academic paper. "
        "Your task is to synthesize these into a comprehensive 'Verdict Report'. "
        "Note the Methodology Classifier's findings: if the paper is qualitative, you must "
        "adjust your expectations (e.g., lack of statistical testing is entirely normal). "
        "You MUST actively look for and highlight CONTRADICTIONS between the agents "
        "(e.g., if the Coherence Editor praises the conclusion, but the Results Analyst found the core data findings flawed)."
    )
    
    human_prompt = f"Here are the specialized agent reports:\n\n"
    for agent_name, report in reports.items():
         human_prompt += f"--- {agent_name} Report ---\n{report}\n\n"
         
    human_prompt += "Please provide the final synthesized Verdict Report, clearly sectioning out 'Paper Summary' (extracting insights from the Paper Summarizer), 'Strengths', 'Weaknesses', 'Agent Contradictions', and 'Final Recommendation'."

    try:
        system_msg = SystemMessage(content=system_prompt)
        human_msg = HumanMessage(content=human_prompt)
        response = orchestrator_llm.invoke([system_msg, human_msg])
        return {"verdict_report": response.content}
    except Exception as e:
        return {"verdict_report": f"Orchestrator failed to synthesize: {str(e)}"}

def build_workflow():
    """Builds the parallel data flow architecture visualised in the state tree."""
    workflow = StateGraph(PeerReviewState)
    
    # 1. Ingestion Layer
    workflow.add_node("Ingestion", ingest_document)
    
    # 2. Parallel Processing Agents
    workflow.add_node("PaperSummarizer", run_paper_summarizer)
    workflow.add_node("MethodologyClassifier", run_methodology_classifier)
    workflow.add_node("ResultsAnalyst", run_results_analyst)
    workflow.add_node("Methodologist", run_methodologist)
    workflow.add_node("LiteratureScout", run_literature_scout)
    workflow.add_node("CoherenceEditor", run_coherence_editor)
    
    # 3. Orchestrator Sync Barrier
    workflow.add_node("SynthesisJoin", synthesis_join)
    
    # Connect edges
    workflow.add_edge(START, "Ingestion")
    
    # Branching (parallel)
    workflow.add_edge("Ingestion", "PaperSummarizer")
    workflow.add_edge("Ingestion", "MethodologyClassifier")
    workflow.add_edge("Ingestion", "ResultsAnalyst")
    workflow.add_edge("Ingestion", "Methodologist")
    workflow.add_edge("Ingestion", "LiteratureScout")
    workflow.add_edge("Ingestion", "CoherenceEditor")
    
    # Joining
    workflow.add_edge("PaperSummarizer", "SynthesisJoin")
    workflow.add_edge("MethodologyClassifier", "SynthesisJoin")
    workflow.add_edge("ResultsAnalyst", "SynthesisJoin")
    workflow.add_edge("Methodologist", "SynthesisJoin")
    workflow.add_edge("LiteratureScout", "SynthesisJoin")
    workflow.add_edge("CoherenceEditor", "SynthesisJoin")
    
    workflow.add_edge("SynthesisJoin", END)
    
    return workflow.compile()
