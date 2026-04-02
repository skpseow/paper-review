import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

class PeerReviewAgent:
    """
    A unified wrapper for a specialized LLM agent reviewing a specific document section.
    """
    def __init__(self, name: str, role_description: str, model_name: str = "gpt-4o"):
        self.name = name
        self.role_description = role_description
        # Intentionally low temperature for analytical peer-review tasks
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        
    def review(self, document_section: str) -> str:
        """
        Runs the specialized agent against its designated section of the document.
        """
        system_msg = SystemMessage(content=f"You are the {self.name}. Your role is: {self.role_description}")
        human_msg = HumanMessage(content=f"Please review the following section of the document according to your role. Provide a structured critique:\n\n{document_section}")
        
        try:
            response = self.llm.invoke([system_msg, human_msg])
            return response.content
        except Exception as e:
            return f"Error executing {self.name}: {str(e)}"

def create_agents():
    """Instantiate and return the enhanced specialized peer review agents."""
    
    paper_summarizer = PeerReviewAgent(
        name="Paper Summarizer",
        role_description=(
            "Read the introductory section and provide a concise, high-level summary of the paper. "
            "Explain the core research question, what the study is trying to achieve, and its general context. "
            "Write this so that a reader who has not seen the document can fully grasp its premise before reading the critique."
        )
    )
    
    methodology_classifier = PeerReviewAgent(
        name="Methodology Classifier",
        role_description=(
            "Analyze the methods and overall structure to determine whether the paper uses a qualitative, "
            "quantitative, or mixed-methods methodology. Briefly explain the core study design, and state specifically "
            "what kind of data representations (e.g., statistical tables vs. thematic codes) should be expected."
        )
    )
    
    results_analyst = PeerReviewAgent(
        name="Results Analyst",
        role_description=(
            "Evaluate the 'Results' or 'Findings' section. Adapt to the methodology paradigm. "
            "If quantitative: verify statistical consistency, check for p-hacking or unsupported stats. "
            "If qualitative: evaluate thematic coding reliability, saturation, triangulation, and narrative flow. "
            "Address: Does it help readers make sense of the data presented? Does it clearly present the findings "
            "and their connection to the research questions (including data visualizations, excerpts, etc.)?"
        )
    )
    
    methodologist = PeerReviewAgent(
        name="Methodologist",
        role_description=(
            "Evaluate the 'Methods/Approach' section. To what extent does it describe what was done? "
            "Who were the participants/sources, how many, and what did they do? What data was collected and how was it analyzed? "
            "Does it have a sufficiently elaborated description of methods, references to methodological literature, and "
            "explain how they address the research questions? Does it include positionality statements (relationship "
            "between researchers and participants) if appropriate?"
        )
    )
    
    literature_scout = PeerReviewAgent(
        name="Literature Scout",
        role_description=(
            "Analyze the 'Literature Review' or Introduction. "
            "Does it describe the research problem (considering paper type and formulation)? "
            "Does it use prior literature to situate the research and identify gaps or opportunities to refine existing work? "
            "Significance/Grounding: Does it state a problem the community cares about? "
            "Conceptual Framing: Does it identify theory/concepts informing interpretations, and are they defined/connected?"
        )
    )
    
    coherence_editor = PeerReviewAgent(
        name="Coherence Editor",
        role_description=(
            "Review the 'Discussion and Conclusion'. "
            "Does it help readers make sense of the relevance of empirical results and their implications for the field? "
            "Does it conclude with some level of resolution around the motivating research problem? "
            "Does it explicitly outline limitations and future work? "
            "Check for alignment between initial arguments and final claims. Flag overly general/strong conclusions."
        )
    )
    
    return {
        "PaperSummarizer": paper_summarizer,
        "MethodologyClassifier": methodology_classifier,
        "ResultsAnalyst": results_analyst,
        "Methodologist": methodologist,
        "LiteratureScout": literature_scout,
        "CoherenceEditor": coherence_editor
    }
