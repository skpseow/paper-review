import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from orchestrator.graph import build_workflow

load_dotenv()

st.set_page_config(page_title="AI Peer Review Suite", page_icon="📜", layout="wide")

st.title("📜 Advanced AI Peer Review Suite")
st.markdown("Upload a research paper (PDF or TXT) and let our specialized AI Swarm analyze it across multiple dimensions in parallel!")

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("⚠️ OPENAI_API_KEY environment variable is not set in your .env file!")
    st.stop()

uploaded_file = st.file_uploader("Upload Paper", type=["pdf", "txt"])

if uploaded_file is not None:
    if st.button("Run Peer Review"):
        with st.spinner("Ingesting document and generating State Tree..."):
            # Save uploaded string/bytes to a temp file
            extension = uploaded_file.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
                tmp.write(uploaded_file.getvalue())
                file_path = tmp.name
                
            try:
                app_graph = build_workflow()
                
                png_data = app_graph.get_graph().draw_mermaid_png()
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(png_data, caption="LangGraph State Tree Workflow")
                    
                with col2:
                    st.info("Running parallel specialized agents to generate evaluation reports...")
                    
                    initial_state = {
                        "file_path": file_path,
                        "document_sections": {},
                        "agent_reports": {},
                        "verdict_report": "",
                        "error": ""
                    }
                    
                    with st.spinner("Agents are reading and evaluating (this usually takes 30-60 seconds)..."):
                        final_state = app_graph.invoke(initial_state)
                        
                    if final_state.get("error"):
                        st.error(f"Workflow error: {final_state['error']}")
                    else:
                        st.success("Synthesis Join complete!")
                        st.markdown("### 🏆 Final Verdict Report")
                        st.markdown(final_state["verdict_report"])
                        
                        st.markdown("---")
                        st.markdown("### 🔍 Individual Agent Reports")
                        for agent, report in final_state.get("agent_reports", {}).items():
                            with st.expander(f"{agent} Report"):
                                st.markdown(report)
                                
            except Exception as e:
                st.error(f"Critical execution error: {str(e)}")
            finally:
                # Clean up temp file
                if os.path.exists(file_path):
                    os.remove(file_path)
