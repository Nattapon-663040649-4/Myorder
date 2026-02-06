import os
import logging
import google.cloud.logging

from callback_logging import log_query_to_model, log_model_response
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.genai import types

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from google.adk.tools import exit_loop

# Initialize Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()

load_dotenv()

# [Technical Constraint] Model selection
model_name = os.getenv("MODEL")

# --- Shared Tools ---

def update_session_state(
    tool_context: ToolContext, key: str, value: str
) -> dict[str, str]:
    """[Technical Constraint] State Management: Store concise findings into session state."""
    current_data = tool_context.state.get(key, [])
    tool_context.state[key] = current_data + [value]
    logging.info(f"[State Update] Key: {key} updated.")
    return {"status": "success"}

def export_verdict_to_txt(
    tool_context: ToolContext,
    folder: str,
    file_name: str,
    text_content: str
) -> dict[str, str]:
    """[Step 4] The Verdict (Output): Save the final neutral report to a .txt file."""
    path = os.path.join(folder, file_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write(text_content)
    return {"status": "success"}

# --- Specialized Agents ---

# [Step 2] Agent A: The Admirer (Parallel Investigation)
the_admirer = Agent(
    name="admirer",
    model=model_name,
    description="Researches significant achievements and positive legacies.",
    instruction="""
    CONTEXT: Analyzing { SUBJECT_TOPIC? }
    
    TASK:
    - [Technical Constraint] Wiki Research: Search Wikipedia specifically for 'achievements', 'success', or 'positive impact'.
    - ALWAYS append 'achievements' or 'contributions' to your search queries.
    - Summarize findings into 3-4 clear bullet points.
    - Use 'update_session_state' to save this summary into 'pos_data'.
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        update_session_state
    ]
)

# [Step 2] Agent B: The Critic (Parallel Investigation)
the_critic = Agent(
    name="critic",
    model=model_name,
    description="Researches controversies, failures, and criticisms.",
    instruction="""
    CONTEXT: Analyzing { SUBJECT_TOPIC? }
    
    TASK:
    - [Technical Constraint] Wiki Research: Search Wikipedia specifically for 'controversies', 'failures', or 'criticisms'.
    - ALWAYS append 'controversy' or 'criticism' to your search queries.
    - Summarize findings into 3-4 clear bullet points.
    - Use 'update_session_state' to save this summary into 'neg_data'.
    """,
    tools=[
        LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())),
        update_session_state
    ]
)

# [Step 2] Investigation Setup (Parallel)
investigation_stage = ParallelAgent(
    name="investigation_stage",
    sub_agents=[the_admirer, the_critic]
)

# [Step 3] Agent C: The Judge (Trial & Review)
the_judge = Agent(
    name="judge",
    model=model_name,
    description="Evaluates balance and completeness of the evidence.",
    instruction="""
    EVIDENCE REVIEW:
    Positives (from Admirer): { pos_data? }
    Negatives (from Critic): { neg_data? }

    EVALUATION:
    - [Technical Constraint] Loop Logic: Check if the data is balanced and sufficiently detailed for { SUBJECT_TOPIC? }.
    - If one side is missing or too thin, provide specific feedback for the next search.
    - If the analysis is neutral and complete, you MUST call the 'exit_loop' tool.
    """,
    tools=[exit_loop]
)

# [Step 3] Loop Logic Setup
trial_review_loop = LoopAgent(
    name="trial_review_loop",
    sub_agents=[investigation_stage, the_judge],
    max_iterations=3
)

# [Step 4] The Verdict (Final Output)
verdict_scribe = Agent(
    name="verdict_scribe",
    model=model_name,
    description="Synthesizes findings into a balanced final report.",
    instruction="""
    FINAL EVIDENCE:
    Positives: { pos_data? }
    Negatives: { neg_data? }

    INSTRUCTIONS:
    - Compile a comprehensive and neutral academic report for { SUBJECT_TOPIC? }.
    - Use 'export_verdict_to_txt' to save it to the 'historical_court_reports' directory.
    - Filename: { SUBJECT_TOPIC? }.txt
    
    FORMAT:
    1. Introduction
    2. Achievements & Successes
    3. Controversies & Criticisms
    4. Neutral Conclusion
    """,
    tools=[export_verdict_to_txt]
)

# Main Workflow Setup
court_system_workflow = SequentialAgent(
    name="court_system_workflow",
    description="Main sequence for the historical analysis.",
    sub_agents=[
        trial_review_loop,
        verdict_scribe
    ]
)

# [Step 1] The Inquiry (Entry Point)
inquiry_clerk = Agent(
    name="inquiry_clerk",
    model=model_name,
    description="Entry point for receiving the historical topic.",
    instruction="""
    - Welcome the user to 'The Historical Court'.
    - Ask for a historical figure or event to analyze.
    - Use 'update_session_state' to save the input as 'SUBJECT_TOPIC'.
    - Transfer control to 'court_system_workflow'.
    """,
    tools=[update_session_state],
    sub_agents=[court_system_workflow]
)

root_agent = inquiry_clerk