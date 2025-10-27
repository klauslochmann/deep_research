
import operator
import time
import sys
from langchain.chat_models import init_chat_model
from langgraph.types import Send
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from tavily import TavilyClient

llm_nano = ChatOpenAI(model='gpt-4.1-nano')
llm_mini = ChatOpenAI(model='gpt-4.1-mini')


class AgentState(TypedDict):
    """the state of the main agent"""
    question: str
    search_terms: list[str]
    search_results: list[str]
    search_summaries: Annotated[list[str], operator.add]
    summary: str
    is_sufficient: bool
    iteration: int

def DeriveQuestions(state: AgentState) -> AgentState:

    print("Going to derive search terms from question.")

    reply = llm_nano.invoke("You are a research assistant, who helps answering a question.\nBased on the question, derive search terms that you want to search online. Always return an JSON array of up to five strings.\nQuestion: " + state["question"])

    p = JsonOutputParser()
    state["search_terms"] = p.parse(reply.text)

    print("Determined search terms: " + ", ".join(state["search_terms"]))

    return state

def SearchOnline(state:AgentState) -> dict:
    print("Going to search via Tavily")
    tavily_client = TavilyClient()
    search_results = []
    seen_urls = set()
    for search_term in state["search_terms"]:
        search_result = tavily_client.search(search_term)
        for x in search_result["results"]:
            url = x["url"]
            if url not in seen_urls:
                seen_urls.add(url)
                search_results.append("# Webpage: " + x["url"] + "\n" + x["content"] + "\n")
    print("The search produced " + str(len(search_results)) + " search results.")
    return {"search_results": search_results}

def continue_to_summaries(state: AgentState):
    print("Going to summarize all found search results.")
    return [
        Send(
            "generate_summary", 
            {
                "question": state["question"],
                "result_text": result_text 
            }
        ) 
        for result_text in state['search_results']
    ]

def GenerateSummary(state:dict) -> AgentState:
    reply = llm_nano.invoke("Answer the following question using only the input given below. Also state the URL of the source at the end.\nQuestion: " + state["question"] + "\n\nHere the input text:\n" + state["result_text"]).text
    
    return {"search_summaries": [reply] }


def WriteFinalAnswer(state:AgentState) -> AgentState:
    print("Going to write final answer to the question.")
    prompt = "Write one concise answer to the following question, using the input from below. State the used sources. If multiple sources show the same answer, then only cite the most credible one. Question: " + state["question"] + "\n\nHere the input text:\n" + "\n\n\n".join(state["search_summaries"])
    print(prompt)
    for x in state["search_summaries"]:
        prompt = prompt + x + "\n\n"
    reply = llm_mini.invoke(prompt).text
    
    state["summary"] = reply
    state["iteration"] = state["iteration"] + 1
    print("The final summary is: " + reply)
    return state

def AssessQualityOfResult(state:AgentState) -> AgentState:
    print("Going to assess the quality.")
    question = state["question"]
    summary = state["summary"]
    search_terms = ", ".join(state["search_terms"])
    prompt=(
         f"You are a research assistant, which gives precise and short answers to a question.\n"
         f"The user asked the question: <question>{question}</question>\n"
         f"The answer you generated is: <answer>{summary}</answer>\n"
         f"Please assess if the answer to the question is good enough. Return a JSON with three fields:"
         f" * sufficient: true/false. "
         f" * search_terms: [list of strings]. "
         f" * reason: state why it is not sufficient. \n"
         f"If the answer is sufficient reply with true. Otherwise reply with false and give a list of "
         f"terms that should be used for an online search to answer the question properly. Previously the "
         f"following terms have already been used: {search_terms}. Make sure to use different "
         f"search term this time.")
    print(prompt)
    reply = llm_mini.invoke(prompt)

    p = JsonOutputParser()
    reply_json = p.parse(reply.text)
    print(reply_json)
    if reply_json["sufficient"]:
        state["is_sufficient"] = True
        return state
    else:
        state["is_sufficient"] = False
        state["search_terms"] = reply_json["search_terms"]
        return state

def continue_after_assessment(state: AgentState):
    print("Taking decision after the assessment")
    if state["iteration"] >= 3:
        return [
            Send(
                "output_answer", 
                state
            )
        ]
    elif state["is_sufficient"]:
        return [
            Send(
                "output_answer", 
                state
            )
        ]
    else: 
        return [
            Send(
                "search_online", 
                state
            )
        ]

def OutputAnswer(state:AgentState) -> AgentState:
    return state

workflow = StateGraph(AgentState)

workflow.add_node("derive_questions", DeriveQuestions)
workflow.add_node("search_online", SearchOnline)
workflow.add_node("generate_summary", GenerateSummary)
workflow.add_node("write_final_answer", WriteFinalAnswer)
workflow.add_node("assess_quality", AssessQualityOfResult)
workflow.add_node("output_answer", OutputAnswer)

workflow.add_edge(START, "derive_questions")
workflow.add_edge("derive_questions", "search_online")
workflow.add_conditional_edges("search_online", continue_to_summaries, ["generate_summary"])
workflow.add_edge("generate_summary", "write_final_answer")
workflow.add_edge("write_final_answer", "assess_quality")
workflow.add_conditional_edges("assess_quality", continue_after_assessment, ["search_online", "output_answer"])
workflow.add_edge("output_answer", END)


deep_research_agent = workflow.compile()

output = deep_research_agent.invoke(
    {
        "question": sys.argv[1],
        "search_terms": [],
        "search_results": [],
        "iteration": 0
    }
    )

print("The final summary is: " + output["summary"])

deep_research_agent.get_graph().draw_mermaid_png(output_file_path="output.png")