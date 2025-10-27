
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


class AgentState(TypedDict):
    """the state of the main agent"""
    question: str
    search_terms: list[str]
    search_results: list[str]
    search_summaries: Annotated[list[str], operator.add]
    summary: str

def DeriveQuestions(state: AgentState) -> AgentState:
    reply = llm_nano.invoke("You are a research assistant, who helps answering a question.\nBased on the question, derive search terms that you want to search online. Always return an JSON array of up to five strings.\nQuestion: " + state["question"]).text
    
    p = JsonOutputParser()
    state["search_terms"] = p.parse(reply)

    return state

def SearchOnline(state:AgentState) -> dict:
    tavily_client = TavilyClient()
    search_results = []
    for search_term in state["search_terms"]:
        search_result = tavily_client.search(search_term)
        search_results.append(search_result["results"][0]["content"])
    return {"search_results": search_results}

def continue_to_summaries(state: AgentState):
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
    reply = llm_nano.invoke("Answer the following question using only the input given below.\nQuestion: " + state["question"] + "\n\nHere the input text:\n" + state["result_text"]).text
    
    return {"search_summaries": [reply] }


def WriteFinalAnswer(state:AgentState) -> AgentState:
    #print("Final Answer, based on " + str(len(state["search_summaries"])) + " summaries.")
    prompt = "Write one concise answer to the following question, using the input from below. Question: " + state["question"] + "\n\nHere the input text:\n"
    for x in state["search_summaries"]:
        prompt = prompt + x + "\n\n"
    reply = llm_nano.invoke(prompt).text
    
    state["summary"] = reply
    return state

workflow = StateGraph(
    AgentState)

workflow.add_node("derive_questions", DeriveQuestions)
workflow.add_node("search_online", SearchOnline)
workflow.add_node("generate_summary", GenerateSummary)
workflow.add_node("write_final_answer", WriteFinalAnswer)



workflow.add_edge(START, "derive_questions")
workflow.add_edge("derive_questions", "search_online")
workflow.add_conditional_edges("search_online", continue_to_summaries, ["generate_summary"])
workflow.add_edge("generate_summary", "write_final_answer")
workflow.add_edge("write_final_answer", END)

deep_research_agent = workflow.compile()

output = deep_research_agent.invoke(
    {
        "question": sys.argv[1],
        "search_terms": [],
        "search_results": []
        }
    )

print("The final summary is: " + output["summary"])