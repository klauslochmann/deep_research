
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

    print("Going to derive search terms from question.")

    reply = llm_nano.invoke("You are a research assistant, who helps answering a question.\nBased on the question, derive search terms that you want to search online. Always return an JSON array of up to five strings.\nQuestion: " + state["question"])

    p = JsonOutputParser()
    state["search_terms"] = p.parse(reply.text)

    print("Determined " + str(len(state["search_terms"])) + " search terms.")

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
    reply = llm_nano.invoke("Answer the following question using only the input given below. State the URL of the source at the end.\nQuestion: " + state["question"] + "\n\nHere the input text:\n" + state["result_text"]).text
    
    return {"search_summaries": [reply] }


def WriteFinalAnswer(state:AgentState) -> AgentState:
    print("Going to write final answer to the question.")
    prompt = "Write one concise answer to the following question, using the input from below. State the sources that you used. Question: " + state["question"] + "\n\nHere the input text:\n" + "\n\n".join(state["search_summaries"])
    for x in state["search_summaries"]:
        prompt = prompt + x + "\n\n"
    reply = llm_nano.invoke(prompt).text
    
    state["summary"] = reply
    return state

workflow = StateGraph(AgentState)

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
