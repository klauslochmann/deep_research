
import operator
import time
import sys
import asyncio
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

from langchain_core.tools import tool

from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from tavily import TavilyClient

llm_nano = ChatOpenAI(model='gpt-4.1-nano')
llm_mini = ChatOpenAI(model='gpt-4.1-mini')

from langchain.agents import create_agent


class AgentState(TypedDict):
    """the state of the main agent"""
    user_input: str
    research_outline: str
    final_report: str

@tool
def searchEngine(topic:str) -> str:
    """Invokes a search engine to search the web and responds with the search results."""
    print("The search engine got called: " + topic)
    tavily_client = TavilyClient()
    search_results = ""
    search_result = tavily_client.search(topic)
    for x in search_result["results"]:
        search_results = search_results + "# Webpage: " + x["url"] + "\n" + x["content"] + "\n\n"
    return search_results

prompt_supervisor = """You are a research agent, writing research reports. As input take the 
    research outline given here:
    <research_outline>
        {research_outline}
    </research_outline>
    Write a detailed and facutally correct report about it. 
    It is very important to provide proper citations for all the content you create. In the 
    text please refer to sources in the style [x]. At the end of the report create a list of 
    all sources with the URLs of the web pages.
    
    For writing this research report you got tool for assisting you. Use those tools as necessary."""

prompt_derive_research_outline = """Your are a research agent, writing research reports. The
    user will provide an idea for a research. It is your task to take that idea and formulate
    a full research outline. The research outline should be specific and describe the research
    question in detail. It is intended as a basis for an automatic research based on AI."""



async def DeriveResearchOutline(state: AgentState):
    """This function derives the research outline based on the user input."""
    agent = create_agent(model=llm_mini, tools=[], system_prompt=prompt_derive_research_outline)
    result = await agent.ainvoke({"messages": [ {"role": "user", "content": state["user_input"] }]})
    return {"research_outline": result["messages"][-1].content }


async def ResearchSupervisor(state: AgentState):
    """This function is the research supervisor."""
    prompt = prompt_supervisor.format(
        research_outline=state["research_outline"]
    )
    print("The system prompt for the supervisor is: " + prompt)
    agent = create_agent(model=llm_mini, tools=[searchEngine], system_prompt=prompt)
    result = await agent.ainvoke({"messages": [ {"role": "user", "content": state["user_input"] }]})
    return {"final_report": result["messages"][-1].content }

workflow = StateGraph(AgentState)

workflow.add_node("derive_research_outline", DeriveResearchOutline)
workflow.add_node("research_supervisor", ResearchSupervisor)


workflow.add_edge(START, "derive_research_outline")
workflow.add_edge("derive_research_outline", "research_supervisor")
workflow.add_edge("research_supervisor", END)

deep_research_agent = workflow.compile()

output = asyncio.run(deep_research_agent.ainvoke(
    {
        "user_input": sys.argv[1]
    }
    ))

print("The final is: " + output["final_report"])
