# Deep Research Agent to answer a question

## Setup the environment

```
uv venv
source .venv/bin/activate
uv sync
```

##
The the environment variables:
```
export OPENAI_API_KEY=<your key here>
export TAVILY_API_KEY=<your key here>
```

## Run langgraph

```
python3 src/deep_research_agent.py "Ask any question here?"
```
