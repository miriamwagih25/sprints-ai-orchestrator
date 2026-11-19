"""Show the full error behind the 400 from the SQL agent."""
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from openai import BadRequestError

from src.config import get_llm
from src.tools.db_queries import describe_learner_db, query_learner_db

agent = create_react_agent(get_llm(), tools=[describe_learner_db, query_learner_db])
try:
    out = agent.invoke({"messages": [HumanMessage("How many learners are there in total?")]})
    print(out["messages"][-1].content)
except BadRequestError as e:
    print(e.response.text)