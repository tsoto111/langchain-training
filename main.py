from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

'''
# Tool Method: This was an import utilizing tavily client to create a tool method to trigger a web search 
#              based off of a prompt using an AI agent.
from tavily import TavilyClient
'''

# Tavily langchain client implementation written by the Tavily team!
from langchain_tavily import TavilySearch

load_dotenv()

'''
# Tool Method: Initialization of the tavily client
tavilyClient = TavilyClient()
'''

search_prompt = 'Search for 3 job postings for an ai engineer using LangChain in Tulsa Oklahoma area on LinkedIn and list their details.'

'''
# Tool Method: Implementation of an agent tool function which Agent can choose
#              to use based off of the provided prompt. Note that the comments
#              in the method are a required part of the tool in order for the 
#              Agent to reason over using this method or not.
@tool
def search(query: str) -> str:
    \'''
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    \'''
    print(f"Searching for {query}")
    return tavilyClient.search(query=query)
'''

llm = ChatOpenAI(model='gpt-5')

'''
# Tool Method: How we add our search tool as optional methods for the Agent to use.
tools = [search]
'''

tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages":HumanMessage(content=search_prompt)})
    print()
    print(result['messages'][-1].text)


if __name__ == "__main__":
    main()
