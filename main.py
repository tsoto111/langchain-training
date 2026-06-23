import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorStore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

retriever = vectorStore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based on only the following context:

    {context}

    Question: {question}
    
    Provide a detailed answer:""")

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL (LangChain Expression Language).

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format the prompt with context and question
    message = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(message)

    # Step 5: Return the content
    return response.content

if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "What is Pinecone in machine learning?"

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 70)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)