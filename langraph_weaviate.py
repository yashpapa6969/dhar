

import os

os.environ["OPENAI_API_KEY"] = ''

# ! pip install -U --quiet langchain_community tiktoken langchain-mistralai langchainhub chromadb langchain langgraph tavily-python

import os
os.environ['TAVILY_API_KEY'] = 't'
tavily_api_key = os.getenv("TAVILY_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = ""

"""## Index"""

from langchain.vectorstores import Weaviate
import weaviate

weaviate_api_key = ""
auth_config = weaviate.AuthApiKey(api_key=weaviate_api_key)
weaviate_client = weaviate.Client(
            url="",

            auth_client_secret=auth_config
        )
weaviate_client.schema.delete_all()

weaviate_client.schema.get()

### Build Index
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Set embeddings
from langchain_openai import OpenAIEmbeddings

embedding_function= OpenAIEmbeddings(model="text-embedding-3-small")


# Docs to index
urls = [
"https://www.geeksforgeeks.org/csharp-programming-language/",
 "https://www.geeksforgeeks.org/c-sharp-multithreading/?ref=shm"
]

# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorstore
vectorstore = Weaviate.from_documents(
    documents=doc_splits,
    embedding=embedding_function ,
    client=weaviate_client,
    by_text=False
)

retriever = vectorstore.as_retriever()

doc_splits[0]

weaviate_client.schema.get()

"""## LLMs"""

# from groq import Groq

# os.environ["GROQ_API_KEY"] = ""

# groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

### Router
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


# Data model
class web_search(BaseModel):
    """
    The internet. Use web_search for questions that are related to anything else than agents, prompt engineering, and adversarial attacks.
    """

    query: str = Field(description="The query to use when searching the internet.")


class vectorstore(BaseModel):
    """
    A vectorstore containing documents related to agents, prompt engineering, and adversarial attacks. Use the vectorstore for questions on these topics.
    """

    query: str = Field(description="The query to use when searching the vectorstore.")



system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. Otherwise, use web-search."""

# LLM with tool use and preamble
llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0, model="gpt-3.5-turbo",)
structured_llm_router = llm.bind_functions(
[web_search, vectorstore]
)

# Prompt
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

#



### Retrieval Grader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

# LLM with function call
llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0, model="gpt-3.5-turbo")
structured_llm_grader = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader



"""Generate"""

### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser


# Preamble
prompt = hub.pull("rlm/rag-prompt")
# LLM
llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0, model="gpt-3.5-turbo")

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()

# Run


### LLM fallback
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

prompt_template = """You are an assistant for question-answering tasks. Answer the question based upon your knowledge. Use three sentences maximum and keep the answer concise.
Question: {question} \nAnswer:
"""
prompt = PromptTemplate(
    template=prompt_template, input_variables=["question"]
)
# LLM
llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0, model="gpt-3.5-turbo")


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
llm_chain = prompt | llm | StrOutputParser()

# Run

### Hallucination Grader


# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# Preamble
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

# LLM with function call
llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0, model="gpt-3.5-turbo")
structured_llm_grader = llm.with_structured_output(
    GradeHallucinations
)

# Prompt
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader


### Answer Grader


# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# Preamble
system = """You are a grader assessing whether an answer addresses / resolves a question \n
Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""

# LLM with function call
llm = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"], temperature=0, model="gpt-3.5-turbo")
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader

"""## Web Search Tool"""

### Search
# os.environ['TAVILY_API_KEY'] ='<your-api-key>'

from langchain_community.tools.tavily_search import TavilySearchResults

web_search_tool = TavilySearchResults()

"""# Graph

Capture the flow in as a graph.

## Graph state
"""

from typing import List

from typing_extensions import TypedDict


class GraphState(TypedDict):
    """|
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]

"""## Graph Flow"""

import pprint
from langchain.schema import Document


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def llm_fallback(state):
    """
    Generate answer using the LLM w/o vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---LLM Fallback---")
    question = state["question"]
    generation = llm_chain.invoke({"question": question})
    return {"question": question, "generation": generation}


def generate(state):
    """
    Generate answer using the vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    if not isinstance(documents, list):
        documents = [documents]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)

    return {"documents": web_results, "question": question}


### Edges ###


def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    # Fallback to LLM or raise error if no decision
    if 'function_call' not in source.additional_kwargs:
        print("---ROUTE QUESTION TO LLM---")
        return "llm_fallback"
    if len(source.additional_kwargs['function_call']) == 0:
        raise "Router could not decide source"

    # Choose datasource
    datasource = source.additional_kwargs['function_call']["name"]
    if datasource == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    else:
        print("---ROUTE QUESTION TO LLM---")
        return "vectorstore"


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, WEB SEARCH---")
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint.pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

"""## Build Graph"""

import pprint

from langgraph.graph import END, StateGraph, START

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # rag
workflow.add_node("llm_fallback", llm_fallback)  # llm

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
        "llm_fallback": "llm_fallback",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "web_search": "web_search",
        "generate": "generate",
    },
)
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",  # Hallucinations: re-generate
        "not useful": "web_search",  # Fails to answer question: fall-back to web-search
        "useful": END,
    },
)
workflow.add_edge("llm_fallback", END)

# Compile
app = workflow.compile()

# # Run
# inputs = {
#     "question": "What player are the Bears expected to draft first in the 2024 NFL draft?"
# }
# for output in app.stream(inputs):
#     for key, value in output.items():
#         # Node
#         pprint.pprint(f"Node '{key}':")
#         # Optional: print full state at each node
#     pprint.pprint("\n---\n")

# # Final generation
# pprint.pprint(value["generation"])

# Run



def process_input(question):
    inputs = {"question": question}
    final_output = ""
    node_sequence = []

    for output in app.stream(inputs):
        for key, value in output.items():
            node_sequence.append(key)
        final_output = value.get("generation", "")

    node_output = "Node Sequence:\n\n"
    node_output += " -> ".join(node_sequence)

    return final_output, node_output
def rag_pipeline(question: str):

    inputs = {"question": question}
    final_output = ""
    for output in app.stream(inputs):
        for key, value in output.items():
            if "generation" in value:
                final_output = value["generation"]
    return final_output
from flask import Flask, request, jsonify
from flask_cors import CORS

app1 = Flask(__name__)
CORS(app1)
@app1.route('/rag/response', methods=['POST'])
def post():
    input_data = request.get_json()  # Get JSON data from request
    print("input_data====")
    print(input_data)

    inputs = {"question": input_data["question"]}
    print(inputs)
    final_output = ""
    for output in app.stream(inputs):
        for key, value in output.items():
            if "generation" in value:
                final_output = value["generation"]
    return jsonify(final_output)




if __name__ == '__main__':
    app1.run(host='0.0.0.0', port=5000, debug=True)

