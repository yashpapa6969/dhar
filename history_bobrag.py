import weaviate
import weaviate.classes as wvc
import os
import requests
import json
import pandas as pd
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from weaviate.classes.query import Filter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from typing import List

from typing_extensions import TypedDict
from langchain.schema import Document

from langgraph.graph import END, StateGraph, START
from pprint import pprint

from dotenv import load_dotenv
from flask import Flask, request, jsonify
load_dotenv()

app_flask = Flask(__name__)
# Best practice: store your credentials in environment variables
# Access the API keys
# Access the environment variables correctly
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ['LANGCHAIN_ENDPOINT'] = os.getenv("LANGCHAIN_ENDPOINT")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

# Optionally, retrieve additional environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
wcd_api_key = os.getenv("WCD_API_KEY")
wcd_url = os.getenv("WCD_URL")
# client = weaviate.connect_to_weaviate_cloud(
#     cluster_url=wcd_url,                                    # Replace with your Weaviate Cloud URL
#     auth_credentials=wvc.init.Auth.api_key(wcd_api_key),    # Replace with your Weaviate Cloud key
#     headers={"X-OpenAI-Api-Key": openai_api_key}            # Replace with appropriate header key/value pair for the required API
# )

auth_config = weaviate.AuthApiKey(api_key=wcd_api_key)
weaviate_client = weaviate.Client(
    url=wcd_url,                                    # Replace with your Weaviate Cloud URL
    auth_client_secret=auth_config,    # Replace with your Weaviate Cloud key
    additional_headers={"X-OpenAI-Api-Key": openai_api_key}            # Replace with appropriate header key/value pair for the required API
)

embedding= OpenAIEmbeddings(model="text-embedding-3-small")

hist_retriever = WeaviateHybridSearchRetriever(
    client=weaviate_client,
    index_name="Bob",  # Your collection name
    text_key="response",  # The field you want to use as the main text for retrieval
    attributes=["phone", "query","response"],# Additional attributes to retrieve # Set to True if you want to create the schema if it doesn't exist
    alpha=0.6,
    k=3
    )
retriever = WeaviateHybridSearchRetriever(
    client=weaviate_client,
    index_name="Bobfull",  # Your collection name
    text_key="chatbotInstructions",  # The field you want to use as the main text for retrieval
    attributes=["topic", "userQueryExample","chatbotInstructions","sampleResponse"],# Additional attributes to retrieve # Set to True if you want to create the schema if it doesn't exist
    alpha=0.6,
    k=5
    )
# Data model
class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "llm-fallback"] = Field(
        ...,

        description="Given a user question choose to route it to web search or a vectorstore.",
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

# Prompt
system = """You are an expert at routing a user question to a vectorstore or LLm Fallback(use llm knowledge).
The vectorstore contains documents related to Bank user's Query, Financial documents, User Guide on Bank functionality.
Use the vectorstore for questions on these topics. Otherwise, if the questions are generic which can be answered using LLM knowledge use llm-fallback."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# # Prompt
prompt_template= """You are an assistant for question-answering tasks specifically targeted to the banking and finance sector. Use the following pieces of retrieved context and historical context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question} 

Context: {context}

Historical Context: {historicalcontext}

Answer:"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["question","context","historicalcontext"]
)
# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader

# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt
system = """You are a grader assessing whether an answer addresses / resolves a question \n
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader

### Question Re-writer

# LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

# Prompt
system = """You a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

question_rewriter = re_write_prompt | llm | StrOutputParser()

prompt_template = """You are a helpdesk greeting and reply assistant. Your role is to greet users and provide appropriate follow-up responses. You should respond to greetings with a greeting, such as 'Hello, how may I assist you?'.If a user asks a question, respond with a polite refusal, such as 'I’m sorry, I cannot answer that question.'
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

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    filters: dict
    generation: str
    documents: List[str]
    historicalcontext: List[str]

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
    filters = state["filters"]
    if filters is None:
          hist_documents = hist_retriever.invoke(question)
          context_documents = retriever.invoke(question)
          return {"historicalcontext": hist_documents,"documents":context_documents, "question": question}
    else:
          hist_documents = hist_retriever.invoke(question, where_filter=filters)
          context_documents = retriever.invoke(question)
          return {"historicalcontext": hist_documents,"documents":context_documents, "question": question,"filters" : filters}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    hist_documents = state["historicalcontext"]
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "historicalcontext":hist_documents,"question": question})
    return {"documents": documents,"historicalcontext":hist_documents, "question": question, "generation": generation}


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

    hist_documents =state['historicalcontext']
    filtered_hist_docs = []
    for d in hist_documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_hist_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs,"historicalcontext":filtered_hist_docs, "question": question}


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


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    print("---TRANSFORM QUERY---")
    question = state["question"]
    documents = state["documents"]

    # Re-write question
    better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}

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
    print(source.datasource)
    if source.datasource == "llm-fallback":
        print("---ROUTE QUESTION TO LLM FallBack---")
        return "llm_fallback"
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
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
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
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
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("llm_fallback", llm_fallback)  # llm

# Build graph
workflow.add_conditional_edges(
    START,
    route_question,
    {
        "vectorstore": "retrieve",
         "llm_fallback": "llm_fallback",
    },
)
workflow.add_edge("llm_fallback", END)
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# # Compile
# app = workflow.compile()

# def stream_app(question, phone_number=None):
#     # Construct the input dictionary
#     inputs = {
#         "question": question
#     }
    
#     # If a phone number is provided, construct the filters
#     if phone_number:
#         inputs["filters"] = {
#             "path": ["phone"],
#             "operator": "Equal",
#             "valueString": phone_number
#         }
    
#     # Simulating the streaming of outputs
#     for output in app.stream(inputs):
#         for key, value in output.items():
#             # Node
#             pprint(f"Node '{key}':")
#             # Optional: print full state at each node
#             # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
#         pprint("\n---\n")
    
#     # Final generation
#     pprint(value["generation"])
#     return value["generation"]
    
# if __name__=='__main__':
#     #without phone number
#     #output_final = stream_app("How do I reset my password?")

#     #with phone number
#     output_final = stream_app("How do I reset my password?","+91-800-123-4567")
#     print(output_final)

app = workflow.compile()


def stream_app(question, phone_number=None):
    # Construct the input dictionary
    inputs = {
        "question": question
    }
    
    # If a phone number is provided, construct the filters
    if phone_number:
        inputs["filters"] = {
            "path": ["phone"],
            "operator": "Equal",
            "valueString": phone_number
        }
    
    # Simulating the streaming of outputs
    final_output = None
    for output in app.stream(inputs):
        pprint(output)  # Debug print the entire output
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
        pprint("\n---\n")
        final_output = value.get("generation")  # Use get to avoid KeyError if the key is missing
    
    return final_output

@app_flask.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')
    phone_number = data.get('phone_number', None)
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    try:
        output_final = stream_app(question, phone_number)
        if output_final is None:
            return jsonify({"error": "No generation result found"}), 404
        return jsonify({"response": output_final}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app_flask.run(host='0.0.0.0', port=5000)
