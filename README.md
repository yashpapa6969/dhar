# Customer Query Answering System with Real-Time ASR and RAG

This project integrates real-time Automatic Speech Recognition (ASR) using sockets, powered by a Large Language Model (LLM) utilizing Retrieval-Augmented Generation (RAG) to answer customer queries. The RAG implements adaptive, corrective, and self-reflective functionalities to enhance response accuracy and relevance.

## System Overview

When a query is received, it is analyzed and routed based on its nature:
- **Indexed Information**: If the query is related to the indexed information stored in the Weaviate vector database, the system retrieves relevant data from the vector store.
- **Current Events**: For queries about current events, the system conducts a web search to gather up-to-date information.
- **Other Queries**: Any other types of queries are directly addressed by the LLM.

The retrieved or generated information undergoes a self-reflection process to ensure it is relevant and free of hallucinations before being provided as the final answer. This multi-step process ensures accurate and pertinent responses to customer queries.

## Features

- **Real-Time ASR**: Utilizes sockets for collecting audio real time and inference using models such as Distil-Whisper for real-time speech transcription.
- **Retrieval-Augmented Generation (RAG)**: Enhances the LLM with adaptive, corrective, and self-reflective functionalities.
- **Weaviate Vector Database**: Stores and retrieves indexed information.
- **Web Search Integration**: Fetches up-to-date information for current events.
- **Self-Reflection Process**: Ensures relevance and accuracy of responses.
- **Text-To-Speech**: Ensures the generated text response is conveyed in a natural sounding voice.


