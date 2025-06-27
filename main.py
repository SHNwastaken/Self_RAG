import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List, TypedDict
import tempfile # --- NEW ---

from langgraph.graph import StateGraph, END

# --- 1. SETUP AND CONFIGURATION ---

load_dotenv()

st.set_page_config(page_title="Self-Correcting RAG Agent", layout="wide")

# Initialize the Gemini LLM and Embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=.5)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Initialize the Web Search Tool
web_search_tool = TavilySearchResults(k=3, description="A search engine optimized for comprehensive, accurate, and trusted results.")

# --- 2. HELPER FUNCTION FOR VECTOR STORE ---

def create_vectorstore_from_pdf(pdf_bytes, embeddings):
    """Creates a FAISS vector store from PDF bytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        temp_file_path = temp_file.name
    
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    text_splitter = SemanticChunker(embeddings=embeddings)
    docs = text_splitter.split_documents(documents)
    
    vector_store = FAISS.from_documents(docs, embeddings)
    os.remove(temp_file_path) # Clean up the temporary file
    return vector_store

# --- 3. LANGGRAPH AGENT STATE ---

class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str
    source: str
    scratchpad: List[str]
    chat_history: List[BaseMessage] # --- NEW --- To store conversation history
    iteration: int = 0 # --- NEW --- To track refinement iterations
    critique: str = "" # --- NEW --- To store critique from reflection
    missing_concepts: str = "" # --- NEW --- To store missing concepts from reflection

# --- 4. AGENT NODES (WITH A NEW NODE) ---

# --- NEW NODE ---
def transform_query_node(state: AgentState):
    """
    Transforms the user's question into a standalone question based on chat history.
    """
    st.session_state.scratchpad.append("Transforming query based on chat history.")
    question = state["question"]
    chat_history = state["chat_history"]

    # If no chat history, the question is already standalone
    if not chat_history:
        st.session_state.scratchpad.append("No chat history, using original question.")
        return {"question": question}

    prompt = PromptTemplate(
        template="""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:""",
        input_variables=["chat_history", "question"],
    )

    query_transform_chain = prompt | llm | StrOutputParser()

    # Format chat history for the prompt
    formatted_history = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    transformed_question = query_transform_chain.invoke({
        "chat_history": formatted_history,
        "question": question
    })

    st.session_state.scratchpad.append(f"Transformed question: {transformed_question}")
    return {"question": transformed_question}

def retrieve_local_node(state: AgentState):
    """
    Retrieves documents from the local vector store based on the question.
    """
    st.session_state.scratchpad.append("Attempting to retrieve from local vector store.")
    question = state["question"]
    
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.session_state.scratchpad.append("No PDF uploaded. Cannot retrieve from local vector store.")
        # If no vector store, we can't retrieve locally, so we'll just pass an empty list
        documents = []
    else:
        retriever = st.session_state.vector_store.as_retriever()
        result_docs = retriever.invoke(question)
        documents = [doc.page_content for doc in result_docs]
    
    return {
        "documents": documents,
        "question": question,
        "scratchpad": st.session_state.scratchpad,
    }

def grade_documents_node(state: AgentState):
    """
    Critiques the retrieved documents. Decides if they are relevant enough.
    """
    st.session_state.scratchpad.append("Grading retrieved documents for relevance.")
    question = state["question"]
    documents = state["documents"]

    # If no documents were retrieved, they are not relevant
    if not documents:
        st.session_state.scratchpad.append("No documents retrieved. Score: no.")
        return {"source": "web_search"} # Force web search if no docs

    prompt = PromptTemplate(
        template="""You are a grader. Your purpose is to determine if a set of retrieved documents is relevant to a user's question.
        Provide a binary score: 'yes' if the documents are relevant, and 'no' if they are not.
        
        Here are the documents:
        {documents}
        
        Here is the user's question: {question}
        
        Give your decision as a JSON object with a single key 'score'.
        Example: {{"score": "yes"}}
        """,
        input_variables=["documents", "question"],
    )
    
    # Chain for grading
    grade_chain = prompt | llm | JsonOutputParser()
    
    # Invoke the grader
    try:
        score_output = grade_chain.invoke({"documents": documents, "question": question})
        score = score_output.get("score", "no") # Default to 'no' if parsing fails
    except Exception as e:
        st.session_state.scratchpad.append(f"Error grading documents: {e}. Defaulting to 'no'.")
        score = "no" # Default to 'no' on error

    if score.lower() == "yes":
        st.session_state.scratchpad.append("Decision: Documents are relevant. Proceeding to generate answer.")
        return {"source": "vectorstore"}
    else:
        st.session_state.scratchpad.append("Decision: Documents are NOT relevant. Falling back to web search.")
        return {"source": "web_search"}

def web_search_node(state: AgentState):
    """
    Performs a web search if the local documents were not relevant.
    """
    st.session_state.scratchpad.append("Performing web search.")
    question = state["question"]
    
    documents = web_search_tool.invoke({"query": question})
    
    return {"documents": documents, "question": question}

def generate_node(state: AgentState):
    """
    Generates the final answer based on the retrieved documents.
    """
    st.session_state.scratchpad.append("Generating final answer.")
    question = state["question"]
    documents = state["documents"]
    source = state["source"] # Get the source from the state
    critique = state.get("critique", "") # Get critique if available

    if critique:
        prompt_template = PromptTemplate(
            template="""You are an AI assistant. Here is an initial answer and a critique. Produce a new, refined answer that addresses the critique, citing the new documents.
            Critique: {critique}
            Question: {question}
            Documents: {documents}
            Answer:""",
            input_variables=["question", "documents", "critique"],
        )
    else:
        prompt_template = PromptTemplate(
            template="""You are an AI assistant. Based on the following documents, provide a comprehensive and concise answer to the user's question.
            Cite your sources clearly at the end, indicating whether the information came from the "Uploaded PDF" or "Web Search".

            Question: {question}
            Documents: {documents}

            Answer:""",
            input_variables=["question", "documents"],
        )
    
    generation_chain = prompt_template | llm
    answer = generation_chain.invoke({"question": question, "documents": documents, "critique": critique})
    
    return {"answer": answer.content, "source": source}

def reflect_node(state: AgentState):
    """
    Reflects on the generated answer and determines if it needs correction.
    """
    st.session_state.scratchpad.append("Reflecting on the generated answer.")
    question = state["question"]
    answer = state["answer"]
    documents = state["documents"]
    chat_history = state["chat_history"]

    prompt = PromptTemplate(
        template="""Given the following question, the generated answer, and the source documents, assess whether the answer is 1) accurate, 2) complete, and 3) properly sourced.
        Reply as JSON: {{ "verdict": "good" | "needs_correction", "critique": "...", "missing_concepts": "..." }}.

        Question: {question}
        Answer: {answer}
        Documents: {documents}
        Chat History: {chat_history}
        
        Reflection:""",
        input_variables=["question", "answer", "documents", "chat_history"],
    )

    reflection_chain = prompt | llm | JsonOutputParser()

    try:
        reflection_output = reflection_chain.invoke({
            "question": question,
            "answer": answer,
            "documents": documents,
            "chat_history": "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        })
        verdict = reflection_output.get("verdict", "needs_correction")
        critique = reflection_output.get("critique", "")
        missing_concepts = reflection_output.get("missing_concepts", "")
    except Exception as e:
        st.session_state.scratchpad.append(f"Error during reflection: {e}. Defaulting to 'needs_correction'.")
        verdict = "needs_correction"
        critique = "Error during reflection."
        missing_concepts = ""

    st.session_state.scratchpad.append(f"Reflection verdict: {verdict}. Critique: {critique}")
    return {"verdict": verdict, "critique": critique, "missing_concepts": missing_concepts}

def refine_query_node(state: AgentState):
    """
    Rewrites/expands the query based on missing concepts from reflection.
    Increments the iteration counter.
    """
    st.session_state.scratchpad.append("Refining query based on critique.")
    question = state["question"]
    missing_concepts = state["missing_concepts"]
    iteration = state["iteration"] + 1

    if iteration > 2: # Cap iterations to avoid infinite loops
        st.session_state.scratchpad.append("Iteration limit reached. Stopping refinement.")
        return {"question": question, "iteration": iteration, "answer": "I'm sorry, I've tried to refine the answer multiple times but I'm unable to provide a satisfactory response at this moment. Please try rephrasing your question.", "source": "agent_failure"}

    prompt = PromptTemplate(
        template="""Rewrite the standalone question to focus on the missing concepts identified in the critique.
        Original Question: {question}
        Missing Concepts: {missing_concepts}
        Rewritten Question:""",
        input_variables=["question", "missing_concepts"],
    )

    query_rewrite_chain = prompt | llm | StrOutputParser()
    rewritten_question = query_rewrite_chain.invoke({
        "question": question,
        "missing_concepts": missing_concepts
    })

    st.session_state.scratchpad.append(f"Rewritten question for refinement: {rewritten_question}")
    return {"question": rewritten_question, "iteration": iteration}


# --- 5. CONDITIONAL EDGES ---

def decide_to_generate(state: AgentState):
    """
    Determines whether to generate an answer or perform a web search.
    """
    if state["source"] == "web_search":
        return "websearch"
    else:
        return "generate"

# --- 6. GRAPH ASSEMBLY ---

graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("retrieve_local", retrieve_local_node)
graph_builder.add_node("grade_documents", grade_documents_node)
graph_builder.add_node("web_search", web_search_node)
graph_builder.add_node("generate", generate_node)
graph_builder.add_node("reflect", reflect_node) # --- NEW ---
graph_builder.add_node("refine_query", refine_query_node) # --- NEW ---
graph_builder.add_node("transform_query", transform_query_node)

# Build edges
graph_builder.set_entry_point("transform_query") # --- MODIFIED --- New entry point
graph_builder.add_edge("transform_query", "retrieve_local") # --- NEW ---
graph_builder.add_edge("retrieve_local", "grade_documents")
graph_builder.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "websearch": "web_search",
        "generate": "generate",
    },
)
graph_builder.add_edge("web_search", "generate")
graph_builder.add_edge("generate", "reflect") # --- MODIFIED ---
graph_builder.add_conditional_edges(
    "reflect",
    lambda state: "end" if state["verdict"] == "good" else "refine_query",
    {"end": END, "refine_query": "refine_query"},
)
graph_builder.add_edge("refine_query", "retrieve_local") # --- NEW ---

# Compile the graph
agent_graph = graph_builder.compile()

# --- 7. STREAMLIT UI ---

st.title("Conversational RAG Agent")
st.markdown("This agent remembers your conversation. Upload a PDF and ask follow-up questions!")

# Initialize chat history and vector store in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "scratchpad" not in st.session_state:
    st.session_state.scratchpad = []
if "chat_history" not in st.session_state: # --- NEW ---
    st.session_state.chat_history = []
if "iteration" not in st.session_state: # --- NEW ---
    st.session_state.iteration = 0

# PDF Upload Section
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        if st.button("Create Vector Store"):
            with st.spinner("Creating vector store... This may take a moment."):
                pdf_bytes = uploaded_file.read()
                st.session_state.vector_store = create_vectorstore_from_pdf(pdf_bytes, embeddings)
                st.success("Vector store created successfully!")
        else:
            st.error("Please upload a PDF file first.")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question about your document or anything else..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Clear scratchpad for new query
    st.session_state.scratchpad = []

    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            # Run the agent
            initial_state = {
                "question": prompt,
                "chat_history": st.session_state.chat_history,
                "scratchpad": st.session_state.scratchpad,
                "iteration": 0 # Initialize iteration counter
            }
            final_state = agent_graph.invoke(initial_state)
            
            answer = final_state.get("answer", "I couldn't find an answer.")
            
            st.markdown(answer)
            
            # Display scratchpad for debugging/demonstration
            with st.expander("Agent's Thought Process"):
                for step in st.session_state.scratchpad:
                    st.write(step)
    # --- MODIFIED --- Update chat history for the next turn
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    st.session_state.chat_history.append(AIMessage(content=answer))