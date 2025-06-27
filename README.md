# Self-Correcting Conversational RAG Agent

A sophisticated Retrieval-Augmented Generation (RAG) agent built with LangGraph, Streamlit, and Google's Gemini AI that can intelligently answer questions from both uploaded PDF documents and web search results. The agent features self-correction capabilities and maintains conversation context for follow-up questions.

## Features

- **📄 PDF Document Processing**: Upload and query PDF documents using semantic chunking
- **🔍 Intelligent Source Selection**: Automatically decides between local document retrieval and web search
- **🧠 Self-Correcting Logic**: Reflects on generated answers and refines them if needed
- **💬 Conversational Memory**: Maintains chat history for contextual follow-up questions
- **🌐 Web Search Integration**: Falls back to web search when local documents aren't relevant
- **🔄 Query Transformation**: Converts follow-up questions into standalone queries
- **📊 Transparent Process**: Shows the agent's thought process and decision-making steps

## Architecture

The agent uses a graph-based workflow powered by LangGraph with the following nodes:

1. **Transform Query**: Converts follow-up questions into standalone queries using chat history
2. **Retrieve Local**: Searches the uploaded PDF using semantic similarity
3. **Grade Documents**: Evaluates relevance of retrieved documents
4. **Web Search**: Performs web search when local documents aren't sufficient
5. **Generate**: Creates answers based on retrieved information
6. **Reflect**: Analyzes the generated answer for accuracy and completeness
7. **Refine Query**: Rewrites queries based on reflection feedback (with iteration limits)

## Prerequisites

- Python 3.8+
- Google AI API key (for Gemini)
- Tavily API key (for web search)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/self-correcting-rag-agent.git
cd self-correcting-rag-agent
```

2. Install required packages:
```bash
pip install streamlit langchain-google-genai langchain-community langchain-experimental langchain-core langgraph faiss-cpu pypdf python-dotenv tavily-python
```

3. Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Upload a PDF document using the sidebar

4. Click "Create Vector Store" to process the document

5. Start asking questions! The agent will:
   - First try to answer from your PDF
   - Fall back to web search if needed
   - Self-correct if the initial answer is insufficient
   - Remember conversation context for follow-up questions

## API Keys Setup

### Google AI API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add it to your `.env` file as `GOOGLE_API_KEY`

### Tavily API Key
1. Sign up at [Tavily](https://tavily.com/)
2. Get your API key from the dashboard
3. Add it to your `.env` file as `TAVILY_API_KEY`

## Configuration

The agent uses the following models and settings:

- **LLM**: Google Gemini 2.5 Flash (temperature: 0.5)
- **Embeddings**: Google Embedding-001
- **Text Splitting**: Semantic chunking for better context preservation
- **Vector Store**: FAISS for efficient similarity search
- **Web Search**: Tavily (returns top 3 results)
- **Max Iterations**: 2 refinement cycles to prevent infinite loops

## Example Use Cases

- **Research Assistant**: Upload research papers and ask follow-up questions
- **Document Analysis**: Analyze contracts, reports, or manuals
- **Educational Tool**: Upload textbooks and get explanations with web context
- **Technical Documentation**: Query API docs with real-time web updates

## Project Structure

```
├── app.py                 # Main Streamlit application
├── .env                   # Environment variables (create this)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Dependencies

```txt
streamlit
langchain-google-genai
langchain-community
langchain-experimental
langchain-core
langgraph
faiss-cpu
pypdf
python-dotenv
tavily-python
```

## How It Works

1. **Query Processing**: User questions are transformed into standalone queries considering chat history
2. **Document Retrieval**: The system searches the uploaded PDF using semantic similarity
3. **Relevance Grading**: An AI grader determines if retrieved documents are relevant
4. **Smart Fallback**: If documents aren't relevant, the system performs a web search
5. **Answer Generation**: Creates comprehensive answers citing sources (PDF or web)
6. **Self-Reflection**: Analyzes the answer for accuracy and completeness
7. **Iterative Refinement**: If needed, refines the query and repeats the process

## Limitations

- Maximum of 2 refinement iterations to prevent infinite loops
- PDF processing time depends on document size
- Web search results depend on Tavily API availability
- Requires active internet connection for web search fallback

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Powered by [Google Gemini AI](https://deepmind.google/technologies/gemini/)
- Web search by [Tavily](https://tavily.com/)
- UI built with [Streamlit](https://streamlit.io/)


⭐ **Star this repository if you find it helpful!**
