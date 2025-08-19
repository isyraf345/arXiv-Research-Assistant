# arXiv-Research-Assistant
arXiv (pronounced "archive") is a free, open-access repository of scientific papers, primarily in physics, mathematics, computer science, and related fields. This project allows you to query and interact with arXiv papers using natural language. This system downloads, processes, and indexes research papers, then provides intelligent answers based on the ingested papers.

Tech stack used:
- RAG + LangChain
- vector database (chroma db)
- openAI for LLM

How to use:
- Create your own .env file that contains information OpenAI pass key and database configuration
- python app.py
