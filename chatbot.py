import os
from getpass import getpass
from dotenv import load_dotenv
from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack_integrations.components.generators.openrouter import OpenRouterChatGenerator
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

# Load environment variables from .env file
load_dotenv()

# Check if OPENAI_API_KEY is set; prompt if not
if "OPENROUTER_API_KEY" not in os.environ:
    os.environ["OPENROUTER_API_KEY"] = getpass("Enter OpenRouter API key: ")

# Initialize document store and add sample documents
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="User prefers concise answers and has a background in tech."),
    Document(content="User enjoys detailed explanations for complex topics."),
    Document(content="User is interested in AI and machine learning applications.")
])

# Define prompt template for personalized responses
prompt_template = [
    ChatMessage.from_system("You are a helpful assistant that personalizes responses based on user preferences. Use the provided documents to tailor your answers."),
    ChatMessage.from_user(
        "Documents:\n{% for doc in documents %}{{ doc.content }}\n{% endfor %}\n"
        "Question: {{question}}\n"
        "Answer concisely if the user prefers brevity, or provide detailed explanations if the user enjoys complexity, based on the documents."
    )
]

# Build RAG pipeline
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", InMemoryBM25Retriever(document_store=document_store))
rag_pipeline.add_component("prompt_builder", ChatPromptBuilder(template=prompt_template, required_variables=["question", "documents"]))
rag_pipeline.add_component("llm", OpenRouterChatGenerator(api_key=Secret.from_env_var("OPENROUTER_API_KEY"), model="gpt-4o-mini"))
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm.messages")

# Function to run the chatbot
def run_chatbot():
    print("Welcome to the Chatbot! Type 'exit' to quit.")
    while True:
        question = input("Ask a question: ")
        if question.lower() == "exit":
            print("Goodbye!")
            break
        results = rag_pipeline.run({
            "retriever": {"query": question},
            "prompt_builder": {"question": question}
        })
        print("Answer:", results["llm"]["replies"][0].text)

# Run the chatbot
if __name__ == "__main__":
    run_chatbot()