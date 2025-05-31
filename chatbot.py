import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from loader import load_txt_document
from vectorstore import build_vectorstore
from prompt import get_custom_prompt

# Load API Key
load_dotenv()
groq_key = os.getenv("GROQ_API_KEY")

# Load and process document
documents = load_txt_document("data/frankenstein.txt")
retriever = build_vectorstore(documents)

# Set up Groq + LangChain
llm = ChatGroq(api_key=groq_key, model="llama3-70b-8192")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": get_custom_prompt()},
    return_source_documents=True
)

def main():
    print(" Groq Document Chatbot (Frankenstein)")
    print("Type your question and press Enter.")
    print("Type 'exit' or 'quit' to stop the program.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break
        if not query:
            continue

        try:
            result = qa_chain({"query": query})
            answer = result.get("result", "No answer found.")
            source_docs = result.get("source_documents", [])
            source_excerpt = (
                source_docs[0].page_content[:300] + "..." if source_docs else "No source available."
            )

            print(f"\n Answer:\n{answer}\n")
            print(f" Source excerpt:\n{source_excerpt}\n")
        except Exception as e:
            print(f"\n Error: {e}\n")

if __name__ == "__main__":
    main()
