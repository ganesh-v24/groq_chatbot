from langchain.prompts import PromptTemplate

def get_custom_prompt():
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""\nYou are a helpful assistant answering questions based on the following document:\n\n{context}\n\nQuestion: {question}\nAnswer:\n"""
    )

