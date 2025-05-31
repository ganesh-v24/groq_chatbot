from langchain_community.document_loaders import TextLoader

def load_txt_document(path):
    loader = TextLoader(path, encoding="utf-8")
    return loader.load()
