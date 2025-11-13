from langchain_community.document_loaders import PyPDFLoader
import pprint
import os

PATH_DATA = os.path.dirname(os.getcwd()) + "/data/"
PATH_INPUT_PDF = PATH_DATA + "slide.pdf"

# Load lecture slide
loader = PyPDFLoader(
    PATH_INPUT_PDF,
    mode="page",)

docs = loader.load()
print(len(docs))
pprint.pp(docs[0].metadata)


