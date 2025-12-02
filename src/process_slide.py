from langchain_community.document_loaders import PyPDFLoader
import pprint
import json
import os
from src.flashcard_tools import process_slides_with_agent

PATH_DATA = os.path.dirname(os.getcwd()) + "/lecture_to_flashcard/data/"
PATH_INPUT_PDF = PATH_DATA + "slide.pdf"

# Load lecture slide
loader = PyPDFLoader(
    PATH_INPUT_PDF,
    mode="page",
)

docs = loader.load()
pprint.pp(docs[0].metadata)


# NEW: Process with agent
flashcards = process_slides_with_agent(
    docs, max_slides=3
)  # Start with 3 slides for testing

print(f"\n=== Generated {len(flashcards)} Flashcards ===")
for i, card in enumerate(flashcards):
    print(f"\nFlashcard {i + 1} (Slide {card['slide_number']}, Type: {card['type']}):")
    print(f"Q: {card['question']}")
    print(f"A: {card['answer']}")

# Save to JSON file
with open("flashcards.json", "w") as f:
    json.dump(flashcards, f, indent=2)

print("\nFlashcards saved to flashcards.json")
