from langchain.tools import BaseTool
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from typing import Type
import json


class FlashcardInput(BaseModel):
    slide_content: str = Field(description="Content from a lecture slide")
    slide_number: int = Field(description="Slide number for reference")


class DefinitionFlashcardTool(BaseTool):
    name: str = "create_definition_flashcard"
    description: str = "Creates a definition-based flashcard from technical content"
    args_schema: Type[BaseModel] = FlashcardInput

    def _run(self, slide_content: str, slide_number: int) -> str:
        # This will be called by the agent
        prompt = f"""
        Create a definition flashcard from this slide content.
        Focus on key technical terms, concepts, or mathematical definitions.
        PRESERVE ALL LaTeX MATH NOTATION: $equation$ or $$display_math$$
        
        Slide {slide_number}:
        {slide_content}
        
        Return JSON format: {{"question": "...", "answer": "...", "type": "definition"}}
        """

        # You'll call an LLM here - for now, let's structure the response
        return prompt


class ProcessFlashcardTool(BaseTool):
    name: str = "create_process_flashcard"
    description: str = (
        "Creates a flashcard about processes, algorithms, or step-by-step procedures"
    )
    args_schema: Type[BaseModel] = FlashcardInput

    def _run(self, slide_content: str, slide_number: int) -> str:
        prompt = f"""
        Create a process/procedure flashcard from this content.
        Focus on algorithms, steps, workflows, or methodologies.
        PRESERVE ALL LaTeX MATH NOTATION.
        
        Slide {slide_number}:
        {slide_content}
        
        Return JSON format: {{"question": "...", "answer": "...", "type": "process"}}
        """
        return prompt


class FormulaFlashcardTool(BaseTool):
    name: str = "create_formula_flashcard"
    description: str = (
        "Creates flashcards for mathematical formulas, theorems, or equations"
    )
    args_schema: Type[BaseModel] = FlashcardInput

    def _run(self, slide_content: str, slide_number: int) -> str:
        prompt = f"""
        Create a formula/theorem flashcard from this mathematical content.
        Focus on equations, proofs, theorems, or mathematical concepts.
        PRESERVE ALL LaTeX MATH NOTATION exactly.
        
        Slide {slide_number}:
        {slide_content}
        
        Return JSON format: {{"question": "...", "answer": "...", "type": "formula"}}
        """
        return prompt


class ConceptFlashcardTool(BaseTool):
    name: str = "create_concept_flashcard"
    description: str = (
        "Creates flashcards for high-level concepts, comparisons, or relationships"
    )
    args_schema: Type[BaseModel] = FlashcardInput

    def _run(self, slide_content: str, slide_number: int) -> str:
        prompt = f"""
        Create a concept flashcard from this content.
        Focus on conceptual understanding, comparisons, cause-effect relationships.
        PRESERVE ALL LaTeX MATH NOTATION.
        
        Slide {slide_number}:
        {slide_content}
        
        Return JSON format: {{"question": "...", "answer": "...", "type": "concept"}}
        """
        return prompt


def create_flashcard_agent():
    # Initialise LLM
    llm = ChatOllama(
        model="phi4-mini:latest",
        temperature=0.3,
        max_tokens=2000,
    )

    # Create tools
    tools = [
        DefinitionFlashcardTool(),
        ProcessFlashcardTool(),
        FormulaFlashcardTool(),
        ConceptFlashcardTool(),
    ]

    # Agent system message
    system_message = SystemMessage(
        content="""You are an expert educational AI that creates high quality flashcards from lecture slides.

ANALYZE each slide and choose the BEST flashcard type:
- Use create_definition_flashcard for: terms, definitions, vocabulary
- Use create_process_flashcard for: algorithms, steps, procedures, workflows  
- Use create_formula_flashcard for: equations, theorems, proofs, math formulas
- Use create_concept_flashcard for: concepts, comparisons, relationships, theories

CRITICAL RULES:
1. PRESERVE ALL LaTeX math notation exactly: $inline$ and $$display$$
2. Create ONE high-quality flashcard per slide
3. Questions should be clear and test understanding
4. Answers should be accurate and concise
5. Focus on the most important concept from each slide

After creating a flashcard, summarise what you created."""
    )

    # Initialize agent
    agent = create_agent(
        tools=tools,
        model=llm,
        # agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        system_prompt=system_message,
    )

    return agent


def process_slides_with_agent(docs, max_slides=5):
    """Process slides using the agentic system"""
    agent = create_flashcard_agent()
    all_flashcards = []

    for i, doc in enumerate(docs[:max_slides]):  # Limit for testing
        print(f"\n=== Processing Slide {i + 1} ===")
        print(f"Content preview: {doc.page_content[:200]}...")

        try:
            # Let the agent decide which tool to use
            result = agent.run(
                f"Analyze this slide content and create the most appropriate flashcard. "
                f"Slide number: {i + 1}\n"
                f"Slide content: {doc.page_content}"
            )

            # Extract flashcard from result
            flashcard = extract_flashcard_from_result(result)
            if flashcard:
                flashcard["slide_number"] = i + 1
                flashcard["metadata"] = doc.metadata
                all_flashcards.append(flashcard)
                print(f"Created: {flashcard.get('type', 'unknown')} flashcard")

        except Exception as e:
            print(f"Error processing slide {i + 1}: {str(e)}")
            # Fallback: create simple flashcard
            fallback = create_fallback_flashcard(doc.page_content, i + 1)
            all_flashcards.append(fallback)

    return all_flashcards


def extract_flashcard_from_result(result: str) -> dict:
    """Extract structured flashcard from agent result"""
    try:
        # Look for JSON in the result
        if "{" in result and "}" in result:
            json_str = result[result.find("{") : result.rfind("}") + 1]
            return json.loads(json_str)
    except Exception:
        pass

    # Fallback: create basic structure
    return {
        "question": "Generated from slide",
        "answer": result[:500],  # Truncate if too long
        "type": "general",
    }


def create_fallback_flashcard(content: str, slide_num: int) -> dict:
    """Create a basic flashcard when agent fails"""
    return {
        "question": f"Key concept from slide {slide_num}",
        "answer": content[:300] + "..." if len(content) > 300 else content,
        "type": "fallback",
        "slide_number": slide_num,
    }
