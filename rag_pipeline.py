"""
rag_pipeline.py
---------------
LangChain RAG pipeline that:
1. Takes extracted image features + retrieved similar cards
2. Constructs a carefully engineered prompt
3. Calls Groq LLM to generate a structured wedding card description
"""

import os
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.core.output_parser import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def build_llm(temperature: float = 0.4) -> ChatGroq:
    """Initialize the Groq LLM via LangChain."""
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=1024,
    )


def format_retrieved_examples(
    retrieved: List[Tuple[Document, float]], max_examples: int = 3
) -> str:
    """
    Format retrieved similar descriptions into a structured examples block
    for the LLM prompt.
    """
    if not retrieved:
        return "No similar examples found."

    lines = []
    for i, (doc, score) in enumerate(retrieved[:max_examples], 1):
        sku = doc.metadata.get("sku", "N/A")
        similarity_pct = round(score * 100, 1)
        lines.append(f"--- Example {i} (SKU: {sku}, Similarity: {similarity_pct}%) ---")
        lines.append(doc.page_content.strip())
        lines.append("")

    return "\n".join(lines)


def build_prompt_template() -> ChatPromptTemplate:
    """
    Construct the RAG prompt template with clear instructions,
    retrieved examples, and output format specification.
    """
    system_message = """You are an expert wedding card catalog copywriter for an Indian wedding stationery company.
Your job is to write accurate, rich, and professional product descriptions for wedding cards.

STRICT RULES:
1. Only describe what is visible or can be inferred from the image features provided.
2. Never invent dimensions, GSM, or specific product codes unless they appear in the image data.
3. Match the tone of the example descriptions below — professional, specific, detail-oriented.
4. Use the exact bullet format shown with ✦ symbols.
5. Keep the description between 80-150 words.
6. Mention: card type, color palette, key visual elements, finish/material hints, and what's included (if inferable).
7. Avoid vague words like "beautiful" or "stunning" — use specific descriptors.
8. Never say "I cannot determine" — make reasonable inferences from the visual data."""

    human_message = """## VISUAL FEATURES EXTRACTED FROM THE UPLOADED CARD:
{image_features}

## SIMILAR WEDDING CARD DESCRIPTIONS FROM OUR CATALOG (for tone and style reference):
{retrieved_examples}

## TASK:
Write a structured product description for this wedding card using EXACTLY 3 bullet points — no more, no less.

Use EXACTLY this format:
"This [card_type] wedding card:
✦ Includes [contents of the set — main card + number of inserts + envelope, e.g. 'a main card with 2 inserts and an envelope']
✦ Features [main design elements, motifs, theme, and visual highlights — e.g. Ganesha, floral border, laser cut, etc.]
✦ Comes in [color palette] with [finish/print technique] on [paper quality], giving it a [style — premium/affordable/traditional] look and feel"

STRICT RULES:
- Output EXACTLY 3 bullet points using ✦ — never 4, never 2.
- Point 1 = contents only (card + inserts + envelope)
- Point 2 = design, theme, motifs, visual elements only
- Point 3 = colors + finish/material + style classification
- Do NOT mention inserts or envelope in points 2 or 3.
- Do NOT mention colors or finish in point 2.
- Do NOT add any text before or after the description. Output ONLY the description."""

    return ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message),
    ])


def generate_description(
    image_features: dict,
    retrieved: List[Tuple[Document, float]],
    style_preference: str = "auto",
) -> dict:
    """
    Full RAG pipeline: takes image features + retrieved examples,
    generates a polished wedding card description.

    Args:
        image_features: Dict of extracted visual features from image_processor
        retrieved: List of (Document, score) from vector_store
        style_preference: 'affordable', 'premium', or 'auto'

    Returns:
        dict with 'description' (str) and 'top_matches' (list of dicts)
    """
    llm = build_llm()
    prompt = build_prompt_template()
    chain = prompt | llm | StrOutputParser()

    # Format image features for the prompt
    features_text = _format_features(image_features, style_preference)
    examples_text = format_retrieved_examples(retrieved, max_examples=3)

    description = chain.invoke({
        "image_features": features_text,
        "retrieved_examples": examples_text,
    })

    # Format top matches for display
    top_matches = []
    for doc, score in retrieved[:5]:
        top_matches.append({
            "sku": doc.metadata.get("sku", "N/A"),
            "description": doc.page_content.strip(),
            "similarity": round(score * 100, 1),
            "image_url": doc.metadata.get("image_url", ""),
        })

    return {
        "description": description.strip(),
        "top_matches": top_matches,
    }


def _format_features(features: dict, style_preference: str = "auto") -> str:
    """Convert features dict into a clean bulleted text block for the prompt."""
    lines = []

    if features.get("card_type"):
        lines.append(f"• Card Type: {features['card_type']}")
    if features.get("theme"):
        lines.append(f"• Theme: {features['theme']}")
    if features.get("style"):
        effective_style = style_preference if style_preference != "auto" else features["style"]
        lines.append(f"• Style: {effective_style}")
    if features.get("colors"):
        lines.append(f"• Color Palette: {', '.join(features['colors'])}")
    if features.get("elements"):
        lines.append(f"• Visual Elements: {', '.join(features['elements'])}")
    if features.get("motifs"):
        lines.append(f"• Motifs & Decorations: {', '.join(features['motifs'])}")
    if features.get("finish"):
        lines.append(f"• Finish: {features['finish']}")
    if features.get("paper_quality"):
        lines.append(f"• Paper Quality: {features['paper_quality']}")
    if features.get("raw_description"):
        lines.append(f"• Visual Summary: {features['raw_description']}")

    return "\n".join(lines)