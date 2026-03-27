"""
image_processor.py
------------------
Handles wedding card image analysis using Groq's vision-capable model.
Extracts visual features: colors, theme, elements, style, motifs.
"""

import base64
import os
import json
from io import BytesIO
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


def encode_image_to_base64(image: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded JPEG string."""
    buffered = BytesIO()
    # Convert to RGB if needed (removes alpha channel for JPEG)
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def extract_image_features(image: Image.Image, style_preference: str = "auto") -> dict:
    """
    Use Groq vision model to extract structured visual features from a wedding card image.

    Args:
        image: PIL Image object of the uploaded wedding card
        style_preference: 'affordable', 'premium', or 'auto'

    Returns:
        dict with keys: colors, theme, elements, style, paper_quality, motifs, raw_description
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    image_b64 = encode_image_to_base64(image)

    style_hint = ""
    if style_preference != "auto":
        style_hint = f"\nNote: The user has indicated this card should be described as '{style_preference}' in tone."

    prompt = f"""You are an expert wedding card catalog writer. Analyze this wedding card image in detail.
{style_hint}

Respond ONLY with a valid JSON object (no markdown, no code fences) with exactly these keys:

{{
  "colors": ["list of dominant colors, e.g. ivory, gold, deep red"],
  "theme": "one of: floral, palace, traditional, royal, modern, minimalist, religious, nature, geometric",
  "elements": ["visual elements present, e.g. Ganesha, peacock, lotus, bride and groom, mandap, kalash"],
  "style": "one of: premium, traditional, affordable, luxury, elegant, rustic",
  "paper_quality": "e.g. matte, glossy, textured, embossed — infer from visual cues",
  "motifs": ["decorative motifs, e.g. paisley, floral border, foil print, laser cut, meenakari"],
  "card_type": "e.g. single fold, double fold, box card, scroll card, multi-insert",
  "finish": "e.g. gold foil stamping, embossing, spot UV, digital print, screen print",
  "raw_description": "2-3 sentence visual description of the card as a whole"
}}

Be specific and accurate. Only include elements you can actually see."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=800,
            temperature=0.2,
        )

        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        features = json.loads(raw)
        return features

    except json.JSONDecodeError:
        # Fallback: return raw text in a structured wrapper
        return {
            "colors": ["gold", "cream"],
            "theme": "traditional",
            "elements": ["floral border"],
            "style": style_preference if style_preference != "auto" else "traditional",
            "paper_quality": "matte",
            "motifs": ["floral"],
            "card_type": "single fold",
            "finish": "digital print",
            "raw_description": raw if "raw" in locals() else "Wedding card image",
        }
    except Exception as e:
        raise RuntimeError(f"Image feature extraction failed: {str(e)}")


def features_to_query_string(features: dict) -> str:
    """
    Convert extracted image features into a natural language query
    for vector database retrieval.
    """
    colors = ", ".join(features.get("colors", []))
    elements = ", ".join(features.get("elements", []))
    motifs = ", ".join(features.get("motifs", []))
    theme = features.get("theme", "traditional")
    style = features.get("style", "traditional")
    card_type = features.get("card_type", "wedding card")

    query = (
        f"{style} {theme} {card_type} with {colors} color scheme. "
        f"Features {elements}. "
        f"Decorative elements include {motifs}. "
        f"{features.get('finish', '')} finish on {features.get('paper_quality', 'quality')} paper."
    )
    return query.strip()
