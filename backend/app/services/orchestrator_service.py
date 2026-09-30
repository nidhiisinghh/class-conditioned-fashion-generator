"""
Weave LLM Orchestrator Service
Two-sided control loop component:
1. Structures free-text user briefs into typed category/fabric/style/color objects.
   Triggers exactly one clarifying question if ambiguous (BR002, US-03).
2. Routes plain-language designer feedback to either inpainting or regeneration (BR004, US-07, ADR-004/005).
"""

import re
import json
import logging
from enum import Enum
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class RoutedOperation(str, Enum):
    GENERATE = "generate"
    INPAINT = "inpaint"
    VARIATION = "variation"

# Fashion Domain Taxonomies for Heuristic Parsing
KNOWN_CATEGORIES = [
    "Dress", "Jacket", "Coat", "Pants", "Shirt",
    "Top", "Skirt", "Sweater", "Suit", "Jumpsuit", "Shorts", "Hoodie"
]

KNOWN_FABRICS = [
    "silk organza", "silk", "heavy wool", "wool", "linen", "cashmere",
    "leather", "cotton poplin", "cotton", "denim", "satin", "chiffon",
    "velvet", "taffeta", "lace", "tweed"
]

KNOWN_COLORS = [
    "terracotta", "bone white", "charcoal black", "black", "white", "dusty rose",
    "rose", "navy", "midnight navy", "olive green", "olive", "camel", "beige",
    "emerald", "burgundy", "cream", "pastel"
]

KNOWN_STYLES = [
    "minimalist", "avant-garde", "streetwear", "couture", "vintage",
    "structured", "flowing", "relaxed", "tailored", "oversized", "bohemian"
]


class LLMOrchestrator:
    """
    Core orchestrator that makes the LLM a control-loop decision maker,
    not just a one-shot prompt formatter.
    """
    LOCAL_EDIT_KEYWORDS = {
        "sleeve", "sleeves", "collar", "pocket", "pockets", "hem", "cuff", "cuffs",
        "zipper", "button", "buttons", "neckline", "slit", "waistband", "strap",
        "straps", "seam", "lapel", "hood", "fringe", "embroidery"
    }

    GLOBAL_EDIT_KEYWORDS = {
        "fabric", "material", "color", "colour", "shade", "palette", "mood",
        "silhouette", "style", "fit", "vibe", "aesthetic", "pattern", "print",
        "texture", "overall", "entire", "look"
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def structure_brief(self, raw_text: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Structures a free-text brief into {category, style, color, fabric, mood}.
        If the garment category is completely missing or ambiguous, returns
        a single clarifying question without generating (BR002).
        """
        text = raw_text.strip()
        lower_text = text.lower()

        # 1. Detect Category
        detected_category = None
        for cat in KNOWN_CATEGORIES:
            # Handle plural and lower
            if re.search(r'\b' + cat.lower() + r's?\b', lower_text):
                detected_category = cat
                break
        
        # Category aliases
        if not detected_category:
            if any(w in lower_text for w in ["gown", "frock", "robe"]):
                detected_category = "Dress"
            elif any(w in lower_text for w in ["blazer", "tuxedo"]):
                detected_category = "Suit"
            elif any(w in lower_text for w in ["jeans", "trousers", "slacks", "chinos"]):
                detected_category = "Pants"
            elif any(w in lower_text for w in ["blouse", "tee", "t-shirt"]):
                detected_category = "Top"
            elif any(w in lower_text for w in ["overcoat", "trench"]):
                detected_category = "Coat"

        # If category is completely absent, trigger ONE clarifying question (US-03)
        if not detected_category:
            clarifying_question = (
                "What type of garment would you like to design? (e.g. an evening dress, tailored jacket, or casual trousers)"
            )
            return {
                "category": None,
                "raw": raw_text
            }, clarifying_question

        # 2. Detect Fabric
        detected_fabric = "luxury textile"
        for fab in KNOWN_FABRICS:
            if fab in lower_text:
                detected_fabric = fab
                break

        # 3. Detect Color
        detected_color = "terracotta"  # Weave signature default
        for col in KNOWN_COLORS:
            if col in lower_text:
                detected_color = col
                break

        # 4. Detect Style & Mood
        detected_style = "contemporary"
        for sty in KNOWN_STYLES:
            if sty in lower_text:
                detected_style = sty
                break

        detected_mood = "studio editorial"
        if any(w in lower_text for w in ["breezy", "airy", "spring", "summer"]):
            detected_mood = "breezy seasonal"
        elif any(w in lower_text for w in ["dramatic", "bold", "dark"]):
            detected_mood = "dramatic contrast"

        structured = {
            "category": detected_category,
            "style": detected_style,
            "color": detected_color,
            "fabric": detected_fabric,
            "mood": detected_mood,
            "refined_prompt": (
                f"a high-end {detected_style} {detected_color} {detected_category.lower()} "
                f"crafted from fine {detected_fabric}, {detected_mood}, studio fashion photography"
            )
        }
        return structured, None

    extract_structured_brief = structure_brief

    def route_feedback(
        self, feedback_text: str, current_brief: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Decides whether feedback is a local fix (Inpaint) or a global redesign (Regenerate).
        Strictly satisfies PRD Document 8.2 and ADR-005.
        """
        text = feedback_text.lower()
        brief_copy = dict(current_brief or {})

        # Check for local detail keywords first
        matched_local = [kw for kw in self.LOCAL_EDIT_KEYWORDS if re.search(r'\b' + kw + r'\b', text)]
        if matched_local:
            reason = f"Targeted adjustment to garment sub-element: {', '.join(matched_local)}. Routing to local inpainting."
            return {
                "operation": RoutedOperation.INPAINT.value,
                "reasoning": reason,
                "updated_brief": brief_copy,
            }

        # Check for variation
        if any(w in text for w in ["more like this", "variation", "options", "different angle", "similar"]):
            return {
                "operation": RoutedOperation.VARIATION.value,
                "reasoning": "Exploring aesthetic variations around current concept direction.",
                "updated_brief": brief_copy,
            }

        # Check for global style keywords or default
        matched_global = [kw for kw in self.GLOBAL_EDIT_KEYWORDS if re.search(r'\b' + kw + r'\b', text)]
        if matched_global:
            reason = f"Broad structural change to garment {', '.join(matched_global)}. Routing to full regeneration."
            # Check if color or fabric is specified in feedback
            for col in KNOWN_COLORS:
                if col in text:
                    brief_copy["color"] = col
            for fab in KNOWN_FABRICS:
                if fab in text:
                    brief_copy["fabric"] = fab
            for sty in KNOWN_STYLES:
                if sty in text:
                    brief_copy["style"] = sty

            return {
                "operation": RoutedOperation.GENERATE.value,
                "reasoning": reason,
                "updated_brief": brief_copy,
            }

        # Default fallback
        return {
            "operation": RoutedOperation.INPAINT.value,
            "reasoning": "Instruction interpreted as localized refinement. Preserving approved silhouette.",
            "updated_brief": brief_copy,
        }
