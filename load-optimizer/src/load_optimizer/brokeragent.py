"""BrokerAgent: Conversational agent for collecting shipment data and calculating readiness."""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Any

import certifi
import httpx
from fastapi import HTTPException
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, APIError
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# In-memory session storage (MVP)
_session_store: dict[str, dict[str, Any]] = {}

# Controlled product types
VALID_PRODUCT_TYPES = ["cartons", "pallets", "crates", "bags", "rolls", "drums", "loose"]

# Valid container types
VALID_CONTAINER_TYPES = ["20GP", "40GP", "40HC"]


class AgentDraft(BaseModel):
    """Draft shipment data structure for BrokerAgent."""
    # Quote-ready fields
    origin_country: str | None = Field(None, description="Origin country")
    origin_city: str | None = Field(None, description="Origin city")
    destination_country: str | None = Field(None, description="Destination country")
    destination_city: str | None = Field(None, description="Destination city")
    container_type: str | None = Field(None, description="Container type: 20GP|40GP|40HC")
    quantity: int | None = Field(None, gt=0, description="Quantity of units")
    unit_weight: float | None = Field(None, gt=0, description="Unit weight value")
    unit_weight_unit: str | None = Field(None, description="Unit weight unit: kg|lb")
    unit_length: float | None = Field(None, gt=0, description="Unit length dimension")
    unit_width: float | None = Field(None, gt=0, description="Unit width dimension")
    unit_height: float | None = Field(None, gt=0, description="Unit height dimension")
    unit_dimension_unit: str | None = Field(None, description="Unit dimension unit: cm|in")
    product_type: str | None = Field(None, description="Product type from controlled list")
    
    # Optimize-ready fields (additional)
    stackable: bool | None = Field(None, description="Whether items can be stacked")
    palletized: bool | None = Field(None, description="Whether items are on pallets")
    max_stack: int | None = Field(None, gt=0, description="Maximum stacking height")
    
    # Savings-ready fields (additional)
    ready_date: str | None = Field(None, description="Ready date (ISO format)")
    pickup_window_start: str | None = Field(None, description="Pickup window start (ISO format)")
    pickup_window_end: str | None = Field(None, description="Pickup window end (ISO format)")


class AgentResponse(BaseModel):
    """Response from BrokerAgent conversation."""
    draft_json: AgentDraft = Field(description="Current draft shipment data")
    conversation_history: list[dict[str, str]] = Field(description="Conversation history")
    missing_fields: list[str] = Field(description="List of missing required fields")


def calculate_readiness(draft: dict[str, Any]) -> dict[str, bool]:
    """
    Calculate readiness flags server-side (don't trust model).
    
    Quote-ready requires:
        origin(country+city), destination(country+city), container_type (20GP|40GP|40HC),
        quantity, unit_weight(value+kg/lb), unit_dimensions(L/W/H + unit), product_type
    
    Optimize-ready adds: stackable (bool) + palletized (bool) (max_stack optional)
    
    Savings-ready adds: ready_date OR pickup_window(start+end)
    """
    ready_for_quote = (
        draft.get("origin_country") is not None and draft.get("origin_country") != "" and
        draft.get("origin_city") is not None and draft.get("origin_city") != "" and
        draft.get("destination_country") is not None and draft.get("destination_country") != "" and
        draft.get("destination_city") is not None and draft.get("destination_city") != "" and
        draft.get("container_type") is not None and draft.get("container_type") in VALID_CONTAINER_TYPES and
        draft.get("quantity") is not None and draft.get("quantity") > 0 and
        draft.get("unit_weight") is not None and draft.get("unit_weight") > 0 and
        draft.get("unit_weight_unit") is not None and draft.get("unit_weight_unit") in ["kg", "lb"] and
        draft.get("unit_length") is not None and draft.get("unit_length") > 0 and
        draft.get("unit_width") is not None and draft.get("unit_width") > 0 and
        draft.get("unit_height") is not None and draft.get("unit_height") > 0 and
        draft.get("unit_dimension_unit") is not None and draft.get("unit_dimension_unit") in ["cm", "in"] and
        draft.get("product_type") is not None and draft.get("product_type") in VALID_PRODUCT_TYPES
    )
    
    ready_for_optimize = ready_for_quote and (
        draft.get("stackable") is not None and
        draft.get("palletized") is not None
    )
    
    ready_for_savings = ready_for_optimize and (
        draft.get("ready_date") is not None and draft.get("ready_date") != "" or
        (draft.get("pickup_window_start") is not None and draft.get("pickup_window_start") != "" and
         draft.get("pickup_window_end") is not None and draft.get("pickup_window_end") != "")
    )
    
    return {
        "ready_for_quote": ready_for_quote,
        "ready_for_optimize": ready_for_optimize,
        "ready_for_savings": ready_for_savings,
    }


def get_missing_fields_for_readiness(draft: dict[str, Any], target: str) -> list[str]:
    """Get missing fields for a specific readiness level."""
    missing = []
    
    if target == "quote":
        if not draft.get("origin_country"): missing.append("origin_country")
        if not draft.get("origin_city"): missing.append("origin_city")
        if not draft.get("destination_country"): missing.append("destination_country")
        if not draft.get("destination_city"): missing.append("destination_city")
        if not draft.get("container_type") or draft.get("container_type") not in VALID_CONTAINER_TYPES:
            missing.append("container_type")
        if not draft.get("quantity") or draft.get("quantity") <= 0: missing.append("quantity")
        if not draft.get("unit_weight") or draft.get("unit_weight") <= 0: missing.append("unit_weight")
        if not draft.get("unit_weight_unit") or draft.get("unit_weight_unit") not in ["kg", "lb"]:
            missing.append("unit_weight_unit")
        if not draft.get("unit_length") or draft.get("unit_length") <= 0: missing.append("unit_length")
        if not draft.get("unit_width") or draft.get("unit_width") <= 0: missing.append("unit_width")
        if not draft.get("unit_height") or draft.get("unit_height") <= 0: missing.append("unit_height")
        if not draft.get("unit_dimension_unit") or draft.get("unit_dimension_unit") not in ["cm", "in"]:
            missing.append("unit_dimension_unit")
        if not draft.get("product_type") or draft.get("product_type") not in VALID_PRODUCT_TYPES:
            missing.append("product_type")
    
    elif target == "optimize":
        missing = get_missing_fields_for_readiness(draft, "quote")
        if draft.get("stackable") is None: missing.append("stackable")
        if draft.get("palletized") is None: missing.append("palletized")
    
    elif target == "savings":
        missing = get_missing_fields_for_readiness(draft, "optimize")
        if not draft.get("ready_date") and not (draft.get("pickup_window_start") and draft.get("pickup_window_end")):
            missing.append("ready_date or pickup_window")
    
    return missing


def _session_state_context_bullets(session_state: dict[str, Any]) -> str:
    """Summarize session_state dict as bullet points for system context."""
    if not session_state:
        return "(no session state yet)"
    lines = []
    for k, v in session_state.items():
        if isinstance(v, dict):
            lines.append(f"- {k}: {json.dumps(v)}")
        else:
            lines.append(f"- {k}: {v}")
    return "\n".join(lines) if lines else "(empty)"


async def agent_conversation(
    session_id: str,
    message: str,
    recent_messages: list[dict[str, str]],
    session_state: dict[str, Any],
) -> dict[str, Any]:
    """
    Handle BrokerAgent conversation turn using DB-backed context.

    recent_messages: last N messages from DB (list of {role, content}), already including current user message.
    session_state: key/value from session_state table (e.g. draft, ...).

    Returns:
        {
            "assistant_message": str,
            "draft_json": dict,
            "missing_fields": list[str],
            "ready_for_quote": bool,
            "ready_for_optimize": bool,
            "ready_for_savings": bool,
        }
    """
    draft = session_state.get("draft")
    if not isinstance(draft, dict):
        draft = {}
    conversation_history = recent_messages

    # Read OPENAI_API_KEY
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")
    
    # Initialize OpenAI client
    http_client = httpx.Client(timeout=30.0, verify=certifi.where())
    client = OpenAI(api_key=api_key, http_client=http_client)
    
    # Build system instruction
    missing_fields = get_missing_fields_for_readiness(draft, "savings")  # Check for highest level
    readiness = calculate_readiness(draft)
    
    state_bullets = _session_state_context_bullets(session_state)
    system_instruction = f"""You are a logistics broker agent helping collect shipment information.

Session state (from DB):
{state_bullets}

Current draft data: {json.dumps(draft, indent=2)}

Current status:
- Ready for quote: {readiness['ready_for_quote']}
- Ready for optimize: {readiness['ready_for_optimize']}
- Ready for savings: {readiness['ready_for_savings']}

Rules:
1. Use natural language only (no JSON in responses, but extract structured data).
2. Ask for missing info in batches of 2-4 questions per turn (don't overwhelm the user).
3. If user provides dimensions without units, ask them to confirm CM or IN (do not assume).
4. Product type must be one of: {', '.join(VALID_PRODUCT_TYPES)}. If unclear, ask user to choose from this list.
5. Container type must be one of: {', '.join(VALID_CONTAINER_TYPES)}.
6. Capture UNIT weight only (not total); backend computes total from unit_weight * quantity.
7. Be conversational and helpful, not robotic.

Valid fields:
- origin_country, origin_city
- destination_country, destination_city
- container_type (20GP|40GP|40HC)
- quantity (integer > 0)
- unit_weight (float > 0), unit_weight_unit (kg|lb)
- unit_length, unit_width, unit_height (all float > 0), unit_dimension_unit (cm|in)
- product_type ({'|'.join(VALID_PRODUCT_TYPES)})
- stackable (bool), palletized (bool), max_stack (int, optional)
- ready_date (ISO date string) OR pickup_window_start + pickup_window_end (ISO date strings)

Extract information from user messages and respond naturally. Ask follow-up questions to collect missing data."""
    
    # Build LLM input: system + last 20 messages (already from DB, chronological)
    input_messages = [
        {"role": "system", "content": system_instruction}
    ]
    input_messages.extend(conversation_history)
    
    try:
        # First: Extract structured data using structured output
        try:
            extract_response = client.responses.parse(
                model="gpt-4o-mini",
                input=input_messages,
                text_format=AgentDraft,
                temperature=0.0  # Deterministic for data extraction
            )
            
            # Update draft with extracted data (merge, don't replace)
            extracted_draft = extract_response.output_parsed.model_dump(exclude_none=True)
            draft.update(extracted_draft)
        except Exception as extract_error:
            logger.warning(f"Structured extraction failed, continuing with current draft: {extract_error}")
            # Continue with existing draft if extraction fails
        
        # Second: Generate natural language response using chat completion
        # Get missing fields first to inform the response
        missing_fields_pre = get_missing_fields_for_readiness(draft, "savings")
        
        try:
            chat_messages = [
                {"role": "system", "content": "You are a friendly logistics broker. Acknowledge information received naturally and ask for any missing details politely. Keep responses brief (2-3 sentences)."},
                *conversation_history[-4:],  # Last few messages for context
            ]
            
            chat_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=chat_messages,
                temperature=0.3,
                max_tokens=200
            )
            assistant_message = chat_response.choices[0].message.content or "I've updated your shipment information."
        except Exception as chat_error:
            logger.warning(f"Chat completion failed, using fallback message: {chat_error}")
            assistant_message = "I've updated your shipment information."
        
        # Get missing fields
        missing_fields = get_missing_fields_for_readiness(draft, "savings")
        readiness = calculate_readiness(draft)
        
        # If we still need info, generate helpful follow-up questions
        if missing_fields and not assistant_message.startswith("Here"):
            # Generate questions for missing fields (2-4 max)
            fields_to_ask = missing_fields[:min(4, len(missing_fields))]
            questions = []
            
            field_prompts = {
                "origin_country": "What is the origin country?",
                "origin_city": "What is the origin city?",
                "destination_country": "What is the destination country?",
                "destination_city": "What is the destination city?",
                "container_type": f"What container type do you need? ({', '.join(VALID_CONTAINER_TYPES)})",
                "quantity": "How many units are you shipping?",
                "unit_weight": "What is the weight per unit?",
                "unit_weight_unit": "Is the weight in kg or lb?",
                "unit_length": "What is the length per unit?",
                "unit_width": "What is the width per unit?",
                "unit_height": "What is the height per unit?",
                "unit_dimension_unit": "Are dimensions in cm or inches?",
                "product_type": f"What type of product? Choose from: {', '.join(VALID_PRODUCT_TYPES)}",
                "stackable": "Can the items be stacked? (yes/no)",
                "palletized": "Are the items on pallets? (yes/no)",
                "ready_date": "What is the ready date? (YYYY-MM-DD format)",
                "pickup_window_start": "What is the pickup window start date?",
                "pickup_window_end": "What is the pickup window end date?",
            }
            
            for field in fields_to_ask:
                if field in field_prompts:
                    questions.append(field_prompts[field])
            
            if questions:
                assistant_message = f"{assistant_message}\n\nI still need a few more details:\n" + "\n".join(f"- {q}" for q in questions)
        
        # Session and messages are persisted by the API layer (DB)
        return {
            "assistant_message": assistant_message,
            "draft_json": draft,
            "missing_fields": missing_fields,
            "ready_for_quote": readiness["ready_for_quote"],
            "ready_for_optimize": readiness["ready_for_optimize"],
            "ready_for_savings": readiness["ready_for_savings"],
        }
        
    except Exception as e:
        logger.error(f"BrokerAgent error: {repr(e)}", exc_info=True)
        
        # Return error-friendly response
        error_detail = str(e)[:300] if str(e) else "Unknown error"
        raise HTTPException(
            status_code=500,
            detail=f"Agent conversation failed: {type(e).__name__}: {error_detail}"
        )


def calculate_quote(draft_json: dict[str, Any]) -> dict[str, Any]:
    """
    Calculate baseline quote based on draft_json.
    
    For MVP, uses simple rule-based pricing based on origin/destination/container type.
    """
    # Convert units to standard format (kg, cm)
    unit_weight_kg = draft_json["unit_weight"]
    if draft_json["unit_weight_unit"] == "lb":
        unit_weight_kg = unit_weight_kg * 0.453592
    
    # Calculate total weight
    quantity = draft_json["quantity"]
    total_weight_kg = unit_weight_kg * quantity
    
    # Simple pricing rules (placeholder - replace with real pricing logic)
    container_type = draft_json["container_type"]
    base_rates = {
        "20GP": 2000.0,  # USD base rate
        "40GP": 3500.0,
        "40HC": 3800.0,
    }
    base_rate = base_rates.get(container_type, 3000.0)
    
    # Add weight-based surcharge (simplified)
    weight_surcharge = max(0, (total_weight_kg - 20000) * 0.05)  # $0.05/kg over 20t
    
    # Lane multiplier (placeholder - use origin/destination in real implementation)
    lane_multiplier = 1.0  # Would lookup real lane rates
    
    total_usd = base_rate + weight_surcharge
    per_unit_usd = total_usd / quantity if quantity > 0 else 0.0
    
    breakdown = [
        {"label": "Base Rate", "usd": base_rate},
        {"label": "Weight Surcharge", "usd": weight_surcharge},
    ]
    
    return {
        "total_usd": round(total_usd, 2),
        "per_unit_usd": round(per_unit_usd, 2),
        "breakdown": breakdown,
    }


def calculate_savings(
    draft_json: dict[str, Any],
    quote_result: dict[str, Any],
    optimize_result: dict[str, Any]
) -> dict[str, Any]:
    """
    Calculate CargoPool savings based on optimize_result.
    
    Uses optimize_result VFR/WFR + capacity to compute pooled scenario and savings.
    For MVP, allows simple rule-based pooling estimate.
    """
    # Extract fill rates from optimize_result
    metrics = optimize_result.get("metrics", {})
    volume_fill_rate = metrics.get("volume_fill_rate", 0.0)
    weight_fill_rate = metrics.get("weight_fill_rate", 0.0)
    
    # Combined fill rate
    combined_fill = (volume_fill_rate + weight_fill_rate) / 2.0 if weight_fill_rate > 0 else volume_fill_rate
    
    # Simple pooling estimate: if fill rate < 70%, assume we can pool with other shipments
    # Savings = percentage of unused capacity that can be shared
    unused_capacity_pct = 1.0 - combined_fill
    pooling_efficiency = min(0.3, unused_capacity_pct * 0.5)  # Max 30% savings from pooling
    
    baseline_per_unit = quote_result.get("per_unit_usd", 0.0)
    savings_pct = pooling_efficiency * 100.0
    pooled_per_unit = baseline_per_unit * (1.0 - pooling_efficiency)
    
    explanation = (
        f"Your shipment utilizes {combined_fill*100:.1f}% of container capacity. "
        f"By pooling with other shipments through CargoPool, you can save approximately {savings_pct:.1f}% "
        f"on shipping costs (${baseline_per_unit:.2f} → ${pooled_per_unit:.2f} per unit)."
    )
    
    return {
        "savings_pct": round(savings_pct, 1),
        "baseline_per_unit": round(baseline_per_unit, 2),
        "pooled_per_unit": round(pooled_per_unit, 2),
        "explanation": explanation,
    }

