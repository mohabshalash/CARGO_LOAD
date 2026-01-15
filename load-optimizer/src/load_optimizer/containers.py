# src/load_optimizer/containers.py
from __future__ import annotations

# Internal usable dims (meters). You can adjust later based on your preferred spec source.
CONTAINER_PRESETS_M: dict[str, dict[str, float]] = {
    "20":   {"length": 5.900,  "width": 2.352, "height": 2.395},
    "20HC": {"length": 5.891,  "width": 2.330, "height": 2.700},
    "40":   {"length": 12.032, "width": 2.352, "height": 2.395},
    "40HC": {"length": 12.032, "width": 2.350, "height": 2.700},
    "48HC": {"length": 14.470, "width": 2.352, "height": 2.698},
    "53HC": {"length": 15.951, "width": 2.489, "height": 2.769},
    "52HC": {"length": 15.951, "width": 2.489, "height": 2.769},  # alias
}

def get_container_dims(preset: str) -> dict[str, float]:
    key = preset.strip().upper()
    if key not in CONTAINER_PRESETS_M:
        raise ValueError(f"Unknown container_preset '{preset}'. Valid: {sorted(CONTAINER_PRESETS_M.keys())}")
    return CONTAINER_PRESETS_M[key]

