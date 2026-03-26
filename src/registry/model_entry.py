import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional


@dataclass
class ModelEntry:
    model_name:    str
    version:       str
    ticker:        str
    registered_at: str
    model_path:    str
    scaler_path:   str
    metrics:       Dict[str, float] = field(default_factory=dict)
    is_best:       bool = False
    notes:         str  = ""

    # ──────────────────────────────────────────
    # Unique ID
    # ──────────────────────────────────────────

    @property
    def entry_id(self) -> str:
        """Unique identifier: TICKER__ModelName__vN"""
        return f"{self.ticker}__{self.model_name}__{self.version}"

    # ──────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Converts entry to a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelEntry":
        """Reconstructs a ModelEntry from a dictionary (e.g. loaded from JSON)."""
        return cls(
            model_name    = data["model_name"],
            version       = data["version"],
            ticker        = data["ticker"],
            registered_at = data["registered_at"],
            model_path    = data["model_path"],
            scaler_path   = data["scaler_path"],
            metrics       = data.get("metrics", {}),
            is_best       = data.get("is_best", False),
            notes         = data.get("notes", ""),
        )

    def summary(self) -> str:
        best_tag = "  [BEST]" if self.is_best else ""
        lines = [
            f"  {self.entry_id}{best_tag}",
            f"    Registered : {self.registered_at}",
            f"    Model Path : {self.model_path}",
        ]
        if self.metrics:
            metrics_str = " | ".join(f"{k}: {v}" for k, v in self.metrics.items())
            lines.append(f"    Metrics    : {metrics_str}")
        if self.notes:
            lines.append(f"    Notes      : {self.notes}")
        return "\n".join(lines)