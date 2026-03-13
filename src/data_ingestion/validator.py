"""
src/data_ingestion/validator.py
--------------------------------
Validates raw stock data before it enters the pipeline.

Responsibilities:
  - Check required columns exist
  - Detect missing values beyond threshold
  - Detect duplicate index entries
  - Confirm sufficient data volume
  - Confirm correct data types

Returns a structured ValidationReport — does NOT raise by default,
so the caller decides whether to halt or warn.
"""

import pandas as pd
from dataclasses import dataclass, field
from typing import List

from config.config import (
    REQUIRED_COLUMNS,
    MAX_MISSING_PCT,
    MIN_REQUIRED_ROWS,
)
from config.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationReport:
    """
    Holds the outcome of a validation run.

    Attributes:
        is_valid  (bool)       : True only if ALL checks passed.
        errors    (List[str])  : Critical issues that block the pipeline.
        warnings  (List[str])  : Non-critical observations.
    """
    is_valid: bool = True
    errors:   List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.is_valid = False
        logger.error(f"[Validation ERROR] {msg}")

    def add_warning(self, msg: str):
        self.warnings.append(msg)
        logger.warning(f"[Validation WARNING] {msg}")

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [f"── Validation {status} ──"]
        if self.errors:
            lines.append(f"  Errors   ({len(self.errors)}): " + " | ".join(self.errors))
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}): " + " | ".join(self.warnings))
        if not self.errors and not self.warnings:
            lines.append("  All checks passed with no issues.")
        return "\n".join(lines)


class DataValidator:
    def validate(self, df: pd.DataFrame) -> ValidationReport:
        report = ValidationReport()

        self._check_not_empty(df, report)

        # If DataFrame is empty, skip remaining checks
        if not report.is_valid:
            return report

        self._check_required_columns(df, report)
        self._check_row_count(df, report)
        self._check_missing_values(df, report)
        self._check_duplicate_index(df, report)
        self._check_numeric_types(df, report)
        self._check_date_continuity(df, report)

        logger.info(f"Validation complete. Status: {'PASSED' if report.is_valid else 'FAILED'}")
        return report

    def _check_not_empty(self, df: pd.DataFrame, report: ValidationReport):
        if df is None or df.empty:
            report.add_error("DataFrame is None or completely empty.")

    def _check_required_columns(self, df: pd.DataFrame, report: ValidationReport):
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            report.add_error(f"Missing required columns: {missing}")

    def _check_row_count(self, df: pd.DataFrame, report: ValidationReport):
        if len(df) < MIN_REQUIRED_ROWS:
            report.add_error(
                f"Insufficient data: {len(df)} rows found, minimum required is {MIN_REQUIRED_ROWS}."
            )

    def _check_missing_values(self, df: pd.DataFrame, report: ValidationReport):
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                continue
            missing_pct = df[col].isnull().mean() * 100
            if missing_pct > MAX_MISSING_PCT:
                report.add_error(
                    f"Column '{col}' has {missing_pct:.1f}% missing values "
                    f"(threshold: {MAX_MISSING_PCT}%)."
                )
            elif missing_pct > 0:
                report.add_warning(
                    f"Column '{col}' has {missing_pct:.1f}% missing values (within threshold)."
                )

    def _check_duplicate_index(self, df: pd.DataFrame, report: ValidationReport):
        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            report.add_error(f"Found {n_dupes} duplicate date entries in the index.")

    def _check_numeric_types(self, df: pd.DataFrame, report: ValidationReport):
        non_numeric = [
            col for col in REQUIRED_COLUMNS
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            report.add_error(f"Non-numeric data in columns: {non_numeric}")

    def _check_date_continuity(self, df: pd.DataFrame, report: ValidationReport):
        """Warns if there are unusually large gaps between trading dates."""
        if not isinstance(df.index, pd.DatetimeIndex):
            report.add_warning("Index is not a DatetimeIndex — skipping date continuity check.")
            return

        gaps = df.index.to_series().diff().dt.days.dropna()
        max_gap = gaps.max()

        # Gaps > 10 calendar days (accounts for weekends + holidays) are suspicious
        if max_gap > 10:
            report.add_warning(
                f"Largest date gap is {int(max_gap)} days — possible missing data period."
            )