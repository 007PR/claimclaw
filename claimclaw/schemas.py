from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ComplaintPayload:
    complainant_name: str
    insurer_name: str
    policy_number: str
    mobile: str
    email: str
    grievance_category: str
    issue_summary: str
    relief_sought: str
    attachments: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PortalCredentials:
    username: str
    password: str


@dataclass(slots=True)
class EvidenceFinding:
    contestable: bool
    flags: list[str]
    rejection_reason: str
    diagnosis_summary: str
    bill_items: list[dict[str, Any]]


@dataclass(slots=True)
class LegalPosition:
    moratorium_applies: bool
    illegal_rejection: bool
    legal_basis: str
    recommended_action: str
