from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_AGENT_MD_PATH = _ROOT / "agent.md"
_SKILLS_DIR = _ROOT / "skills"


def load_agent_system_prompt() -> str:
    return _AGENT_MD_PATH.read_text(encoding="utf-8").strip()


def load_skill_documents() -> dict[str, str]:
    if not _SKILLS_DIR.exists():
        return {}

    skill_docs: dict[str, str] = {}
    for path in sorted(_SKILLS_DIR.glob("*.md")):
        skill_docs[path.stem] = path.read_text(encoding="utf-8").strip()
    return skill_docs
