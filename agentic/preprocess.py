"""Content preprocessing: extract different textual views from session messages."""
from typing import Dict, List
from agentic.extract import Message


def _text_blocks(messages: List[Message], roles: set | None = None) -> List[str]:
    parts = []
    for msg in messages:
        if roles and msg.role not in roles:
            continue
        for block in msg.content:
            if block.get("type") == "text" and block.get("text"):
                parts.append(block["text"])
    return parts


def extract_text_full(messages: List[Message]) -> str:
    parts = []
    for msg in messages:
        for block in msg.content:
            btype = block.get("type")
            if btype == "text" and block.get("text"):
                parts.append(block["text"])
            elif btype == "tool_use":
                parts.append(f"[Tool: {block.get('name', '')}]")
            elif btype == "tool_result":
                content = block.get("content", "")
                if isinstance(content, str) and content.strip():
                    parts.append(content)
            elif btype == "thinking" and block.get("thinking"):
                parts.append(block["thinking"])
    return "\n".join(parts)


def extract_text_only(messages: List[Message]) -> str:
    return "\n".join(_text_blocks(messages))


def extract_user_only(messages: List[Message]) -> str:
    return "\n".join(_text_blocks(messages, roles={"user"}))


def extract_tool_names_only(messages: List[Message]) -> str:
    names = []
    for msg in messages:
        for block in msg.content:
            if block.get("type") == "tool_use" and block.get("name"):
                names.append(block["name"])
    return " ".join(names)


def extract_by_role(messages: List[Message]) -> Dict[str, str]:
    by_role: Dict[str, List[str]] = {}
    for msg in messages:
        for block in msg.content:
            if block.get("type") == "text" and block.get("text"):
                by_role.setdefault(msg.role, []).append(block["text"])
    return {role: "\n".join(texts) for role, texts in by_role.items()}
