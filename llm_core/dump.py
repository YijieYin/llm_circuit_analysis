"""Dump LLM prompts to markdown files for manual use in Colab / Claude.ai / ChatGPT.

Used by the --dump-prompts flag in pipeline scripts. The format is one .md file
per item, with system and user prompts in clearly labelled sections so the user
can paste either piece into whatever LLM interface they're using.
"""

from pathlib import Path


def write_prompt_file(path, item_id, system_prompt, user_prompt,
                      required_keys=None, extra_notes=None):
    """Write a single LLM prompt to a markdown file.

    path: where to write (.md path)
    item_id: short identifier shown as the H1 (cell type, paper id, "src__tgt", ...)
    system_prompt, user_prompt: the two strings to paste
    required_keys: optional iterable — shown as expected-output schema hint
    extra_notes: optional string appended at the end (e.g. usage instructions)
    """
    parts = [
        f"# {item_id}",
        "",
        "Paste the **System prompt** as a custom-instruction / system message if your",
        "LLM interface supports it; otherwise paste both blocks together as one message.",
        "",
        "## System prompt",
        "",
        system_prompt.rstrip(),
        "",
        "## User prompt",
        "",
        user_prompt.rstrip(),
        "",
    ]
    if required_keys:
        parts.extend([
            "## Expected JSON keys in response",
            "",
            *[f"- `{k}`" for k in sorted(required_keys)],
            "",
        ])
    if extra_notes:
        parts.extend(["## Notes", "", extra_notes.rstrip(), ""])
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def safe_filename(s, max_len=120):
    """Make a string safe for use as a filename. Lowercase, alnum + underscores."""
    import re
    s = re.sub(r"[^\w\-]+", "_", str(s)).strip("_").lower()
    return s[:max_len] or "item"