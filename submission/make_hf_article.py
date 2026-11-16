"""Turn submission/WRITEUP.md into a Hugging Face article (submission/hf_article.md).

    python submission/make_hf_article.py

HF's article editor finds no relative files and does not render <picture> or GitHub callouts, so: every image becomes
a plain markdown image served from the Space (which hosts all of them), the stat cards become a table, GitHub
callouts become block quotes, relative links point at the repository, and the title moves to the article's title field.
"""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPACE_FILES = "https://huggingface.co/spaces/mitushigoyal5/shipwright/resolve/main/"
REPO_FILES = "https://github.com/tejasnafde/Shipwright/blob/main/"


def picture_to_image(match: re.Match) -> str:
    """A light/dark <picture> becomes one markdown image, the light version, served from the Space."""
    source = re.search(r'<img src="([^"]+)" alt="([^"]*)"', match.group(0))
    path, alt = source.group(1), source.group(2)
    return f"![{alt}]({SPACE_FILES}{path})"


def cards_to_table(match: re.Match) -> str:
    """The HTML stat cards become a markdown table: number, what it is, context."""
    cells = re.findall(r"<h2>(.*?)</h2>(.*?)<br><sub>(.*?)</sub>", match.group(0))
    rows = [f"| **{number}** | {label} | {context} |" for number, label, context in cells]
    return "| | | |\n|---|---|---|\n" + "\n".join(rows)


def convert(text: str) -> tuple[str, str]:
    title = re.match(r"# (.*)\n", text).group(1)
    body = text[text.index("\n") + 1:].lstrip("\n")
    body = re.sub(r"<table>.*?</table>", cards_to_table, body, flags=re.S)
    body = re.sub(r"<picture>.*?</picture>", picture_to_image, body, flags=re.S)
    body = re.sub(r"> \[!(IMPORTANT|NOTE|TIP|WARNING)\]\n", "", body)
    # Relative links ([text](blog.md), [text](../scripts/x.py)) point at the repository.
    body = re.sub(r"\]\((?!https?://|#|mailto:)(\.\./)?([^)]+)\)",
                  lambda m: f"]({REPO_FILES}{'' if m.group(1) else 'submission/'}{m.group(2)})", body)
    return title, body


if __name__ == "__main__":
    title, body = convert((HERE / "WRITEUP.md").read_text())
    (HERE / "hf_article.md").write_text(body)
    print(f"title: {title}\nwrote {HERE / 'hf_article.md'} ({len(body.split())} words)")
