# Hackathon submission

Material for the Lexsi Labs AI Safety and Alignment Hackathon (Track 3, problem statement 8: self-healing agents).
The service itself is documented in the repository's `README.md` and `docs/`; nothing here is needed to run it.

| Path | What it is |
|---|---|
| `WRITEUP.md` | the write-up: problem, approach, libraries, models and data with licenses, experiments, results, limitations |
| `blog.md` | the blog post |
| `img/` | charts and the loop diagram used by the write-up, the blog and the Space |
| `make_charts.py`, `make_loop_diagram.py` | regenerate `img/` from the published numbers (`pip install matplotlib`) |
| `space/` | the results page published as the Hugging Face Space `mitushigoyal5/shipwright` |
| `analysis/tool-fix-proposer.md` | a study: does the loop's tool-fix proposer find the guard fixes that moved the numbers? |

Every headline number in the write-up is checked against the public run files by `python scripts/reproduce.py`.
