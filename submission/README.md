# Hackathon submission

Material for the Lexsi Labs AI Safety and Alignment Hackathon (Track 3, problem statements 8 Self-healing agents, 1 Reliable tool calling, 2 Efficient trajectories and 3 Agentic distillation).
The service itself is documented in the repository's `README.md` and `docs/`; nothing here is needed to run it.

| Path | What it is |
|---|---|
| `WRITEUP.md` | the write-up: problem, approach, libraries, models and data with licenses, experiments, results, limitations |
| `blog.md` | the blog post |
| `img/` | charts and the loop diagram used by the write-up, the blog and the Space |
| `make_charts.py`, `make_loop_diagram.py` | regenerate `img/` from the published numbers (`pip install matplotlib`) |
| `space/` | the results page published as the Hugging Face Space `mitushigoyal5/shipwright` |
| `demo_video/` | the [demo video](https://huggingface.co/spaces/mitushigoyal5/shipwright/resolve/main/video/shipwright-demo.mp4) (published on the Space): storyboard, narration, `make_voiceover.py` and `build_demo.py`, which records the real control room and assembles the MP4 |
| `analysis/tool-fix-proposer.md` | a study: does the loop's tool-fix proposer find the guard fixes that moved the numbers? |

Every headline number in the write-up is checked against the public run files by `python scripts/reproduce.py`.
