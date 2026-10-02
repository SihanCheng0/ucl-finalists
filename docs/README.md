# Docs

| Document | Read it for |
|-|-|
| [how-it-works.md](how-it-works.md) | The whole process in plain language: the five stages, how to read each screen, what the model found |
| [specs/2026-10-01-ucl-finalists-design.md](specs/2026-10-01-ucl-finalists-design.md) | The pipeline and model design: data sources, features, validation, the local-AI analyst and its checks |
| [plans/2026-10-01-ucl-finalists.md](plans/2026-10-01-ucl-finalists.md) | How the pipeline was built, task by task |
| [specs/2026-10-01-ucl-lab-dashboard-design.md](specs/2026-10-01-ucl-lab-dashboard-design.md) | The UCL Lab design: API, pipeline runner, live season, screens |
| [plans/2026-10-01-ucl-lab-plan-a.md](plans/2026-10-01-ucl-lab-plan-a.md) | How the dashboard was built, task by task |

The specs record why things are the way they are. When a change alters a design decision, update the spec in the
same pull request.

`images/` holds the README screenshots. To retake them in light mode, start UCL Lab (`make web`) and run:

```bash
uv run --with websocket-client python scripts/screenshots.py
```
