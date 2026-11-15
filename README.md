# Car advisor

Local chatbot over auto.co.il road tests. LangGraph, one Claude model, session ids.

The short explanation of the approach is in [APPROACH.md](APPROACH.md). The proposed production architecture is in [docs/aws-architecture.md](docs/aws-architecture.md). The PoC itself does not use AWS.

```bash
uv sync
uv run python -m car_advisor.ingest    # download, clean, embed. Skips articles already stored.
uv run python -m car_advisor.extract   # fill the car catalog. Needs the API key.
uv run car-advisor
uv run car-advisor-web                 # page at http://127.0.0.1:8000
```

The CLI and the page both return the answer when the turn finishes. Traces go to the terminal.

Put `ANTHROPIC_API_KEY` in `.env` before extracting the catalog or chatting.

Saved article text is copyrighted. Keep this repo private.

At about 1,000 articles the catalog filters first, ingest only embeds articles whose text changed, and Chroma holds the vectors on disk.
