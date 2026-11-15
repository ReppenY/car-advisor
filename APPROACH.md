# Approach

The chatbot answers from road tests, not from the model's memory of cars. The model decides what to look up and how to say it. Code decides which cars exist, which numbers are allowed, and when an answer is finished.

## Architecture

One Claude model. Everything else runs locally: the page, the CLI, SQLite, and Chroma. How the same product would run in production is a separate proposal, in [docs/aws-architecture.md](docs/aws-architecture.md). The PoC does not use AWS.

```
article URL
  → save HTML, clean to markdown
  → chunk, record number forms, embed
  → SQLite catalog + Chroma

question
  → triage (split the question, note a preference)
  → agent ⇄ act, at most 6 steps
  → accepted answer
```

Triage reads the message once. A question about named cars is direct. A request for a suggestion is a recommendation. A message with several parts becomes a plan. A single part does not.

The agent then calls tools. `find_cars` looks up reviewed cars in the catalog. `search_passages` retrieves what those reviews say. `compare` runs that retrieval once per file, so one review cannot fill the whole answer. Independent calls run together. At the last step the agent can only submit.

The reply stays short: one to three sentences for a price or a spec, a few lines for a list, a recommendation, or a comparison, and at most one follow-up question. Asking for a short answer keeps later replies in that session shorter. The page shows one sources line under the answer.

`act` checks the submission before the person sees it. Sources must be passages returned on this turn. Every number must appear in a cited passage, a tool result from this turn, the user's own words, or the saved preference. A comparison must cite each file that had a hit. A rejected answer goes back to the model with the specific problem. A bad schema does the same. The turn ends when an answer is accepted, or when the step limit is reached.

Chat memory is local. Each conversation — the messages, the preference profile, and a short-answer style if one was asked for — is a LangGraph checkpoint in `data/checkpoints.db`, keyed by session id. The browser stores only that id in `localStorage`, so a refresh reopens the same chat. The sidebar lists those saved chats, newest first, and leaves eval runs out. A new conversation is a new id; the old one stays in the file. The CLI uses the same file. A turn sends the thread to the model, then writes the result back to that local checkpoint. There is no remote chat store.

## What the code owns

Hebrew names and prices are where a model invents. Names resolve through aliases and a short edit distance: a make with one reviewed model answers about that model, a make with several asks which one, and a close name is offered as a suggestion rather than treated as a match. Prices are stored as digits and as the forms the articles use (`129,000`, `129 אלף`). `190 אלף` does not pass for an article that says `189,900`.

Preferences are remembered with the user's words and a strength. "Only" or "must" is a hard filter on the next recommendation. A softer interest only ranks. A direct question does not bring the preference back into the answer. A later hard statement overrides an earlier soft one.

## Reviewed cars and rivals

Each car in an article is stored as a subject, the car being tested, or a mention, a rival named in passing. Only subjects are recommended. A question about a rival, such as the Ford Explorer, is answered from the review that names it. A car in no article comes back as a name that could not be found, with close names listed as suggestions.

## Decisions backed by measurement

Name matching uses a distance that grows with the length of the name. A flat distance of 1 or 2 matched "CX-5" to the Geely EX5, a wrong confident answer. On 52 labeled names the length-scaled rule had no wrong confident match, and precision and recall were both 1.

On 18 labeled queries, BM25 was stronger on numeric mentions, trims, and short model names. Embeddings were stronger on Hebrew and on descriptive wording. Dense search had a relevant passage in the top 4 more often (0.83) than hybrid (0.72). End to end, on 31 agent cases and one run each, hybrid was the only method that kept every checked fact (31/31, against 29 for dense and 27 for BM25). That is the fact check, not a pass on every behavior. Hybrid stays.

Two unseen reviews, the Kia EV5 and the Honda HR-V, were ingested with no code changes. Subject roles, prices, and name matching were all correct, and seven rival models, including four Kia siblings, stayed mentions. The sample is two articles, so these are counts.

## How it's evaluated

Each layer is scored on its own. Unit tests cover names and roles. The agent eval checks which tools were called and with what arguments, then whether every number traces to a source retrieved on that turn. Retrieval is measured separately. Multi-turn memory has its own cases. Deterministic checks come first. An LLM judge scores only the free-text behavior notes, and those scores stay out of the result until they have been labeled by hand. Bugs found by hand were added as eval cases.

## Limitations

Most of these numbers are one run on a small set. They are directional. The misses that remain are mostly behavioral, such as asking when a name is ambiguous and adapting a recommendation after the preference has widened, not failures of retrieval. Car filters run in Python over the catalog rows. At a much larger catalog they belong in SQL. New articles are added by URL. In production a scheduled crawler would add them, as in the [AWS diagram](docs/aws-architecture.md).

## Past eight articles

Ingest skips an article whose text and embedding model have not changed. Adding a URL from the page uses that same path. Search filters by the resolved car before it ranks passages, and a search across reviews keeps the best hit from each article before filling the rest. Dense search (`intfloat/multilingual-e5-large`) and BM25 are merged with reciprocal rank fusion. BM25 is rebuilt from the stored chunks when the process starts. Chroma holds the vectors on disk.
