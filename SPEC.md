# Car advisor

Local chatbot over auto.co.il road tests. Answers from retrieved passages. LangGraph, one Claude model, session ids stored in SQLite.

## Persistence

- `data/raw/<slug>.html` and `data/articles/<slug>.md` are the saved articles.
- Chunks, number forms, and car rows live in `data/catalog.db`.
- Vectors live in Chroma at `data/chroma`. Ingest embeds an article only when its text hash or the embedding model changed.
- BM25 is rebuilt from the stored chunk text when the process starts. That is a read of SQLite, not an embedding pass.
- Conversations are LangGraph checkpoints in `data/checkpoints.db`, keyed by the session id the CLI asks for.

## Retrieval

Hybrid. Dense search uses `intfloat/multilingual-e5-large` with the `query:` and `passage:` prefixes added in code (FastEmbed does not add them). BM25 tokenizes Hebrew and also indexes the word with one prefix (ו ה ב ל מ ש כ) removed. The two lists merge with reciprocal rank fusion. A search with no file filter keeps the best hit per article before filling the rest, so one review cannot take every slot.

## Numbers

Every number found in a chunk is stored as its canonical value plus surface forms: digits, thousands separators, `אלף`, `k`, and shekel spellings when the mention is a price or uses `אלף`/`k`. An answer number is accepted when one of its forms is in a cited chunk, in any tool result from this turn, in the user's messages, or in the profile. A number that appears in a tool result keeps the source line from that result, attached automatically. A number that is still missing is rejected with an instruction to cite the find_cars rows, not to delete the prices. `190 אלף` does not match an article that says `189,900`.

## Graph

`START → triage → agent ⇄ act`. Triage returns `{parts, kind, profile_updates}` in one call. `kind` is `direct` for a question about named cars and `recommendation` for a suggestion or a list. One part means no plan. A profile quote that is not a substring of the user's messages is dropped. The agent may call `FinalAnswer` immediately with empty sources. The last step can only submit. `act` rejects a submission whose sources were not returned this turn, whose numbers are not allowed, or whose comparison omits an article that had a passage. The answer's length is not rejected there. The eval checks it.

A direct price or spec is 1 to 3 sentences, at most 60 words. A list is one line per car (name, price, one fact), at most 5 cars and 90 words, and offers details instead of listing the rest. A recommendation names 2 or 3 cars, one or two lines each, then why the first, at most 120 words. A comparison is a short table or 2 to 3 lines per car, at most 160 words. Pros and cons appear only when asked, at most 2 of each per car, and add 40 words to that cap. An answer ends with at most one question mark. `קצר`, `בקצרה`, or `short` stores a short style on the session, and later answers use half those caps.

`run_turn` runs one turn and returns the accepted answer, or nothing if the turn ends without one. A tool call that never received a result is closed before the next model call, so a later message can still be answered. A failed turn returns an error the page can show.

Profile fields: budget, body, fuel, seats, use, must_haves, avoid, other. Each stored value has a strength (`soft` interest or `hard` constraint) and the turn it came from. A later hard statement overrides an earlier soft one. A later soft statement does not override a hard one. The preference profile is included only on recommendation turns. A mismatch with a preference may be mentioned once for that preference, then not again. Two direct questions about cars outside a body preference broaden it (`suv; also considering city`). Otherwise a recommendation asks once whether the user is open to other types and records the answer. Hard constraints are passed to `find_cars` as filters. Soft ones rank matches first.

## Roles

Each catalog row is a variant of a make+model family: `subject` (a trim this article reviews) or `mentioned` (a rival in that file). Extraction sets the role from the model's Hebrew and English names. A claimed subject whose name is not an exact compact substring of the title, the main heading, or the source URL is stored as `mentioned`. Trims are variant fields. A trim name returns that trim's subject rows: Calligraphy is 390,000 and 411,000, Excellence is 350,000 and 358,000. The family name returns every variant.

Aliases are rebuilt from the whole catalog after every extraction or ingestion batch. A key that points at two families is ambiguous. Ingestion flags families within the allowed edit distance and does not merge them. Pairs in `data/confirmed_distinct.json` are not flagged again. At query time, edit distance fixes typos: exact only up to 3 characters, distance 1 for 4–7, distance 2 from 8 up. Two families inside that distance come back as candidates. A make alias, in Hebrew and in English, points at every family of that make. For a make, the distance uses the longer of the query and the alias, so a short spelling such as פג'ו still reaches פיג'ו. One reviewed model of that make is that car. Several reviewed models are listed, and the user is asked which one. Mentioned-only families are not listed as reviewed.

## Tools

`find_cars` returns subject variants only, at most 8, with `file` and a source line. A name that is only a mention comes back as `Not reviewed. Mentioned in: <slug> (as a rival). Search <slug>.md for what that review says.` An unknown name is `I couldn't find "<name>" by that name.` Makes and models within one extra edit are listed after that sentence as close names, not as matches. The reply does not say the car is certainly absent. If the user insists, the next turn searches passages with no car filter instead of repeating the same lookup. A make with one reviewed model is prefixed `We have the <make> <model> review.` A make with several asks which reviewed model. `search_passages` and `compare` take an optional `car`. When it is set, a subject limits the search to that family's files; a mention, an unknown name, or an ambiguous name returns the resolver message and does not search. No `car` searches the files given, or the corpus.

## Chat

`car-advisor` reads a session id, then a question, and prints the finished answer. Traces go to the terminal.

## Page

`car-advisor-web` serves a local page at `127.0.0.1:8000`. The chat calls `POST /api/chat`, which runs `run_turn` and returns the whole answer when the turn is done. Under the answer it shows one sources line, `file:lines`, when the answer cited passages. The browser keeps the chosen session id. A list of saved conversations, excluding eval runs, lets the user continue one or start a new one, and the thread shows that session's earlier turns. Enter sends the message. Shift+Enter starts a new line. `POST /api/articles` takes one article URL, stores it the same way as ingest, and extracts its cars. A URL already in the catalog is not downloaded again.

## Out of scope for now

AWS in the running PoC. The production proposal is a diagram only, in `docs/aws-architecture.md`. A second model. A reranker.
