# Table summary preparation instructions ? v1

Purpose: describe each synthetic CSV table for semantic retrieval.
These instructions document an assistant-assisted preparation task, not an
automated API pipeline. Keep the resulting text as a versioned input artifact.

Inputs: the original CSV schema, full-data source profiles, and manifest metadata.
Do not open evaluation questions or qrels during preparation. Prior exposure in
the authoring conversation must be disclosed in release provenance.

For each table:
- Write a concise description of the subject and the dimensions of a record.
- Explain the main available measures and units only where the schema supports them.
- Describe supported comparisons generally, without inventing answers or query examples.
- Prefer observed CSV date/period ranges over conflicting manifest coverage.
- Do not infer unique primary keys, continuous time coverage, or complete entity coverage.
- Do not infer salary frequency, percentage denominators, metric formulas, causes,
  trends, extrema, or relationships between independently generated synthetic measures.
- Do not invent a season where there is no season column.
- State that the data is synthetic. Treat metadata as descriptive, not as official statistics.
- Include the original column names for traceability.
- Keep source-quality warnings and review provenance outside the embedding text,
  except for limitations directly needed to understand what the table contains.
- Use the original CSV, not the summary, as evidence for numerical answers.

Assembly format:
Synthetic {domain} table: {title}. {Assistant-authored description}
Observed {period_column} values span {minimum} to {maximum}.
Columns: {exact column names}.

Omit the coverage sentence if there is no observed time/season field.
When multiple temporal fields exist, describe each separately.
Ranges describe observed bounds, not coverage of every date or every entity.
Review status: assistant_checked_pending_human_review.
Human approval is a separate action and must not be invented.
