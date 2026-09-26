"""Compare the same local LLM with and without retrieved tables."""
import argparse
import csv
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
import time

from pydantic import BaseModel
from typing import Literal
from table_rag.answer_generation import _ollama_chat, DEFAULT_ANSWER_MODEL, DEFAULT_OLLAMA_BASE_URL
from table_rag.context_builder import build_context
from table_rag.dense_index import load_dense_index, load_embedding_model
from table_rag.hybrid_retrieval import retrieve_candidates, fuse_weighted
from table_rag.sparse_retrieval import load_bm25
from table_rag.summaries import sha256_file

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / "dataset/generated_evaluation/answer_comparison_questions.json"
OUT = ROOT / "artifacts/answer_comparison"
PROMPT = """Answer the question about this project's synthetic dataset.
Use supplied evidence when available. If you cannot establish the exact answer,
return insufficient_evidence; do not invent dataset values.
Return the requested scalar alone in answer_value (no units, commas, or citations).
Explain briefly in answer. Put supporting table IDs in cited_table_ids.
Never cite tables you have not received. Return JSON matching the schema."""
class Response(BaseModel):
    status: Literal["answered", "insufficient_evidence"]
    answer_value: str
    answer: str
    cited_table_ids: list[str]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare():
    manifest = list(csv.DictReader((ROOT / "dataset/table_manifest.csv").open()))
    specs = [
        (0, ["hospital_name", "report_month", "diagnosis_group"], "admissions_count"),
        (10, ["policing_district", "month", "offence_group"], "arrests_count"),
        (20, ["application", "report_month", "severity"], "incident_count"),
        (30, ["placement_region", "month", "industry_sector"], "placements_completed"),
        (40, ["season", "team"], "goals"),
        (51, ["station_id", "observation_date"], "maximum_temperature_c"),
    ]
    questions, sources, loaded = [], {}, []
    def add(kind, question, expected, table_id, evidence):
        questions.append(dict(question_id=f"answer_{len(questions)+1:03}",
            query_type=kind, question=question, expected_value=str(expected),
            expected_table_ids=[table_id] if table_id else [],
            evidence=evidence, review_status="programmatically_verified_pending_human_review"))
    for i, keys, measure in specs:
        table = manifest[i]
        path = ROOT / "dataset" / table["file_path"]
        rows = list(csv.DictReader(path.open()))
        sources[table["file_path"]] = digest(path)
        unique_rows = [r for r in rows if sum(all(other[k] == r[k] for k in keys) for other in rows) == 1]
        if len(unique_rows) < 2: raise ValueError(f"No unique records for {table['table_id']}")
        row = unique_rows[0]
        filters = {k: row[k] for k in keys}
        assert sum(all(r[k] == v for k,v in filters.items()) for r in rows) == 1
        condition = ", ".join(f"{k}={v}" for k,v in filters.items())
        add("lookup", f"In the synthetic dataset's {table['title']}, what is {measure} for {condition}?",
            row[measure], table["table_id"], dict(filters=filters, column=measure))
        loaded.append((table,rows,keys,measure,unique_rows))
    for table,rows,keys,measure,unique_rows in loaded[:3]:
        # Difference between two fully specified records avoids ambiguous ties.
        a,b = unique_rows[:2]
        describe = lambda r: ", ".join(f"{k}={r[k]}" for k in keys)
        add("comparison", f"In {table['title']}, what is the difference in {measure}: "
            f"record A ({describe(a)}) minus record B ({describe(b)})? Return a signed number.",
            Decimal(a[measure])-Decimal(b[measure]), table["table_id"],
            dict(operation="A minus B", column=measure, A=a, B=b))
    for table,rows,keys,measure,unique_rows in loaded[:2]:
        group = keys[0]; value = rows[0][group]
        selected = [r for r in rows if r[group] == value]
        add("aggregation", f"In {table['title']}, what is the total {measure} for "
            f"{group}={value} across all records in the supplied dataset?",
            sum(Decimal(r[measure]) for r in selected), table["table_id"],
            dict(operation="sum", column=measure, filters={group:value}, rows=len(selected)))
    # The manifest covers 2024/2025; a future forecast is not present.
    assert not any("2035" in json.dumps(t) for t in manifest)
    add("unanswerable", "What will Olive Grove Clinic's exact total admissions be in January 2035?",
        "", None, {"reason":"No future 2035 admissions or forecast is supplied."})
    suite = dict(version=1, purpose="Small illustrative challenge, not a representative benchmark.",
        source_hashes=sources, questions=questions)
    SUITE.parent.mkdir(parents=True,exist_ok=True)
    if SUITE.exists() and json.loads(SUITE.read_text()) != suite:
        raise ValueError("Existing question set differs; preserve it before deliberately rebuilding.")
    SUITE.write_text(json.dumps(suite,indent=2)+"\n")
    return suite

def score(q, response, retrieved):
    answered = response.status == "answered"
    if q["query_type"] == "unanswerable":
        correct = not answered
    else:
        try:
            correct = answered and Decimal(response.answer_value) == Decimal(q["expected_value"])
        except InvalidOperation:
            correct = False
    citations = set(response.cited_table_ids)
    return dict(correct=bool(correct), abstained=not answered,
        citation_ids_valid=citations.issubset(set(retrieved)),
        expected_source_cited=bool(set(q["expected_table_ids"]) & citations),
        # Source-ID checks do not establish semantic support for every claim.
        expected_table_retrieved=bool(set(q["expected_table_ids"]) & set(retrieved)))

def report(rows, config):
    summary={}
    for mode in ["without_rag","with_rag"]:
        selected=[r for r in rows if r["mode"]==mode]
        answerable=[r for r in selected if r["query_type"]!="unanswerable"]
        summary[mode] = dict(completed=len(selected), errors=sum(bool(r.get("error")) for r in selected),
            answerable_correct=sum(r.get("correct",False) for r in answerable),
            answerable_total=len(answerable),
            unanswerable_correct=sum(r.get("correct",False) for r in selected if r["query_type"]=="unanswerable"))
    (OUT/"summary.json").write_text(json.dumps(dict(config=config,summary=summary),indent=2))
    fields=["question_id","mode","query_type","question","expected_value","answer_value",
        "status","correct","abstained","expected_table_retrieved","expected_source_cited",
        "citation_ids_valid","retrieved_table_ids","cited_table_ids","answer","seconds","error"]
    with (OUT/"results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    counts=[summary[m]["answerable_correct"] for m in summary]
    ax.bar(["Without RAG","With RAG"],counts,color=["#8799aa","#298c76"])
    ax.set(ylim=(0,11),ylabel="Correct answers out of 11 answerable questions",
        title="Same local model: effect of retrieved evidence")
    ax.set_yticks(range(12))
    for i,n in enumerate(counts): ax.text(i,n+0.15,str(n),ha="center")
    fig.tight_layout(); fig.savefig(OUT/"correctness.png",dpi=180); plt.close(fig)
    (OUT/"presentation_notes.txt").write_text(
        "Compare identical model, prompt, settings and questions; only supplied evidence changes.\n"
        "6 lookups, 3 comparisons, 2 aggregations, 1 unanswerable question.\n"
        "Show correctness.png and results.csv. Report unanswerable abstention separately.\n"
        "Without-RAG abstention is appropriate for unknown synthetic values, even though it does not answer the question.\n"
        "Numeric scoring checks the structured scalar; review prose for contradictions and source support.\n"
        "Citing an expected table is a source-ID check, not proof of factual grounding.\n"
        "Small assistant-designed set; no claim of statistical significance or general model superiority.\n"
        "Do not omit failed questions. Errors remain failures in the denominator.\n")
    print(json.dumps(summary,indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-only",action="store_true")
    p.add_argument("--local-files-only",action="store_true")
    p.add_argument("--ollama-url",default=DEFAULT_OLLAMA_BASE_URL)
    p.add_argument("--model",default=DEFAULT_ANSWER_MODEL)
    args=p.parse_args()
    suite=prepare()
    if args.prepare_only:
        print(SUITE); return
    OUT.mkdir(parents=True,exist_ok=True)
    for path,expected in suite["source_hashes"].items():
        if digest(ROOT/"dataset"/path)!=expected: raise ValueError("Source changed")
    config=dict(model=args.model,temperature=0,think=False,num_ctx=32768,
        max_output_tokens=8192,alpha=0.8,top_k=3,suite_sha256=digest(SUITE),
        prompt=PROMPT,schema=Response.model_json_schema())
    log=OUT/"responses.jsonl"
    if log.exists(): raise ValueError(f"Existing results preserved: {log}. Move that run before rerunning.")
    artifacts=ROOT/"artifacts"
    sparse=load_bm25(artifacts/"bm25_index.pkl",source_path=artifacts/"sparse_documents.jsonl")
    dense=load_dense_index(artifacts/"dense_index.npz",
        summary_file_sha256=sha256_file(artifacts/"table_summaries.jsonl"))
    encoder=load_embedding_model(local_files_only=args.local_files_only)
    rows=[]
    for q in suite["questions"]:
        s,d=retrieve_candidates(sparse,dense,encoder,q["question"])
        candidates=fuse_weighted(s,d,top_k=3,alpha=0.8)
        context=build_context(q["question"],candidates,ROOT/"dataset/table_manifest.csv",ROOT/"dataset",top_k=3)
        for mode in ["without_rag","with_rag"]:
            retrieved=list(context.table_ids) if mode=="with_rag" else []
            content=context.context_text if mode=="with_rag" else json.dumps({"question":q["question"],"tables":[]})
            row={**q,"mode":mode,"retrieved_table_ids":retrieved}
            start=time.monotonic()
            try:
                raw=_ollama_chat(base_url=args.ollama_url,model=args.model,
                    messages=[{"role":"system","content":PROMPT+"\n"+json.dumps(Response.model_json_schema())},
                              {"role":"user","content":content}],
                    response_schema=Response.model_json_schema(),num_context=32768,
                    max_output_tokens=8192,timeout_seconds=600)
                row["raw_response"]=raw
                if raw.get("done_reason")=="length": raise ValueError("Truncated response")
                response=Response.model_validate_json(raw["message"]["content"])
                row.update(response.model_dump()); row.update(score(q,response,retrieved))
            except (OSError,RuntimeError,ValueError,KeyError) as e:
                row.update(error=str(e),correct=False)
            row["seconds"]=round(time.monotonic()-start,3)
            row["input_context"]=content
            rows.append(row)
            with log.open("a",encoding="utf-8") as f: f.write(json.dumps(row)+"\n")
            print(q["question_id"],mode,"correct=",row["correct"],flush=True)
    report(rows,config)
    print("Saved:",OUT)

if __name__=="__main__":
    main()

