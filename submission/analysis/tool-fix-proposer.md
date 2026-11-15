# Does the loop's tool-fix proposer find the fixes a person wrote?

Checked 2026-11-13. How the proposer works and how to run it: `docs/pipeline/tool_fix.md`.

The biggest gain so far (gate 8: tool-call error rate 0.60 to 0.08, "Product not found" 59 to 0)
came from tool guards that a person wrote after reading traces, not from the loop. To make an
honest self-healing claim, we ran the loop's own proposer (`scripts/tool_fix.py`: rules, then
Vertex gpt-oss-120b, then validation) on the failures that motivated those guards, in
propose-only mode (a separate ledger per input, `patches/ledger.jsonl` untouched, nothing
applied), and compared.

## Inputs

| Input | What it is | Failed | Routed `tool_fix` | Evidence in it |
|---|---|---|---|---|
| A `r4-cand-train` | round 4 adapter on 47 train-pool tasks (the failures the loop read for rounds 5 to 7) | 36 | 19 | 218 repeats of an identical call, in 25 of 47 runs; 8 "Product not found" |
| B gate 7 candidate | `r7-weight` on the 20 held-out tasks x 3, guards on (read for this analysis only, not used as training data) | 52 | 10 | 59 "Product not found", all an item id given to `get_product_details` |
| C what-if | the 10 runs of B with an item-id error, re-routed to `tool_fix` by hand | 10 | 10 | the same 59 errors |

Both runs had no tool patches active, so the proposer started from the plain tau2 tool list (an
empty ledger). Spend: 3 Cloud Run executions (`shipwright-tejas-mwmdf`, `-kd6pr`, `-4cqsk`),
17 Vertex calls, about 16k tokens in and 4k out, **$0.003**.

Output: A 21 patches (12 tools), B 8 patches, C 2 patches. The validator rejected 12 drafts for
concrete ids (`#W1234567`, `1008292230`); each was redrafted.

## Hand-written fixes against the proposals

| Hand-written fix (commit) | Proposer result |
|---|---|
| Item-id repair: `get_product_details(<item id>)` runs as `get_item_details(item_id)` (3c1f9cb). Gate 8: "Product not found" 59 to 0. | **Diagnosis yes, remedy no.** B `tp-3c8a1b67ea` and C `tp-8087b89fa3`: "Do not use an item_id; those belong to product variants". No proposal names `get_item_details`. C `tp-9895385afd` says to tell the customer the id "may be incorrect" and to use "a search tool", which does not exist. The tau2 text already says "the product id is different from the item id", and gate 7 shows that text does not stop the error. Run A (as routed) does not mention it. |
| `repeated_call` guard: the same lookup with nothing changed since is refused, with a fallback reply (aefdef3) | **Intent yes, as text.** A `tp-b4d7f019bb` (`get_order_details`: "Use only once per conversation; if you already have the order info, do not call again"), A `tp-634baf6851` and `tp-ee53385daf` (`get_item_details`: "only once per item"), B `tp-886d599ca1` ("do not call repeatedly for the same id"). A description cannot enforce it, and "once per conversation" is wrong for a customer with two orders or a re-check after a change (the guard allows both). |
| The repeat warning carries the earlier result (3c1f9cb) | **Missed.** It changes the harness, which a description patch cannot express. |
| One change per order, same product type (aefdef3) | **Restated.** A `tp-0a170de5fc`, `tp-e8540cc4bf`, `tp-89d924d99d`, B `tp-b3ab6cdddb`. The tau2 descriptions already say both. |
| Earlier guards (d2a0c06): order status fits the tool, transfer only when needed | **Intent yes, as text.** "First verify the order status with get_order_details; if not pending, do not call this tool" (A `tp-0a170de5fc`, `tp-172a6b9ce8`, `tp-0c28db80e5`); "Use only when the user explicitly requests a human" (A `tp-49cd19f87e`). Authenticate first: not proposed. |

## Why it missed the fix that mattered

1. **The router hid the evidence.** In B, 54 of the 59 item-id errors (9 of the 10 runs) are in runs routed to
   `weight_fix` (52) or `no_fix` (2); their labels are `unclassified` or `loop`, since no rule labels a not-found
   lookup with an item id. The tool lane only reads `tool_fix` rows.
2. **The excerpt hid it too.** An excerpt shows the first 2 calls of a tool, and the item-id call
   usually comes after a successful product lookup: in only 2 of the 10 runs is it among the
   first 2. Even in C, the model saw mostly successful lookups.
3. **The output space is description text.** The fixes that worked are code: a reroute and an
   enforced refusal. The proposer can only reword a description, and gate 7 had already shown
   that rewording does not stop this model.
4. **`context_overflow` routes to `tool_fix`** (fault layer `tool`), so the proposer blames the
   tool called at the first wrong turn. All 10 `tool_fix` rows in B are `context_overflow`.
   `failing_tools` also reads `dropped` (from "dropped 6 messages") as a tool name; it is
   dropped later as unknown, so it is harmless.

## Defects in the proposals that validation does not catch

- **Tools that do not exist**, in 6 of 31 patches: "a search/list tool" (B `tp-886d599ca1`),
  "a summary tool" (B `tp-14ae6b84cf`), `check_order_items` (B `tp-abc3fe7499`), "payment or
  shipment tools" (A `tp-b4d7f019bb`), "a batch lookup tool" (A `tp-634baf6851`), "a search
  tool" (C `tp-9895385afd`). The round 1 ledger has one already applied (`tp-ce2c57158a`:
  "use the search_user tool").
- **Real-looking examples**: A `tp-8f4c680b05`, `tp-e33d7886fb`, `tp-a380dec314` use `'John'`,
  `'Doe'`, `'12345'`. The id patterns miss names and 5-digit zips, and the failure behind them
  is the student inventing zip `90210`.
- **Wrong advice**: B `tp-14ae6b84cf` limits `get_order_details` to status questions ("Do not
  request full address or item list"); the tool has no such option, and changes need the item
  list. A `tp-a109855d8c` says "ask the model to summarize".

## Possibly useful, not done by hand

- Shorter descriptions to save context (B `tp-95ff6e5abc`, `tp-431290ccf8`). The fixed prompt
  is 3,404 of 8,192 tokens (gate 7), but description length is small next to the
  product listings, so the gain is likely small. Not measured.

## What to claim in the write-up

- The loop finds and counts the failures (detect, classify, route) and its proposer names the
  right problem in text for 2 of the 4 guard changes (repeated lookups; product id vs item id).
- The fixes that moved the numbers (item-id reroute, enforced repeat refusal with the earlier
  result) were written by a person from the loop's traces and counts. The proposer did not
  produce them, and its lane cannot: it only edits descriptions.
- Next, to close the gap: a "repair" output type (`tool A with an id of kind K becomes tool B`)
  checked against the trace, a rule label for a not-found lookup whose id is a known id of
  another kind, excerpts that show the failing call instead of the first call, and a validator
  check that every tool named in a patch exists.

## Reproduce

```sh
GC="gcloud --configuration=lexsi --project=lexsi-hackathon"
$GC run jobs execute shipwright-tejas --region=asia-south1 --wait \
  --args=--fetch=ishank/r4-cand-train,tool_fix.py,runs/r4-cand-train,--ledger=runs/tf-check-r4/proposed_ledger.jsonl,--round=4
$GC run jobs execute shipwright-tejas --region=asia-south1 --wait \
  --args=--fetch=tejas/gate7-cand,tool_fix.py,runs/gate7-cand,--ledger=runs/tf-check-gate7/proposed_ledger.jsonl,--round=7
$GC run jobs execute shipwright-tejas --region=asia-south1 --wait \
  --args=--fetch=tejas/gate7-itemid,tool_fix.py,runs/gate7-itemid,--ledger=runs/tf-check-gate7-itemid/proposed_ledger.jsonl,--round=7
```

`gate7-cand` is a copy of `gate/round-7/r7-weight/r7-weight/` from the runs dataset on the Hub.
`gate7-itemid` keeps only its runs with a "Product not found" from `get_product_details`, with
`route` set to `tool_fix` and `first_wrong_turn` at the first such call. The proposed ledgers are
in `gs://lexsi-hackathon-runs/tejas/tf-check-*/proposed_ledger.jsonl`.
