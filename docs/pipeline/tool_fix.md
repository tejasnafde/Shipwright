# The tool-fix lane

Failures whose fault layer is the tool interface (`docs/pipeline/classify.md`) are fixed without touching the weights,
in two ways: patches to tool descriptions, recorded in a ledger with one-step rollback, and guards that validate a
tool call before it runs. Both are serving-time changes, so a candidate that only adds them is gated with the same
adapter as the deployed system.

## Tool-description patches

```sh
.venv/bin/python scripts/tool_fix.py runs/<classified run>          # propose (rules, then an LLM)
.venv/bin/python scripts/tool_fix.py --list
.venv/bin/python scripts/tool_fix.py --apply <id> ...               # or --apply-proposed
.venv/bin/python scripts/tool_fix.py --rollback <id>
```

- **Input:** `failures.jsonl` rows routed `tool_fix`, and the tau2 tool specs.
- **Rules first:** concrete example ids in descriptions (the values the model copies) become placeholders.
- **Then an LLM** (`--model`, default Vertex gpt-oss-120b) gets one tool spec and up to three failure excerpts and
  returns patches to `description` or `parameters.<name>.description` only. A patch containing a concrete id is
  rejected and redrafted once.
- **Ledger:** `patches/ledger.jsonl` (committed): one row per patch with the before and after text and their sha256,
  the source failures, and a status history (`proposed`, `applied`, `rolled_back`). Status changes only through
  `tool_fix.py`.
- **Apply:** only `applied` rows take effect, in ledger order, in the prompted-tools layer (tau2 is not edited). Each
  must match its `before` hash, so a conflicting patch fails at start rather than silently. Active patch ids go into
  `run_config.json`; a gate can pin exact ids (`--tool-patch-ids`).

Review every proposal before applying it: the proposer can only reword descriptions, and it can name tools that do not
exist. On round 1, 35 applied patches cut copied example ids from 13 failures to 2.

## Tool guards

With `tool_guards: true`, `src/shipwright/llm/guards.py` checks each student tool call against the retail policy before
tau2 runs it. On a violation the agent is asked again with the reason, at most twice; after that the call runs as
written, so a wrong guard can cost turns but never block a task. Rejected replies are kept in the trace
(`guard_rejections`), never in the conversation, so training data stays clean. Guards never run for the teacher.

| Rule | Rejects |
|---|---|
| `early_transfer` | a transfer to a human before any attempt to authenticate |
| `not_authenticated` | any lookup or change before a user id was found |
| `repeated_failed_call` | a call identical to one that already returned an error (answered with a fallback reply) |
| `repeated_call` | the same lookup again when nothing it reads has changed since (the earlier result is quoted back) |
| `lookup_streak` | another lookup after 10 tool calls with no reply to the customer |
| `made_up_id` | an id that no customer message or tool result contains |
| `wrong_order_status` | a pending-order tool on a non-pending order, a delivered-order tool on a non-delivered one |
| `exchange_other_product_type` | an exchange or modification to a different product type |
| `second_change_to_one_order` | a second exchange or modification of the same order |
| `no_confirmation` | a change unless the customer's latest message says yes to the agent's question |

Before the checks, `guards.repaired_call` fixes one mistake deterministically: a `get_<kind>_details` call whose id the
conversation has only seen as another kind of id is rerouted to that kind's tool, if the agent was shown it (an item id
sent to `get_product_details` becomes `get_item_details`). Replayed over 9,085 recorded tool calls, it fired 175 times,
every original had failed, and it produced no false positive.

The guards are also the definition of a wrong turn for the weight fix: `scripts/build_corrections.py` asks the teacher
for the right next turn wherever a guard rejected the candidate's call (`docs/pipeline/training.md`).

`scripts/check_guards.py runs/<teacher run> runs/<student run>` replays recorded runs through the guards offline: the
teacher's passing runs show the false-positive rate, the student's failed runs show coverage.
