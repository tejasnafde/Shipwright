# Narration

One section per scene, read by `build_demo.py` (the `## <scene id>` lines are its keys; everything under one is said in
that scene). Every number here is one the control room shows from the round 16 run files; check it there before
changing it.

## title
Most AI agents fail quietly. A wrong tool call, an invented order number, and nobody notices until the same mistake
happens again tomorrow. Shipwright is a service that makes a small support agent find its own mistakes, fix them, and
prove the fix before it ships.

## loop
The agent is Tiny Aya, a three billion parameter open model, answering retail customers on tau-two bench. Every round,
Shipwright runs it, detects the conversations it failed, and classifies why each one failed. A tool problem is fixed in
the tool layer. A reasoning problem is fixed in the weights, with a stronger teacher showing the right move. Then an
eight-check gate decides whether the new version may replace the old one.

## control_room
This is the control room. It replays a real round from the files the service wrote; nothing here is simulated. Round
sixteen ran with no person in the loop, for three dollars and eighty-nine cents, and the gate said yes, eight checks out
of eight.

## what_changed
What changed? Its tool calls fail seven times less often, forty-four percent down to six, and tickets resolved rose
from ten percent to fifteen. Made-up ids disappeared entirely. It is honest about the cost too: it now skips more policy steps, and its conversations run longer. Here
is one held-out task, side by side. The base model invents an order number, and the store answers: order not found. The
healed student verifies the customer first, looks up the real orders, and solves the task.

## corrections
Where did that come from? The previous candidate replayed the training tasks with policy guards switched on. Every call
a guard blocked became a lesson. Here, the student made up an order number. The teacher, seeing exactly what the
student saw, looks up the real customer instead. Fifty-nine new corrections this round, for twelve cents of teacher time,
two hundred and thirty-five across all three correction rounds, and only the corrected turn is ever trained.

## gate
Then the gate. Both systems play the same twenty held-out tasks, three times each. Task success, consistency, the
failures it targeted, new kinds of failure, and general tool calling, which rose from forty-five to seventy-three
points. All eight checks must pass.

## history
And the gate means it. Twelve gates, eleven refused, each for a reason the traces explain. AgentTune's own deployment
check would have approved ten of those eleven. The loop shipped only when everything held.

## outro
Shipwright. One command, an unattended round, a gate that can say no, and every number recomputed from public run
files. Built on AgentTune and AuditKIT for the Lexsi Labs hackathon.
