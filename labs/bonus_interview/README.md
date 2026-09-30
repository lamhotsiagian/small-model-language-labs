# Bonus: System Design Interview Preparation

Companion script for the bonus chapter. `estimates.py` prints every
back-of-the-envelope number used in Step 2 ("Estimate the Scale") of the 20
interview cases, built on the same `slmlab.budget` accounting as Chapters 1,
2, 4, 16, and 18.

## Flow

1. Run `python labs/bonus_interview/estimates.py` and keep the output open.
2. Pick a case from the chapter. Answer it aloud with the six-step structure:
   clarify, estimate, architecture, deep dive, trade-offs, improvements.
3. Change one assumption in the matching `caseNN()` function (QPS, context,
   precision, device bandwidth) and explain how the design changes.
4. Draw the three-layer diagram (data, model, serving) from memory and compare
   it with the chapter's figure.

Prices in case 1 and case 15 are illustrative assumptions, not vendor quotes.
