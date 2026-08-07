# Apila Sports — Proposal v2 (Backtest-First)

## 0. What changed from v1, and why

The v1 proposal was engineered well but validated backwards. It planned to answer *"does this methodology have predictive value?"* by building the full live system and then forward-testing for a season (500+ games). That's slow, expensive, and — for a personal betting tool — answers the wrong question.

Three structural changes:

1. **Validate against history first, before building the live stack.** You already have thousands of completed games. Run the engine against them and get your answer this week, not in October.
2. **The opponent is the betting market, not "the games."** For a betting tool, "60% accuracy" is close to meaningless. The only metric that matters is whether the model's probabilities beat the market's implied probabilities by enough to overcome the vig. Everything is re-anchored to that.
3. **Success = positive ROI after vig in a walk-forward backtest, confirmed by closing-line value.** Not accuracy. Accuracy and profitability are different things, and it's entirely possible to be "right" most of the time and lose money.

The v1 build (live ingestion, prediction locking, versioning, dashboards) is still good — it just moves *after* the validation gate, and only gets built if Phase 0 says there's an edge.

---

## 1. Objective

Determine, as cheaply and honestly as possible, whether a deterministic rating engine can produce betting-relevant probabilities that beat the closing market after vig — and only build a live product if the evidence says yes.

The primary risk this proposal is designed to kill early: **spending a season funding a losing system one bet at a time because the dashboard looked convincing.**

---

## 2. Success criteria (reframed)

v1 defined success as "operational automation + ~500 predictions showing higher-confidence outperforms lower-confidence." That's necessary but not sufficient for a betting tool. Revised gate:

**Phase 0 passes only if all of these hold on a held-out test set never touched during tuning:**

- Model's win-probability estimates beat the **naive baselines** (pick home team; pick better record) on a proper scoring rule (Brier score / log loss), not just on raw accuracy.
- Simulated flat-stake betting on the model's positive-EV picks returns **ROI > 0 after vig** across thousands of historical games.
- The model shows positive **Closing Line Value (CLV)** — it systematically beats the closing line. CLV is the single best leading indicator of real edge; if you consistently get better-than-closing prices, you have edge even before results are counted.

If any of these fail, **stop.** The correct outcome of a failed Phase 0 is finding out for free.

**Explicit non-goal:** hitting some accuracy number. It's trivial to hit 60%+ accuracy by picking favorites, and favorites are priced so that being 60% right on them still loses money after juice.

---

## 3. The reality check (read this before building anything)

For personal betting, be clear-eyed about the base rates:

- **The vig sets the bar at ~52.4%** to break even at standard -110 pricing. Profit needs more.
- **The closing line is extremely efficient.** It aggregates thousands of sharp bettors and pro syndicates with better data and models than a hand-weighted rule engine. Beating it is the actual task, and it's hard enough that quant shops with paid feeds grind for thin edges.
- **MLB is one of the noisiest sports to predict game-by-game.** Even good models cap around 58–60%, which may be indistinguishable from the naive baseline. A "59.7% MLB" result is not a failure — it may be the physics of the sport. **Lead validation with basketball**, where the better team wins more reliably, so any genuine edge is easier to detect.
- **Backtests that look great are usually overfit.** If Phase 0 looks profitable, be suspicious first, not excited.

None of this means don't build it. It means build the cheap validator first and let it tell you the truth.

---

## 4. Phase 0 — Backtest & Validation Harness (the new core)

This is the whole project's center of gravity. Roughly a week of work, no frontend, no scheduler.

### 4.1 Lookahead bias — the discipline that makes or breaks it

Every feature for a game on date **D** must be computed using **only data available before D**. No season-end aggregates, no "final team rating" applied retroactively. A team's rating in a July game must be its rating *as of that July date*.

This is the most common way backtests lie. If the numbers look magical, assume leakage until proven otherwise. Build the point-in-time feature store first and test it deliberately.

### 4.2 Historical closing odds ingestion

This is the hardest data to source and the most important — without it there is no market benchmark and no profitability test. Options, roughly in order of reliability:

- Paid historical odds API (cleanest, worth it for validation).
- Public archives / datasets (e.g. Kaggle historical odds dumps, sportsbook archive sites).
- Scraping (fragile, gray-area — last resort).

Ingest **closing** odds specifically (the price right before tip/first pitch), because that's the efficient number you must beat.

### 4.3 Devig — market implied probability

Convert odds to implied probability and remove the vig so you're comparing against the market's *true* probability estimate:

- American -110 → implied = 110 / (110 + 100) = 0.524.
- Both sides sum to >1 (that's the vig). Normalize the two sides (or use a Shin/logarithmic method) to get devigged probabilities that sum to 1.

### 4.4 Baselines to beat

Compute these on the same games so the model has something honest to be measured against:

- **Home team always** (surprisingly strong).
- **Better season-to-date record.**
- **Market devigged probability** (the real opponent).

If the engine can't beat the first two decisively and stay competitive with the third, there's no story.

### 4.5 Betting simulation (the true metric)

For each historical game:

1. Compute model probability for each side (point-in-time features → rating → probability).
2. Compute break-even/implied probability from that game's closing odds.
3. Bet a side only when **model_prob > implied_prob** (positive expected value). EV per unit = `model_prob × decimal_odds − 1`.
4. Stake flat units first (Kelly later, once edge is proven — Kelly on a fake edge just blows up faster).
5. Track bankroll, **ROI**, and per-confidence-tier ROI across the full set.

ROI ≤ 0 over thousands of games → no edge → stop.

### 4.6 Probability mapping must be *fit*, not guessed

v1's rating-difference → win-probability conversion, and the confidence-tier thresholds (≥70% = High, etc.), were guessed. Guessed thresholds produce miscalibrated probabilities, which quietly destroys any EV calculation. Fit the rating-difference → probability mapping (e.g. logistic regression on rating diff) on the **training** set and freeze it before touching the test set.

### 4.7 Validation methodology (anti-overfitting)

- **Split by time, never randomly** — random splits leak future into past. Use **walk-forward**: train on seasons 1..n, test on season n+1, roll forward.
- **Hold out one full season you never look at** during any tuning or weight adjustment. It's the only number you're allowed to believe.
- v1's "tune v1.0 → v1.1 based on 500 games" is exactly where overfitting happens. Retuning on the same data you evaluate on manufactures fake improvement.

### 4.8 Sample-size caveat (why backtest, not forward-test)

500 games across 2 sports × 4 confidence tiers leaves ~60–80 games per bucket. A "72% High Confidence" figure off 80 games carries roughly ±10 points of error — you literally can't tell 72% from 62%. Trustworthy calibration curves need **thousands per bucket**, which only history can provide cheaply.

### 4.9 Phase 0 metrics dashboard (minimal, script output is fine)

- Accuracy vs each baseline
- **Brier score / log loss** (proper scoring — rewards calibrated probabilities, not just correct sides)
- Calibration curve **with confidence intervals**
- **ROI after vig**, overall and by confidence tier
- **CLV** — % of bets that beat the closing line and by how much

### 4.10 Go / No-Go gate

Only if Section 2's three conditions hold on the held-out season do we proceed to Phase 1. Otherwise the project's job is done — it saved the bankroll.

---

## 5. Data architecture (kept from v1, extended)

The provider-independent design from v1 was the right instinct — keep it, add odds as a first-class citizen:

```
DATA PROVIDERS  (stats + closing odds)
      ↓
Data Normalization
      ↓
Sports Database (point-in-time aware)
      ↓
Feature Engine (as-of-date only)
      ↓
Prediction Engine (versioned)
      ↓
Betting Sim / Market Comparison
```

- **MLB stats:** pybaseball / Baseball Savant (pitching: ERA, FIP, WHIP, K%, BB%, recent starts; batting: OPS, wOBA, wRC+, runs/game).
- **Basketball stats:** provider depends on league (nba_api for NBA; historical CSVs for others). WNBA/NCAA as data allows; NBA offseason means don't depend on it exclusively — good, basketball is the validation lead anyway.
- **Odds:** new required provider. Store closing line per game.

---

## 6. Prediction engine (kept, with corrections)

Keep the deterministic weighted design and the separate MLB engine (pitcher-weighted). Two corrections:

- The v1 weights (basketball: 25% team strength / 15% form / …; MLB: 25% starting pitcher / 20% offense / …) are **priors, not truth.** Fine as a starting point; don't defend them.
- Convert rating difference to probability via a **fitted** mapping (Section 4.6), not a hand-drawn curve.

Versioning (Engine v1.0, v1.1, …) stays — but every version is judged only on the held-out set, never on the data it was tuned on.

---

## 7. Confidence engine (kept, with caveat)

Keep the tier concept (High / Moderate / Low / Too Close). But: the exact thresholds are guesses until recalibrated on real distributions, and each tier's reported hit rate is only as trustworthy as its sample size and error bar. Report tiers *with* confidence intervals or they'll mislead you.

---

## 8. Phase 1 — Live Product (only if Phase 0 passes)

This is essentially the v1 build, now downstream of a proven edge. Nothing here changes if the engine has no edge — which is the whole point of gating it.

- **Automatic schedule updates** (every ~6h: fetch fixtures, add/update, pull results, score prior predictions).
- **Prediction refresh** (recompute as new info lands — e.g. confirmed starting pitcher; latest pre-game prediction is official).
- **Prediction locking** at game start (store prediction, probability, confidence, timestamp, engine version, input snapshot — immutable). Now doubly important because live results must match backtest conditions to stay comparable.
- **Results & accuracy tracking** + **calibration tracking** (both, as v1 correctly insisted).
- **Live CLV tracking** — keep comparing your locked probability to the actual closing line; it's your earliest live signal that the edge is real or gone.
- **~5 screens:** Home (today's edges), Games, Game Analysis, Results, Performance.
- **Stack:** Next.js + Tailwind (frontend), Python + FastAPI (backend), PostgreSQL, Pandas/Polars, Cron/GitHub Actions scheduler, GitHub. No LLM/ML infra. Biggest ongoing cost is data (now including odds), not compute.

---

## 9. Phase 2 — ML (unchanged intent)

Phase 0 + Phase 1 produce the genuinely valuable asset: your own dataset of `point-in-time features → model probability → market probability → actual result`. Only then does ML make sense (XGBoost/LightGBM to learn which features actually matter), and only adopted if it beats the rule engine **and** the market on the held-out set. Same gate, higher bar.

---

## 10. Milestones

- **M0 — Point-in-time feature store** (basketball first). Prove no lookahead bias.
- **M1 — Historical odds ingestion + devig.** The market benchmark exists.
- **M2 — Prediction engine + fitted probability mapping.** Train on train set only.
- **M3 — Betting simulation + baselines.** ROI, Brier, CLV on held-out season.
- **M4 — Go/No-Go gate.** Decide honestly.
- **M5+ — Live product (Phase 1)** *only if M4 passes.*

---

## 11. One line on bankroll

Even a validated edge is a slow, variance-heavy grind, not a paycheck. Whatever Phase 0 says, size stakes to survive long losing streaks (flat units or fractional Kelly), and treat the model as a filter for +EV spots — not a license to bet more.
