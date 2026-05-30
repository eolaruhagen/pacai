# Defensive Prototype Plan

## Summary
Create `pacai/student/captureTEST.py` as a copy of the current `capture.py`. Improve only the existing `DefensiveAgent`; leave the offensive class, its features, and its weights unchanged.

Also create `plan-docs/defensive-agent-plan.md` documenting both this prototype and the later unified-agent ideas.

## Prototype Changes
- Keep the current one-attacker, one-defender team structure.
- Preserve the existing defensive food-awareness features.
- Make `food_distance_sum` apply only when no invader is on our side. Once an invader enters, the defender should stop hovering around food and commit to the threat.
- Keep the existing scared behavior: if scared and within `SAFE_INVADER_DISTANCE = 8`, increase distance from the invader without continuing to run farther once safely separated.

## New Defensive Features
- `distance_to_approaching_entry`
  - During calm states, compute the open home-side border entrances.
  - Find which entrance is closest to either visible opponent.
  - Reward the defender for moving toward that entrance.
  - Keep food-cluster protection active as a secondary calm-state signal.
- `distance_to_threatened_capsule`
  - A capsule is threatened if an invader is within `CAPSULE_GUARD_RANGE = 6`.
  - If the defender is not scared, prioritize moving toward the threatened capsule instead of directly chasing the invader.
  - If the defender is scared, scared-distance behavior takes precedence.
- `oscilating_action`
  - Reuse the existing reverse-action pattern from the offensive extractor.
  - Penalize immediately reversing direction only during calm patrol states.
  - Do not apply this penalty while chasing an invader, guarding a threatened capsule, or escaping while scared.

## Initial Weights
Keep all existing defensive weights unchanged. Add simple starter values only so the prototype can be tested before the optimizer runs:
- `distance_to_approaching_entry = -10.0`
- `distance_to_threatened_capsule = -45.0`
- `oscilating_action = -15.0`

These values are placeholders, not tuned results.

## Documentation
The new plan document should clearly separate current work from future work.

Document as future work only:
- Replace separate agent classes with one shared flexible agent class.
- Start both agents in defense mode.
- Switch to offense after either `2` confirmed defensive stops or `450` total game turns.
- Wait until the home side is clear before switching.
- Once attack mode starts, keep it permanent overall; temporarily send the closest attacker home when an invader appears.
- With two defenders, collapse on one invader or split when there are two.
- During a capsule rush, assign one defender to guard the capsule and the other to chase.
- Leave cross-agent communication implementation to the teammate working on that layer.
- Keep offensive lane separation as a later experiment.

## Verification
- Confirm `capture.py` is untouched.
- Run syntax and style checks on `captureTEST.py`.
- Compare current `capture.py` and prototype results with the `1200`-turn cap.
- Run at least `30` games as red and `30` as blue against baseline.
- Test known weak boards: `capture-crowded`, `capture-office`, `capture-alley`, `random-4`, `random-8`, and `random-10`.
- Watch for fewer reversals, earlier entrance coverage, fewer capsule wipes, and improved scores by turn `1200`.

## Assumptions
- The prototype remains intentionally small and reuses existing helpers wherever possible.
- Comments stay sparse and natural; no deliberate typos or artificial degradation.
- No offensive feature changes, unified-class work, shared-memory implementation, or deep weight tuning are included in this prototype.
