# albion-assistant

Windows automation assistant for Albion laborer runs. Press one key: it collects
the laborer's output, advances the tier if offered, works out which laborer is
ready, hands it the matching journal from the inventory, and accepts.

## Install

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python scripts\laborer-v4.py
```

| Key | Action |
| --- | --- |
| `2` / numpad `2` | Scan the inventory and cache journal slots |
| `1` / numpad `1` / `Ctrl+C` | Run one laborer cycle |
| `Esc` | Panic - abort the cycle in flight |
| `²` (`` ` `` on QWERTY) | Quit |

Press `2` first: the journal cache is what tells v4 which tiers you can act on.

Useful flags:

```powershell
python scripts\laborer-v4.py --selftest   # validate assets, measure timings, exit
python scripts\laborer-v4.py --dry-run    # every step except input injection
python scripts\laborer-v4.py --verbose    # per-step debug logging
python scripts\laborer-v4.py --reset      # discard learned anchors and cache
```

Tuning lives in `laborer-v4.config.json`, written with defaults on first run.
Delete it to start over. Learned state (anchors, journal slots) is in
`laborer-v4.state.json`; both are gitignored as machine-specific.

## How v4 works

**Anchored ROIs.** Every element - take-all, advance-tier, ready, accept, the
laborer portrait - sits in the same dialog, so their relative offsets are
constant. v4 learns them on the first cycle and thereafter searches a small box
instead of the screen. A miss escalates to a coarse-to-fine full search that
re-learns the offset, so the UI moving heals itself.

**Detect once, classify once.** Rather than searching the screen separately for
each of 18 laborer templates, v4 finds *where* the portrait is, then scores all
18 against that single crop.

| | v3 | v4 |
| --- | --- | --- |
| Laborer identification, cold | ~3500 ms | ~210 ms |
| Laborer identification, warm | ~3500 ms | ~10 ms |

**Tier decided by what actually differs.** Measured on these assets: two tiers
of one profession are pixel-identical over 95% of the icon and differ only in a
~25x23 badge box. A whole-icon score puts siblings within **0.004** of each
other, so it cannot decide between them - which is why v3 kept needing patches
here. v4 weights the comparison by how much the siblings disagree at each
pixel, derived from the assets themselves. The runner-up ends up at least 1.8x
worse even under heavy capture noise.

**Identify first, then check the bag.** Candidate reduction narrows the *locate*
step to tiers whose journal is in the inventory, where the cost is. The
*classify* step deliberately uses every template: if only t2 journals are held,
a ready t5-bs still matches t2-bs at 0.996, and a comparison set narrowed to
the inventory has no way to notice. v4 identifies it correctly and then refuses
with "no t5-bs journal in the inventory" - a supply problem, not a misread.

**The accept prompt is the success signal.** Handing over a journal is retried
until the accept prompt appears, not until the inventory slot looks empty. A
successful hand-over can therefore never be clicked twice.

Also: process DPI awareness (v3 only worked at 100% scaling), a foreground
window gate so typing `1` in a browser cannot start a click sequence, a panic
key, per-cycle exception containment, a failure circuit breaker, and distinct
beeps per outcome.

## Sounds

| Sound | Meaning |
| --- | --- |
| One short high | Cycle complete |
| One short low | Laborer not ready |
| Two long | Stop laborer (t6) - handled manually |
| Three low | Ready, but no journal for it in the inventory |
| Descending pair | Could not identify, or the hand-over failed |
| One long buzz | Error / circuit breaker tripped |

## Assets

`assets/laborers/<tier>-<profession>.png`, `assets/journals/<name>.png`,
`assets/menu/*.png`. `mapping.json` maps a laborer to the journal it needs.

Two things `--selftest` will nag about, both worth fixing:

- The laborer crops are 17 different sizes. v4 measures and corrects the offsets
  automatically, but re-cropping them to identical bounds is more robust.
- `t6-imb` and `t6-tk` have no template. They are refused as an unknown tier
  rather than acted on, so this is safe, but capturing them turns a refusal into
  a clean "stop laborer" beep.

## Tests

```powershell
python scripts\tests\test_laborer_v4.py
```

No game, no screen capture, no input: templates are pasted into synthetic
frames, optionally degraded with sub-pixel shift, gain/bias and noise, and the
classifier is checked for correctness rather than for not crashing.

## Older versions

`scripts/laborer-v3.py` and `scripts/laborer-v2.py` still run. v3 remains a
working fallback.
