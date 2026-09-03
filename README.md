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

## v5: starting a cycle with a left click

```powershell
python scripts\laborer-v5.py
```

Same engine, same assets, same config - v5 only changes how a cycle *starts*.
`"` (the `3` key) arms and disarms a left-click trigger; while it is armed, a
click in the game runs a cycle, and `&` / `1` keeps working alongside it. Off,
the action key is the only way in, exactly as in v4. A short rising beep means
armed, a falling one means off - the console is behind the game.

| Key | Action |
| --- | --- |
| `"` / numpad `3` | Arm / disarm the left-click trigger |
| everything else | As v4 above |

The trigger fires on the button coming *up*, not going down: the engine's first
act is to inject clicks of its own, and doing that while the physical button is
still held hands the game a press with no release in between. A click only
counts while the game is the foreground window and the pointer is inside its
rect, and no click counts until the button has been seen released - which is
what stops the engine's own injected clicks from starting another cycle.

Your click is also what *opens* the laborer dialog, and the dialog animates in,
so a cycle starts one second later (`click_trigger.open_delay`) rather than on
the frame the click landed - otherwise the engine goes looking for take-all on
a panel that is not drawn yet, finds nothing, and the empty cycle counts
against the failure breaker. `Esc` still works during that second. The action
key skips the gap, since pressing it means the dialog is already open.
Everything here is under `click_trigger` in `laborer-v5.config.json`, seeded
from your v4 config and state on first run so nothing has to be re-tuned or
re-scanned.

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

**One click, then get out of the way.** Because that signal exists, the
hand-over is a *single* shift-click; the cursor then moves off the slot so the
item tooltip stops covering the dialog the accept prompt is drawn in. Only if no
prompt turns up does v4 come back to the slot and send a slower burst of clicks
- and only after re-verifying that the journal is still there.

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
python scripts\tests\test_laborer_v5.py
```

No game, no screen capture, no input: templates are pasted into synthetic
frames, optionally degraded with sub-pixel shift, gain/bias and noise, and the
classifier is checked for correctness rather than for not crashing.

The v5 suite covers only what v5 adds - the click trigger's state machine,
driven over scripted button timelines, and the config layering.

## Older versions

`scripts/laborer-v3.py` and `scripts/laborer-v2.py` still run. v3 remains a
working fallback.
