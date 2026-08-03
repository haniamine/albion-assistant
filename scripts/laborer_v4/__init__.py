"""Albion laborer assistant, v4.

Design notes for the pieces that matter:

*   ``vision``   - one long-lived mss instance, and templates whose fully
                   opaque alpha is dropped (an all-255 mask is numerically
                   identical to no mask but costs 2.4x).
*   ``state``    - learned dialog anchors + the journal slot cache, invalidated
                   together when the screen geometry changes.
*   ``assets``   - candidate reduction: only laborer tiers whose journal is in
                   the inventory are ever matched against the screen.
*   ``identify`` - detect once, then classify one crop; profession by score,
                   tier by badge colour, both requiring a margin.
*   ``engine``   - the cycle, with anchored lookups that self-heal by falling
                   back to a coarse-to-fine full search.
"""

__all__ = ["app"]
