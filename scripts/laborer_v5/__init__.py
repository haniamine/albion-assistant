"""Albion laborer assistant, v5.

v5 is v4's engine with a different way of starting a cycle. Everything that
looks at the screen or touches the mouse - ``vision``, ``assets``, ``identify``,
``inventory``, ``state``, ``engine``, ``winput`` - is imported from
``laborer_v4`` unchanged; only the launcher layer lives here.

*   ``trigger`` - the left mouse button as an action starter: an armed edge
                  detector that ignores the clicks the engine itself injects.
*   ``config``  - v4's config tree plus a ``click_trigger`` section and the
                  toggle key, in its own file seeded from v4's.
*   ``app``     - the hotkey loop, which now has two ways in.
"""

__all__ = ["app"]
