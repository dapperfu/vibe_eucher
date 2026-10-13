"""Tests for the plain trick-summary table and icon key."""

import contextlib
import re
from io import StringIO

import pytest
from rich.console import Console

from eucher.cards import Card, Rank, Suit
from eucher.players import Player
from eucher.players.profiles import HumanProfile
from eucher.tui.text_tui import TextTUI

_SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")


def _sample_tui() -> TextTUI:
    """Build a hand whose first trick matches a copied summary row."""
    tui = TextTUI()
    profile = HumanProfile()
    players = [
        Player(name, index, profile)
        for index, name in enumerate(["You", "Stephen", "Mark", "Melissa"])
    ]
    tui.set_players(players)
    tui.log_trick(
        1,
        [
            Card(Suit.CLUBS, Rank.KING),
            Card(Suit.SPADES, Rank.JACK),
            Card(Suit.CLUBS, Rank.NINE),
            Card(Suit.CLUBS, Rank.JACK),
        ],
        [0, 1, 2, 3],
        3,
    )
    tui.log_trick(
        2,
        [
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.HEARTS, Rank.JACK),
            Card(Suit.SPADES, Rank.NINE),
            Card(Suit.HEARTS, Rank.NINE),
        ],
        [3, 0, 1, 2],
        3,
    )
    return tui


def _color_state_at_line_end(line: str) -> tuple[int | None, int | None]:
    """
    Return the foreground and background SGR codes still active at end of line.

    None means the terminal default. Extended 38/48 colors use the code itself.
    """
    foreground: int | None = None
    background: int | None = None
    for match in _SGR_RE.finditer(line):
        params = match.group(1)
        codes = [int(part) if part else 0 for part in params.split(";")] if params else [0]
        index = 0
        while index < len(codes):
            code = codes[index]
            if code == 0:
                foreground = None
                background = None
            elif code == 39:
                foreground = None
            elif code == 49:
                background = None
            elif 30 <= code <= 37 or 90 <= code <= 97:
                foreground = code
            elif 40 <= code <= 47 or 100 <= code <= 107:
                background = code
            elif code in (38, 48):
                target = code
                if index + 1 < len(codes) and codes[index + 1] == 5 and index + 2 < len(codes):
                    if target == 38:
                        foreground = codes[index + 2]
                    else:
                        background = codes[index + 2]
                    index += 2
                elif index + 1 < len(codes) and codes[index + 1] == 2 and index + 4 < len(codes):
                    if target == 38:
                        foreground = code
                    else:
                        background = code
                    index += 4
            index += 1
    return foreground, background


def _assert_colors_closed(rendered: str) -> None:
    """Fail when a line or the whole summary ends with a color still active."""
    for line in rendered.splitlines():
        foreground, background = _color_state_at_line_end(line)
        assert foreground is None, line
        assert background is None, line
    foreground, background = _color_state_at_line_end(rendered)
    assert foreground is None
    assert background is None


def _render_rich_summary(tui: TextTUI, monkeypatch: pytest.MonkeyPatch) -> str:
    """Render the Rich summary with color forced on, independent of the tty."""
    buffer = StringIO()

    def capturing_console(*args: object, **kwargs: object) -> Console:
        kwargs["file"] = buffer
        kwargs["force_terminal"] = True
        kwargs["color_system"] = "standard"
        kwargs["width"] = 100
        kwargs["highlight"] = False
        return Console(*args, **kwargs)

    monkeypatch.setattr("eucher.tui.text_tui.Console", capturing_console)
    monkeypatch.delenv("NO_COLOR", raising=False)
    tui.display_trick_summary()
    return buffer.getvalue()


def test_rich_summary_is_plain_text_with_icon_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The trick table has no color. Icons and the Winner column mark the trick."""
    rendered = _render_rich_summary(_sample_tui(), monkeypatch)

    assert "🎯 led the trick" in rendered
    assert "🏆 won the trick" in rendered
    assert "J♣🏆" in rendered
    assert "K♣🎯" in rendered
    assert "Melissa" in rendered
    assert "\x1b[" not in rendered


def test_plain_summary_is_plain_text_with_icon_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The non-Rich summary is plain text with the same icon key."""
    monkeypatch.setattr("eucher.tui.text_tui.RICH_AVAILABLE", False)
    tui = _sample_tui()

    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        tui.display_trick_summary()
    rendered = buffer.getvalue()

    assert "🎯 led the trick" in rendered
    assert "🏆 won the trick" in rendered
    assert "J♣🏆" in rendered
    body = rendered.replace("\x1b[0m", "")
    assert "\x1b[" not in body
    _assert_colors_closed(rendered)
