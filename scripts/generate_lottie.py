"""Generate the Lottie animations used by the Streamlit app.

The four animations are built programmatically (pure vector shapes, Lottie /
Bodymovin JSON schema) so the project ships with original, licence-free
assets in the blue/violet brand palette::

    python scripts/generate_lottie.py

Any animation from https://lottiefiles.com can replace them: just drop the
JSON in ``assets/lottie/`` under the same file name.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "lottie"

Json = dict[str, Any]
Color = list[float]

BLUE: Color = [0.145, 0.388, 0.922, 1]
VIOLET: Color = [0.486, 0.227, 0.929, 1]
INDIGO: Color = [0.310, 0.275, 0.898, 1]
LAVENDER: Color = [0.769, 0.710, 0.992, 1]
SKY: Color = [0.576, 0.773, 0.992, 1]
PALE: Color = [0.933, 0.949, 1.0, 1]
WHITE: Color = [1, 1, 1, 1]
GREEN: Color = [0.133, 0.773, 0.369, 1]
GREY: Color = [0.580, 0.639, 0.722, 1]


# --------------------------------------------------------------------------- #
# Property helpers
# --------------------------------------------------------------------------- #
def static(value: Any) -> Json:
    """A non-animated property."""
    return {"a": 0, "k": value}


def animated(keyframes: list[tuple[float, Any]], linear: bool = False) -> Json:
    """An animated property from ``[(frame, value), ...]`` keyframes."""
    ease_out = {"x": [0.167], "y": [0.167]} if linear else {"x": [0.42], "y": [0]}
    ease_in = {"x": [0.833], "y": [0.833]} if linear else {"x": [0.58], "y": [1]}
    frames: list[Json] = []
    for index, (time, value) in enumerate(keyframes):
        frame: Json = {"t": time, "s": value if isinstance(value, list) else [value]}
        if index < len(keyframes) - 1:
            frame["o"], frame["i"] = ease_out, ease_in
        frames.append(frame)
    return {"a": 1, "k": frames}


def transform(
    position: Json | None = None,
    scale: Json | None = None,
    rotation: Json | None = None,
    opacity: Json | None = None,
    anchor: Json | None = None,
) -> Json:
    """Layer transform block."""
    return {
        "o": opacity or static(100),
        "r": rotation or static(0),
        "p": position or static([0, 0, 0]),
        "a": anchor or static([0, 0, 0]),
        "s": scale or static([100, 100, 100]),
    }


# --------------------------------------------------------------------------- #
# Shape helpers
# --------------------------------------------------------------------------- #
def fill(color: Color, opacity: float | Json = 100) -> Json:
    return {"ty": "fl", "c": static(color), "o": opacity if isinstance(opacity, dict) else static(opacity), "r": 1}


def stroke(color: Color, width: float, opacity: float = 100) -> Json:
    return {"ty": "st", "c": static(color), "o": static(opacity), "w": static(width), "lc": 2, "lj": 2}


def ellipse(x: float, y: float, width: float, height: float | None = None) -> Json:
    return {"ty": "el", "p": static([x, y]), "s": static([width, height or width])}


def rect(x: float, y: float, width: float, height: float, radius: float = 0) -> Json:
    return {"ty": "rc", "p": static([x, y]), "s": static([width, height]), "r": static(radius)}


def path(points: list[list[float]], closed: bool = False) -> Json:
    zeros = [[0, 0] for _ in points]
    return {"ty": "sh", "ks": static({"i": zeros, "o": zeros, "v": points, "c": closed})}


def trim(end: Json) -> Json:
    return {"ty": "tm", "s": static(0), "e": end, "o": static(0), "m": 1}


def group(*items: Json, position: Json | None = None, opacity: Json | None = None) -> Json:
    group_transform = {
        "ty": "tr",
        "p": position or static([0, 0]),
        "a": static([0, 0]),
        "s": static([100, 100]),
        "r": static(0),
        "o": opacity or static(100),
    }
    return {"ty": "gr", "it": [*items, group_transform]}


class Animation:
    """Minimal builder for a Lottie composition."""

    def __init__(self, name: str, size: int, frames: int, fps: int = 30) -> None:
        self.doc: Json = {
            "v": "5.7.4", "fr": fps, "ip": 0, "op": frames, "w": size, "h": size,
            "nm": name, "ddd": 0, "assets": [], "layers": [],
        }
        self.frames = frames

    def _add(self, layer: Json) -> int:
        layer.update({"ddd": 0, "ind": len(self.doc["layers"]) + 1, "sr": 1, "ao": 0,
                      "ip": 0, "op": self.frames, "st": 0, "bm": 0})
        self.doc["layers"].append(layer)
        return layer["ind"]

    def shape(self, name: str, shapes: list[Json], ks: Json, parent: int | None = None) -> int:
        """Add a shape layer (first added = drawn on top)."""
        layer: Json = {"ty": 4, "nm": name, "ks": ks, "shapes": shapes}
        if parent is not None:
            layer["parent"] = parent
        return self._add(layer)

    def null(self, name: str, ks: Json) -> int:
        """Add an invisible controller layer other layers can be parented to."""
        return self._add({"ty": 3, "nm": name, "ks": ks})

    def save(self, filename: str) -> Path:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        target = OUTPUT_DIR / filename
        target.write_text(json.dumps(self.doc, separators=(",", ":")), encoding="utf-8")
        return target


# --------------------------------------------------------------------------- #
# Animations
# --------------------------------------------------------------------------- #
def build_ai() -> Animation:
    """Home page: an AI core with orbiting data nodes."""
    anim = Animation("ai", size=400, frames=120)
    centre = static([200, 200, 0])

    # Core
    anim.shape("core-dot", [group(ellipse(0, 0, 26), fill(WHITE))],
               transform(centre, scale=animated([(0, [100, 100, 100]), (60, [70, 70, 100]), (120, [100, 100, 100])])))
    anim.shape("core", [group(ellipse(0, 0, 92), fill(VIOLET))],
               transform(centre, scale=animated([(0, [100, 100, 100]), (60, [110, 110, 100]), (120, [100, 100, 100])])))

    # Orbiting nodes (each layer spins around the centre)
    orbits = [(110, 26, BLUE, 0, 360), (110, 16, SKY, 180, 540), (160, 22, VIOLET, 90, -270),
              (160, 14, LAVENDER, 250, -110), (160, 18, INDIGO, 330, -30)]
    for index, (radius, size, color, start, end) in enumerate(orbits):
        anim.shape(f"node-{index}", [group(ellipse(radius, 0, size), fill(color))],
                   transform(centre, rotation=animated([(0, start), (120, end)], linear=True)))

    # Orbit rings
    anim.shape("ring-1", [group(ellipse(0, 0, 220), stroke(BLUE, 2, 35))], transform(centre))
    anim.shape("ring-2", [group(ellipse(0, 0, 320), stroke(VIOLET, 2, 25))], transform(centre))

    # Expanding ripples
    for index, offset in enumerate((0, 60)):
        scale = [(offset, [100, 100, 100]), (offset + 60, [230, 230, 100])]
        opacity = [(offset, 35), (offset + 60, 0)]
        if offset:  # second ripple wraps around the loop
            scale = [(0, [100, 100, 100]), (0.01, [100, 100, 100]), *scale]
            opacity = [(0, 0), (offset - 1, 0), *opacity]
        anim.shape(f"ripple-{index}", [group(ellipse(0, 0, 92), fill(VIOLET))],
                   transform(centre, scale=animated(scale), opacity=animated(opacity)))
    return anim


def build_robot() -> Animation:
    """Prediction page: a friendly robot that bobs, blinks and waves."""
    anim = Animation("robot", size=400, frames=90)
    body = anim.null("bob", transform(animated([(0, [200, 205, 0]), (45, [200, 191, 0]), (90, [200, 205, 0])])))

    blink = animated([(0, [100, 100, 100]), (50, [100, 100, 100]), (55, [100, 8, 100]),
                      (60, [100, 100, 100]), (90, [100, 100, 100])])
    for side, x in (("left", -34), ("right", 34)):
        anim.shape(f"eye-{side}", [group(ellipse(0, 0, 28), fill(BLUE))],
                   transform(static([x, -52, 0]), scale=blink), parent=body)
    anim.shape("mouth", [group(rect(0, -18, 40, 9, 5), fill(VIOLET))], transform(), parent=body)
    anim.shape("face", [group(rect(0, -45, 142, 86, 26), fill(PALE))], transform(), parent=body)
    anim.shape("antenna-ball", [group(ellipse(0, 0, 24), fill(BLUE))],
               transform(static([0, -146, 0]),
                         scale=animated([(0, [100, 100, 100]), (22, [135, 135, 100]), (45, [100, 100, 100]),
                                         (67, [135, 135, 100]), (90, [100, 100, 100])])), parent=body)
    anim.shape("antenna", [group(rect(0, -124, 7, 30, 3), fill(INDIGO))], transform(), parent=body)
    anim.shape("head", [group(rect(0, -45, 184, 132, 36), fill(VIOLET))], transform(), parent=body)
    anim.shape("ears", [group(rect(-100, -45, 18, 46, 8), fill(BLUE)),
                        group(rect(100, -45, 18, 46, 8), fill(BLUE))], transform(), parent=body)
    anim.shape("chest-light", [group(ellipse(0, 72, 22), fill(WHITE))],
               transform(opacity=animated([(0, 100), (45, 35), (90, 100)])), parent=body)
    anim.shape("body", [group(rect(0, 78, 132, 84, 28), fill(BLUE))], transform(), parent=body)
    anim.shape("arm-left", [group(rect(-84, 74, 24, 62, 12), fill(VIOLET))], transform(), parent=body)
    # Right arm pivots around the shoulder to wave
    anim.shape("arm-right", [group(rect(0, 28, 24, 62, 12), fill(VIOLET))],
               transform(static([84, 46, 0]),
                         rotation=animated([(0, 0), (15, -150), (27, -120), (39, -150), (51, -120), (70, 0), (90, 0)])),
               parent=body)
    anim.shape("neck", [group(rect(0, 28, 44, 20, 6), fill(INDIGO))], transform(), parent=body)

    # Ground shadow shrinks when the robot rises
    anim.shape("shadow", [group(ellipse(0, 0, 170, 20), fill(GREY, 28))],
               transform(static([200, 352, 0]),
                         scale=animated([(0, [100, 100, 100]), (45, [84, 84, 100]), (90, [100, 100, 100])])))
    return anim


def build_analytics() -> Animation:
    """Dashboard page: bars growing under a trend line."""
    anim = Animation("analytics", size=400, frames=150)
    heights = [90, 150, 120, 200, 250]
    xs = [80, 140, 200, 260, 320]
    base_y, fade_start, fade_end = 318, 130, 148

    # Trend line + markers (drawn on top of the bars)
    points = [[x, base_y - h - 34] for x, h in zip(xs, heights, strict=False)]
    line_opacity = animated([(0, 100), (fade_start, 100), (fade_end, 0)])
    for index, (x, y) in enumerate(points):
        appear = 48 + index * 8
        anim.shape(f"dot-{index}",
                   [group(ellipse(0, 0, 16), fill(WHITE), stroke(INDIGO, 4))],
                   transform(static([x, y, 0]),
                             scale=animated([(0, [0, 0, 100]), (appear, [0, 0, 100]), (appear + 8, [100, 100, 100]),
                                             (fade_start, [100, 100, 100]), (fade_end, [0, 0, 100])])))
    anim.shape("trend", [group(path(points), trim(animated([(0, 0), (45, 0), (88, 100)])), stroke(INDIGO, 5))],
               transform(opacity=line_opacity))

    for index, (x, height) in enumerate(zip(xs, heights, strict=False)):
        start = index * 8
        grow = animated([(0, [100, 0, 100]), (start + 0.01, [100, 0, 100]), (start + 26, [100, 100, 100]),
                         (fade_start, [100, 100, 100]), (fade_end, [100, 0, 100])])
        anim.shape(f"bar-{index}",
                   [group(rect(0, -height / 2, 40, height, 8), fill(BLUE if index % 2 == 0 else VIOLET))],
                   transform(static([x, base_y, 0]), scale=grow))

    anim.shape("axis", [group(rect(200, base_y + 3, 310, 5, 2.5), fill(LAVENDER))], transform())
    return anim


def build_success() -> Animation:
    """Shown after a prediction: a check mark with a small burst."""
    anim = Animation("success", size=300, frames=90)
    centre = static([150, 150, 0])

    anim.shape("check", [group(path([[-38, 2], [-12, 28], [40, -26]]),
                               trim(animated([(0, 0), (18, 0), (38, 100)])), stroke(WHITE, 14))],
               transform(centre))
    anim.shape("disc", [group(ellipse(0, 0, 150), fill(GREEN))],
               transform(centre, scale=animated([(0, [0, 0, 100]), (16, [114, 114, 100]), (24, [100, 100, 100])])))
    anim.shape("burst-ring", [group(ellipse(0, 0, 150), stroke(GREEN, 5))],
               transform(centre,
                         scale=animated([(0, [100, 100, 100]), (14, [100, 100, 100]), (50, [165, 165, 100])]),
                         opacity=animated([(0, 0), (14, 60), (50, 0)])))
    for index in range(8):
        angle = index * 45
        colour = [BLUE, VIOLET][index % 2]
        radians = math.radians(angle)
        start = [150 + 82 * math.sin(radians), 150 - 82 * math.cos(radians), 0]
        end = [150 + 128 * math.sin(radians), 150 - 128 * math.cos(radians), 0]
        anim.shape(f"spark-{index}", [group(ellipse(0, 0, 11), fill(colour))],
                   transform(animated([(0, start), (16, start), (48, end)]),
                             opacity=animated([(0, 0), (15, 0), (18, 100), (48, 0)])))
    return anim


def main() -> None:
    builders = {
        "ai.json": build_ai,
        "robot.json": build_robot,
        "analytics.json": build_analytics,
        "success.json": build_success,
    }
    for filename, builder in builders.items():
        target = builder().save(filename)
        print(f"wrote {target.relative_to(OUTPUT_DIR.parents[1])} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
