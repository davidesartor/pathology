"""Viewer: highlights are named drawings; a capture saves the rectangle as on screen, highlights included, with a scale bar."""
import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import Page

from conftest import open_files, set_view
from test_slides import screenshot


def highlight(page: Page, corners: list[tuple[int, int]]) -> dict:
    """Click the corners of a highlight on screen and close it; returns the highlight."""
    page.click("#highlight")
    for x, y in corners:
        page.mouse.click(x, y)
        page.wait_for_timeout(400)  # apart, or two clicks make a double-click, which closes the highlight
    page.keyboard.press("Enter")
    page.click("#highlight")
    return page.evaluate("slide.areas.at(-1)")


def test_highlight_is_named_and_kept(viewer: Page, tiffs):
    open_files(viewer, tiffs["ome_zlib"][0])
    viewer.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")
    corners = [(400, 200), (600, 200), (600, 350), (400, 350)]
    area = highlight(viewer, corners)
    ring = np.array(area["ring"])
    assert ring.shape == (4, 2)
    assert np.allclose(np.diff(ring, axis=0), np.diff(corners, axis=0), atol=1)  # 1 slide px per screen px
    assert (area["number"], area["name"]) == (1, "Highlight 1")
    assert viewer.input_value("#areas .area input[type=text]") == "Highlight 1"
    viewer.fill("#areas .area input[type=text]", "tumour edge")
    assert viewer.evaluate("slide.areas[0].name") == "tumour edge"


@pytest.mark.parametrize("rotation, flip", [(0, False), (90, False), (-30, False), (0, True), (135, True)])
def test_capture_saves_what_is_on_screen(viewer: Page, tiffs, rotation, flip):
    """At 1 screen px per slide px, a capture is the screen's rectangle up to resampling, highlight included, scale bar bottom left."""
    open_files(viewer, tiffs["ome_zlib"][0])
    viewer.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")
    set_view(viewer, rotation, flip)
    color = highlight(viewer, [(500, 230), (700, 230), (700, 320), (500, 320)])["color"]
    (left, top), (right, bottom) = (400, 150), (880, 560)
    want = screenshot(viewer, view_only=True)[top:bottom, left:right]

    viewer.click("#capture")
    viewer.mouse.move(left, top)
    viewer.mouse.down()
    viewer.mouse.move(right, bottom, steps=10)
    with viewer.expect_download() as download:
        viewer.mouse.up()
    got = np.asarray(Image.open(download.value.path()).convert("RGB")).astype(int)
    assert abs(got.shape[0] - want.shape[0]) <= 1 and abs(got.shape[1] - want.shape[1]) <= 1, got.shape
    above_scale_bar = slice(0, want.shape[0] - 120), slice(0, min(got.shape[1], want.shape[1]))
    assert np.median(np.abs(got[above_scale_bar] - want[above_scale_bar])) <= 3
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)])
    assert (np.abs(got - rgb).max(-1) <= 10).sum() > 100  # the highlight's outline
    assert got[-40:, :40].mean() < 0.8 * want[-40:, :40].mean()  # the scale bar box darkens what is under it (the slide has a pixel size)
