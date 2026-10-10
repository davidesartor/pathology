"""Viewer: highlights and measurements are named drawings; a capture saves its frame as on screen, drawings included, with a ruler."""
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


def drag(page: Page, start: tuple[int, int], end: tuple[int, int]):
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(*end, steps=10)
    page.mouse.up()


@pytest.mark.parametrize("rotation, flip", [(0, False), (90, False), (-30, False), (0, True), (135, True)])
def test_capture_saves_what_is_on_screen(viewer: Page, tiffs, rotation, flip):
    """At 1 screen px per slide px, a capture is the frame's rectangle on screen up to resampling, highlight included, ruler bottom left."""
    open_files(viewer, tiffs["ome_zlib"][0])
    viewer.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")
    set_view(viewer, rotation, flip)
    color = highlight(viewer, [(500, 230), (700, 230), (700, 320), (500, 320)])["color"]
    (left, top), (right, bottom) = (400, 150), (880, 560)
    want = screenshot(viewer, view_only=True)[top:bottom, left:right]

    # the frame opens free-shaped at 60% of the 1000 × 800 view; its corners are dragged onto the rectangle
    viewer.click("#capture")
    drag(viewer, (200, 160), (left, top))
    drag(viewer, (800, 640), (right, bottom))
    with viewer.expect_download() as download:
        viewer.click("#capture-save")
    got = np.asarray(Image.open(download.value.path()).convert("RGB")).astype(int)
    assert abs(got.shape[0] - want.shape[0]) <= 1 and abs(got.shape[1] - want.shape[1]) <= 1, got.shape
    above_ruler = slice(0, want.shape[0] - 60), slice(0, min(got.shape[1], want.shape[1]))
    assert np.median(np.abs(got[above_ruler] - want[above_ruler])) <= 3
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)])
    assert (np.abs(got - rgb).max(-1) <= 10).sum() > 100  # the highlight's outline
    assert (got[-15:-12, 13:58] < 40).all()  # the black ruler, 12 px in from the bottom left (the slide has a pixel size)


def test_capture_frame_keeps_its_shape_and_settings(viewer: Page, tiffs):
    """A fixed shape holds while resizing; shape and corner come back on the next visit."""
    open_files(viewer, tiffs["ome_zlib"][0])
    viewer.click("#capture")
    viewer.click("#capture-shape [data-value='4:3']")
    drag(viewer, (800, 640), (900, 700))
    viewer.click("#capture-corner [data-value='top-right']")
    assert viewer.evaluate("JSON.parse(localStorage.getItem('capture-settings')).size") is None  # kept only once a capture is saved
    viewer.reload()
    open_files(viewer, tiffs["ome_zlib"][0])
    viewer.click("#capture")
    assert viewer.locator("#capture-shape .active").get_attribute("data-value") == "4:3"
    assert viewer.locator("#capture-corner .active").get_attribute("data-value") == "top-right"
    size = viewer.locator("#capture-size").text_content()
    width, height = (int(n.replace(",", "")) for n in size.split(" px")[0].split(" × "))
    assert abs(width / height - 4 / 3) < 0.01, size


def test_capture_preset_width(viewer: Page, tiffs):
    """A preset or typed width spans that much tissue, zooming out to fit; resizing by hand lets it go."""
    open_files(viewer, tiffs["ome_zlib"][0])
    viewer.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")
    viewer.click("#capture")
    viewer.click("#capture-shape [data-value='4:3']")
    viewer.click("#capture-width [data-value='500']")
    assert viewer.locator("#capture-size").text_content().startswith("2,000 × 1,500 px · 500 µm × 375 µm")
    assert viewer.evaluate("viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true))") <= 0.9 * 1000 / 2000 + 1e-3
    assert viewer.locator("#capture-width .active").get_attribute("data-value") == "500"
    rect = viewer.evaluate("(() => { const s = viewer.viewport.getContainerSize(); return [s.x, s.y]; })()")
    corner = (rect[0] / 2 + 0.45 * rect[0], rect[1] / 2 + 0.45 * rect[0] * 3 / 4)
    drag(viewer, corner, (corner[0] - 100, corner[1] - 75))
    assert viewer.locator("#capture-width .active").count() == 0
    viewer.fill("#capture-width-custom", "300")
    viewer.press("#capture-width-custom", "Enter")
    assert viewer.locator("#capture-size").text_content().startswith("1,200 × 900 px · 300 µm × 225 µm")
    assert viewer.locator("#capture-width .active").get_attribute("data-part") == "capture-width-custom"


def test_zoom_presets(viewer: Page, tiffs):
    """10× is 1 µm per screen px; the slide is 0.25 µm per px."""
    open_files(viewer, tiffs["ome_zlib"][0])
    viewer.click("#zoom [data-zoom='10']")
    viewer.wait_for_timeout(1500)  # the zoom animates
    assert abs(viewer.evaluate("viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true))") - 0.25) < 0.005
    assert viewer.locator("#zoom .active").get_attribute("data-zoom") == "10"


def test_measure_line(viewer: Page, tiffs):
    """A dragged line is kept with its length in µm; shift keeps it level."""
    open_files(viewer, tiffs["ome_zlib"][0])
    viewer.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")
    viewer.click("#measure")
    drag(viewer, (300, 300), (500, 300))
    viewer.keyboard.down("Shift")
    drag(viewer, (300, 400), (500, 420))
    viewer.keyboard.up("Shift")
    first, second = viewer.evaluate("slide.areas")
    assert first["kind"] == "line" and abs(np.hypot(*np.diff(first["ring"], axis=0)[0]) - 200) <= 1
    assert second["ring"][0][1] == second["ring"][1][1]  # level
    assert viewer.locator("#areas .area .count").first.text_content() == "50 µm"
