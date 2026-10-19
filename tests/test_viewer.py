"""Viewer: highlights are named drawings; a capture saves its frame as on screen, drawings included, with a ruler."""
import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import Page

from conftest import drag, open_files, open_settings, set_view, wait_for
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
    ring = np.array(area["rings"][0])
    assert ring.shape == (4, 2)
    assert np.allclose(np.diff(ring, axis=0), np.diff(corners, axis=0), atol=1)  # 1 slide px per screen px
    assert (area["number"], area["name"]) == (1, "Highlight 1")
    assert viewer.input_value("#areas .area input[type=text]") == "Highlight 1"
    viewer.fill("#areas .area input[type=text]", "tumour edge")
    assert viewer.evaluate("slide.areas[0].name") == "tumour edge"


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
    assert (np.abs(got - rgb).max(-1) <= 10).sum() > 30  # the highlight's 1.5 px outline
    assert (got[-15:-12, 13:58] < 40).all()  # the black ruler, 12 px in from the bottom left (the slide has a pixel size)


def test_calibration_sets_magnification(viewer: Page, tiffs):
    """Calibrated to 0.25 mm per screen px, 40× through a 10× eyepiece is 0.625 µm per screen px; Default forgets it."""
    open_files(viewer, tiffs["ome_zlib"][0])
    open_settings(viewer)
    viewer.click("#calibrate")
    viewer.click("#calibration-reference [data-value='usbc']")
    assert abs(viewer.evaluate("$('calibration-shape').getBoundingClientRect().width") - 8.25 / viewer.evaluate("screenMmPerPx()")) < 0.5
    viewer.locator("#calibration-scale").fill("4")
    assert viewer.evaluate("$('calibration-shape').getBoundingClientRect().width") == 33
    viewer.click("#calibration button[value=save]")
    assert wait_for(viewer, "localStorage.getItem('screen-mm-per-device-px')") == "0.25"  # the dialog closes a task later
    open_settings(viewer)
    viewer.click("#eyepiece-power [data-value='10']")
    viewer.click("#zoom [data-zoom='40']")
    wait_for(viewer, "$('status').textContent.includes('40.0×')")
    assert abs(viewer.evaluate("viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom())") - 0.25 / 0.625) < 1e-3
    open_settings(viewer)
    viewer.click("#calibrate")
    viewer.click("#calibration button[value=default]")
    wait_for(viewer, "localStorage.getItem('screen-mm-per-device-px') === null")


def test_paint_erase_and_merge_highlights(viewer: Page, tiffs):
    """A brush stroke away from the highlights starts one, erasing across one splits it, merging joins two under the kept one's name, and undo steps back."""
    open_files(viewer, tiffs["ome_zlib"][0])
    viewer.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(0.25), viewer.viewport.imageToViewportCoordinates(3300, 2000), true)")
    viewer.click("#highlight")
    viewer.click("#highlight-mode [data-value=paint]")
    viewer.evaluate("changeBrush({size: 20})")  # 40 slide px across a radius at 0.25 screen px per slide px

    def screen(x: float, y: float) -> tuple[float, float]:
        return tuple(viewer.evaluate("([x, y]) => { const point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y)), rect = viewer.container.getBoundingClientRect(); return [rect.left + point.x, rect.top + point.y]; }", [x, y]))

    def stroke(*points: tuple[float, float]):
        viewer.mouse.move(*screen(*points[0]))
        viewer.mouse.down()
        for point in points[1:]:
            viewer.mouse.move(*screen(*point), steps=10)
        viewer.mouse.up()

    def areas() -> list[list[float]]:
        return viewer.evaluate("slide.areas.map(area => area.rings.map(ringArea).sort((a, b) => a - b))")

    stroke((2000, 2000), (3000, 2000))
    stroke((2900, 2000), (2900, 2600))  # touches the first, so adds to it
    stroke((4300, 1800), (4700, 1800), (4700, 2200), (4300, 2200), (4300, 1800))
    capsule = 1000 * 80 + np.pi * 40 ** 2
    first, loop = areas()
    assert first == [pytest.approx(capsule + 560 * 80, rel=0.03)]
    assert loop == [pytest.approx(320 ** 2, rel=0.03), pytest.approx(480 ** 2, rel=0.03)]  # the hole inside it, and its outline

    viewer.click("#highlight-mode [data-value=erase]")
    stroke((2500, 1800), (2500, 2200))
    assert len(areas()[0]) == 2 and sum(areas()[0]) == pytest.approx(capsule + 560 * 80 - 80 * 80, rel=0.03)
    viewer.keyboard.press("Backspace")
    assert len(areas()[0]) == 1

    viewer.locator("#areas .area button[title^='line width']").first.click()
    viewer.locator("#areas .area-edit input[type=range]").fill("4")
    assert viewer.evaluate("slide.areas[0].width") == 4
    viewer.locator("#areas .area button[title^='line width']").nth(1).click()
    viewer.locator("#areas .area-edit select").select_option(label="Highlight 1")
    assert viewer.evaluate("slide.areas.map(area => area.name)") == ["Highlight 1"]
    assert len(areas()[0]) == 3  # apart, the two keep their rings
    viewer.click("#highlight-undo")  # touch screens have no Backspace
    assert viewer.evaluate("slide.areas.map(area => [area.name, area.width])") == [["Highlight 1", 4], ["Highlight 2", 1.5]]
