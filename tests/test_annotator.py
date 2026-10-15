"""Annotator: nuclei layers are counted, inside the ROIs once some are shown, ROIs save as loaded, overlays turn with the view, and captures show them."""
import json

import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import Page

from conftest import drag, open_files, pointer_at, set_view, wait_for
from test_slides import screenshot

# ROIs in slide px: a square with a hole, and a second square; edges on multiples of 100
OUTER, HOLE, SECOND = (500, 500, 3500, 3500), (1500, 1500, 2000, 2000), (4000, 500, 5000, 1500)


def ring(x0: int, y0: int, x1: int, y1: int) -> list[list[int]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def inside(box: tuple[int, int, int, int], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = box
    return (x0 <= x) & (x <= x1) & (y0 <= y) & (y <= y1)


def feature_collection(features: list[dict]) -> str:
    return json.dumps({"type": "FeatureCollection", "features": features})


@pytest.fixture
def nuclei(tmp_path) -> tuple[str, dict[str, np.ndarray]]:
    """GeoJSON of small square nuclei of two classes, none within a few px of an ROI edge; returns the file and their centres."""
    rng = np.random.default_rng(0)
    centres = rng.integers([10, 10], [5990, 3990], (3000, 2))
    centres = centres[np.all(np.abs((centres + 50) % 100 - 50) > 5, axis=1)]
    names = rng.choice(["Lymphocyte", "Tumor"], len(centres))
    features = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring(x - 3, y - 3, x + 3, y + 3)]},
                 "properties": {"objectType": "detection", "classification": {"name": str(name), "color": [0, 0, 255] if name == "Tumor" else [255, 0, 0]}}}
                for (x, y), name in zip(centres.tolist(), names)]
    path = tmp_path / "nuclei.geojson"
    path.write_text(feature_collection(features))
    return str(path), {name: centres[names == name] for name in ["Lymphocyte", "Tumor"]}


@pytest.fixture
def rois(tmp_path) -> str:
    features = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring(*OUTER), ring(*HOLE)]}, "properties": {"objectType": "annotation"}},
                {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring(*SECOND)]}, "properties": {"objectType": "annotation"}}]
    path = tmp_path / "rois.geojson"
    path.write_text(feature_collection(features))
    return str(path)


def class_counts(page: Page) -> dict[str, int]:
    labels = page.evaluate("[...$('layers').querySelectorAll('label')].map(label => [label.children[2].textContent, label.children[3].textContent])")
    return {name: int(count.replace(",", "")) for name, count in labels}


def stats_note(page: Page) -> str:
    return page.evaluate("$('layers').querySelector('.stats')?.textContent ?? ''")


def test_counts_inside_rois(annotator: Page, tiffs, nuclei, rois):
    path, centres = nuclei
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.set_input_files("#files", path)
    wait_for(annotator, "$('layers').querySelectorAll('label').length >= 2")
    assert {name: class_counts(annotator)[name] for name in centres} == {name: len(c) for name, c in centres.items()}
    assert "counted over the whole layer" in stats_note(annotator)

    annotator.set_input_files("#roi-file", rois)
    wait_for(annotator, "$('layers').querySelector('.stats')?.textContent.includes('inside the ROIs')")
    counted = {name: int(((inside(OUTER, *c.T) & ~inside(HOLE, *c.T)) | inside(SECOND, *c.T)).sum()) for name, c in centres.items()}
    assert {name: class_counts(annotator)[name] for name in centres} == counted

    lymphocytes, tumor = counted["Lymphocyte"], counted["Tumor"]
    area_mm2 = (3000 ** 2 - 500 ** 2 + 1000 ** 2) * (0.25 / 1000) ** 2
    assert stats_note(annotator).split("\n") == [
        f"Lymphocyte / (Lymphocyte + Tumor): {100 * lymphocytes / (lymphocytes + tumor):.1f}%",
        f"Lymphocyte / all nuclei: {100 * lymphocytes / (lymphocytes + tumor):.1f}%",
        f"Lymphocyte: {round(lymphocytes / area_mm2):,} per mm² ({area_mm2:.1f} mm² of ROI)",
        "counted inside the ROIs and annotations shown",
    ]


def test_rois_save_as_loaded(annotator: Page, tiffs, rois):
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.set_input_files("#roi-file", rois)
    wait_for(annotator, "!$('save-roi').disabled")
    with annotator.expect_download() as download:
        annotator.click("#save-roi")
    saved = json.loads(open(download.value.path()).read())
    assert [feature["geometry"]["coordinates"] for feature in saved["features"]] == [[ring(*OUTER), ring(*HOLE)], [ring(*SECOND)]]
    assert all(feature["properties"]["objectType"] == "annotation" for feature in saved["features"])


@pytest.mark.parametrize("rotation, flip", [(0, False), (90, False), (-30, False), (0, True), (135, True)])
def test_overlay_follows_view(annotator: Page, tiffs, tmp_path, rotation, flip):
    """The pointer over the middle of a drawn ROI reads the ROI's middle, however the view is turned."""
    path = tmp_path / "roi.geojson"
    path.write_text(feature_collection([{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring(2400, 1500, 2700, 1700)]}, "properties": {}}]))
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.set_input_files("#roi-file", str(path))
    wait_for(annotator, "!$('save-roi').disabled")
    annotator.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(0.5), null, true)")
    set_view(annotator, rotation, flip)

    shown = screenshot(annotator, view_only=True)
    annotator.evaluate("for (const box of $('layers').querySelectorAll('input[type=checkbox]')) box.click()")
    annotator.evaluate("new Promise(requestAnimationFrame)")
    hidden = screenshot(annotator, view_only=True)
    ys, xs = np.nonzero(np.abs(shown - hidden).max(-1) > 30)
    assert len(xs) > 100
    slide_x, slide_y = pointer_at(annotator, round(xs.mean()), round(ys.mean()))
    assert abs(slide_x - 2550) <= 4 and abs(slide_y - 1600) <= 4, (slide_x, slide_y)


def test_calibration_sets_magnification(annotator: Page, tiffs):
    """Calibrated to 0.25 mm per screen px and shared with the viewer: 1 screen px per 0.25 µm slide px is 100× through a 10× eyepiece."""
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")
    annotator.click("#eyepiece-power [data-value='10']")
    annotator.click("#calibrate")
    annotator.locator("#calibration-scale").fill("4")
    annotator.click("#calibration button[value=save]")
    wait_for(annotator, "$('status').textContent.includes('100.0×')")
    assert annotator.evaluate("localStorage.getItem('screen-mm-per-device-px')") == "0.25"
    assert json.loads(annotator.evaluate("localStorage.getItem('eyepiece')"))["power"] == 10


@pytest.mark.parametrize("rotation, flip", [(0, False), (90, False), (-30, False), (0, True), (135, True)])
def test_capture_saves_what_is_on_screen(annotator: Page, tiffs, nuclei, tmp_path, rotation, flip):
    """At 1 screen px per slide px, a capture is the frame's rectangle on screen up to resampling, the ROI and the nuclei inside it included."""
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")
    set_view(annotator, rotation, flip)
    (left, top), (right, bottom) = (400, 150), (880, 560)
    # an ROI about the middle of the captured rectangle, so it stays inside it however the view is turned
    x, y = annotator.evaluate(f"(() => {{ const point = toImage(viewer, new OpenSeadragon.Point({(left + right) / 2}, {(top + bottom) / 2})); return [Math.round(point.x), Math.round(point.y)]; }})()")
    path = tmp_path / "roi.geojson"
    path.write_text(feature_collection([{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring(x - 140, y - 140, x + 140, y + 140)]}, "properties": {}}]))
    annotator.set_input_files("#roi-file", str(path))
    annotator.set_input_files("#files", nuclei[0])
    wait_for(annotator, "$('layers').querySelectorAll('label').length >= 3")
    annotator.evaluate("new Promise(requestAnimationFrame)")
    want = screenshot(annotator, view_only=True)[top:bottom, left:right]

    annotator.click("#capture")
    drag(annotator, (200, 160), (left, top))
    drag(annotator, (800, 640), (right, bottom))
    with annotator.expect_download() as download:
        annotator.click("#capture-save")
    got = np.asarray(Image.open(download.value.path()).convert("RGB")).astype(int)
    assert abs(got.shape[0] - want.shape[0]) <= 1 and abs(got.shape[1] - want.shape[1]) <= 1, got.shape
    above_ruler = slice(0, want.shape[0] - 60), slice(0, min(got.shape[1], want.shape[1]))
    assert np.median(np.abs(got[above_ruler] - want[above_ruler])) <= 3
    assert (np.abs(got - [0, 255, 255]).max(-1) <= 10).sum() > 300  # the ROI's 3 px outline
    assert (got[-15:-12, 13:58] < 40).all()  # the black ruler, 12 px in from the bottom left

    # the nuclei show, but only inside the ROI: hiding them changes the capture there and nowhere else
    annotator.evaluate("for (const label of $('layers').querySelectorAll('label')) if (label.textContent.match(/Lymphocyte|Tumor/)) label.firstChild.click()")
    with annotator.expect_download() as download:
        annotator.click("#capture-save")
    without = np.asarray(Image.open(download.value.path()).convert("RGB")).astype(int)
    ys, xs = np.nonzero(np.abs(got - without).max(-1) > 30)
    assert len(xs) > 100
    centre = (got.shape[1] / 2, got.shape[0] / 2)
    assert np.hypot(xs - centre[0], ys - centre[1]).max() < 140 * 2 ** 0.5 + 3
