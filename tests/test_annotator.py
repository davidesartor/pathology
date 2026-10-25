"""Annotator: nuclei layers are counted, inside the ROIs once some are shown, ROIs save as loaded, overlays turn with the view, and captures show them."""
import json

import numpy as np
import pytest
from PIL import Image
from playwright.sync_api import Page

from conftest import drag, open_files, open_settings, pointer_at, set_view, wait_for
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
    labels = page.evaluate("[...$('layers').querySelectorAll('label:has(input[type=checkbox])')].map(label => [label.children[2].textContent, label.children[3].textContent])")
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

    # each ROI counted on its own, the hole left out of the first
    per_roi = {name: [int((inside(OUTER, *c.T) & ~inside(HOLE, *c.T)).sum()), int(inside(SECOND, *c.T).sum())] for name, c in centres.items()}
    rows = annotator.evaluate("[...$('layers').querySelectorAll('.per-roi tr')].map(tr => [...tr.cells].map(cell => cell.textContent))")
    table = {row[0]: row[1:] for row in rows}
    assert table["ROI"] == ["1", "2"]
    assert {name: [int(count.replace(",", "")) for count in table[name]] for name in centres} == per_roi
    assert table["mm²"] == [f"{(3000 ** 2 - 500 ** 2) * (0.25 / 1000) ** 2:.2f}", f"{1000 ** 2 * (0.25 / 1000) ** 2:.2f}"]
    assert table["Lymphocyte / (Lymphocyte + Tumor)"] == [f"{100 * l / (l + t):.1f}%" for l, t in zip(per_roi["Lymphocyte"], per_roi["Tumor"])]


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
    open_settings(annotator)
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


def test_zoomed_out_tiny_nuclei_still_show(annotator: Page, tiffs, tmp_path):
    """Too far out for outlines, a nucleus under a screen px across is still stroked as a small round spot."""
    path = tmp_path / "nucleus.geojson"
    path.write_text(feature_collection([{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring(1998, 1498, 2002, 1502)]},
                                         "properties": {"objectType": "detection", "classification": {"name": "Tumor", "color": [255, 0, 0]}}}]))
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.set_input_files("#files", str(path))
    wait_for(annotator, "$('layers').querySelectorAll('label').length >= 1")
    annotator.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(0.3), null, true)")
    annotator.evaluate("viewer.viewport.panTo(viewer.viewport.imageToViewportCoordinates(2000, 1500), true)")
    annotator.evaluate("draw()")
    alpha = np.array(annotator.evaluate("""(() => {
        const {x, y} = toScreen(viewer, 2000, 1500), dpr = window.devicePixelRatio || 1;
        const data = panes[0].ctx.getImageData(Math.round(x * dpr) - 10, Math.round(y * dpr) - 10, 21, 21).data;
        return Array.from({length: 21 * 21}, (_, i) => data[4 * i + 3]);
    })()""")).reshape(21, 21)
    ys, xs = np.nonzero(alpha > 64)
    assert 1 <= len(xs) <= 16, len(xs)  # a disc of radius 0.6 + 0.75 screen px
    assert abs(xs.mean() - 10) <= 1 and abs(ys.mean() - 10) <= 1


def test_zoomed_out_nuclei_are_circles(annotator: Page, tiffs, tmp_path):
    """Zoomed out, a nucleus is stroked as the circle across its longer side: a 200 × 100 px rectangle, a ring 200 px across, hollow."""
    path = tmp_path / "nucleus.geojson"
    path.write_text(feature_collection([{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring(1900, 1450, 2100, 1550)]},
                                         "properties": {"objectType": "detection", "classification": {"name": "Tumor", "color": [255, 0, 0]}}}]))
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.set_input_files("#files", str(path))
    wait_for(annotator, "$('layers').querySelectorAll('label').length >= 1")
    annotator.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(0.3), null, true)")
    annotator.evaluate("viewer.viewport.panTo(viewer.viewport.imageToViewportCoordinates(2000, 1500), true)")
    annotator.evaluate("draw()")
    alpha = np.array(annotator.evaluate("""(() => {
        const {x, y} = toScreen(viewer, 2000, 1500), dpr = window.devicePixelRatio || 1;
        const data = panes[0].ctx.getImageData(Math.round(x * dpr) - 50, Math.round(y * dpr) - 50, 101, 101).data;
        return Array.from({length: 101 * 101}, (_, i) => data[4 * i + 3]);
    })()""")).reshape(101, 101)
    ys, xs = np.nonzero(alpha > 128)
    diameter = 200 * 0.3  # 60 screen px
    assert abs((xs.max() - xs.min() + 1) - diameter) <= 3 and abs((ys.max() - ys.min() + 1) - diameter) <= 3, (xs.min(), xs.max(), ys.min(), ys.max())
    assert abs(xs.mean() - 50) <= 1 and abs(ys.mean() - 50) <= 1  # centred on the rectangle
    assert (alpha[40:61, 40:61] == 0).all()  # an outline, not a filled dot


@pytest.mark.parametrize("scale", [2, 0.3])
def test_opacity_shows_class_confidence(annotator: Page, tiffs, tmp_path, scale):
    """At temperature 0 nuclei are opaque, as without logits; above it, a nucleus near chance fades to LEAST_OPACITY, outline or circle."""
    nuclei = [((1900, 1500), [10, 0]), ((2100, 1500), [0.1, 0])]
    path = tmp_path / "nuclei.geojson"
    path.write_text(feature_collection([{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring(x - 10, y - 10, x + 10, y + 10)]},
                                         "properties": {"objectType": "detection", "classification": {"name": "Tumor", "color": [255, 0, 0]},
                                                        "measurements": {"logit(Tumor)": tumor, "logit(Stroma)": stroma}}}
                                        for (x, y), (tumor, stroma) in nuclei]))
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.set_input_files("#files", str(path))
    wait_for(annotator, "$('layers').querySelectorAll('label').length >= 1")
    annotator.evaluate(f"viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom({scale}), null, true)")
    annotator.evaluate("viewer.viewport.panTo(viewer.viewport.imageToViewportCoordinates(2000, 1500), true)")

    def opacity(temperature: float) -> list[int]:
        annotator.evaluate(f"$('temperature').value = {temperature}; $('temperature').dispatchEvent(new Event('input'))")
        annotator.evaluate("new Promise(requestAnimationFrame)")
        return annotator.evaluate("""[1900, 2100].map(x => {
            const {x: sx, y: sy} = toScreen(viewer, x, 1500), dpr = window.devicePixelRatio || 1;
            const data = panes[0].ctx.getImageData(Math.round(sx * dpr) - 30, Math.round(sy * dpr) - 30, 61, 61).data;
            return Math.max(...data.filter((_, i) => i % 4 === 3));
        })""")

    assert all(alpha > 200 for alpha in opacity(0))
    confident, uncertain = opacity(1)
    assert confident > 200 and 20 < uncertain < 60, (confident, uncertain)  # LEAST_OPACITY 0.15 ≈ 38


def test_paint_and_erase_rois(annotator: Page, tiffs):
    """Brush strokes become ROI rings of the stroke's area: erasing across one splits it, a painted loop leaves an exclusion zone, and undo steps back."""
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(0.25), viewer.viewport.imageToViewportCoordinates(3300, 2000), true)")
    annotator.click("#draw-roi")
    annotator.click("#roi-mode [data-value=paint]")
    annotator.evaluate("changeROIMode({brushSize: 20})")  # 40 slide px across a radius at 0.25 screen px per slide px

    def screen(x: float, y: float) -> tuple[float, float]:
        return tuple(annotator.evaluate("([x, y]) => { const point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y)), rect = viewer.container.getBoundingClientRect(); return [rect.left + point.x, rect.top + point.y]; }", [x, y]))

    def stroke(*points: tuple[float, float]):
        annotator.mouse.move(*screen(*points[0]))
        annotator.mouse.down()
        for point in points[1:]:
            annotator.mouse.move(*screen(*point), steps=10)
        annotator.mouse.up()

    def areas() -> list[float]:
        return sorted(annotator.evaluate("drawnLayer().drawn.map(ringArea)"))

    stroke((2000, 2000), (3000, 2000))
    capsule = 1000 * 80 + np.pi * 40 ** 2
    assert len(areas()) == 1 and areas()[0] == pytest.approx(capsule, rel=0.03)

    annotator.click("#roi-mode [data-value=erase]")
    stroke((2500, 1800), (2500, 2200))
    assert len(areas()) == 2 and sum(areas()) == pytest.approx(capsule - 80 * 80, rel=0.03)  # less the brush's width across it

    annotator.click("#roi-mode [data-value=paint]")
    stroke((4300, 1800), (4700, 1800), (4700, 2200), (4300, 2200), (4300, 1800))
    assert annotator.evaluate("ringDepths(drawnLayer().drawn).sort().join()") == "0,0,0,1"
    assert areas()[-1] == pytest.approx(480 ** 2, rel=0.03)  # the loop's outline, its corners rounded by the brush
    assert areas()[-2] == pytest.approx(320 ** 2, rel=0.03)  # the hole inside it

    annotator.keyboard.press("Backspace")
    assert len(areas()) == 2
    annotator.click("#roi-undo")  # touch screens have no Backspace
    assert len(areas()) == 1


def test_erasing_inside_a_roi_keeps_its_outline(annotator: Page, tiffs):
    """Erasing within a ROI cuts an exclusion zone and leaves the ROI's outline as it was; erasing within the zone again changes nothing."""
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(0.25), viewer.viewport.imageToViewportCoordinates(3300, 2000), true)")
    annotator.click("#draw-roi")
    # a ROI tens of thousands of px across, so a mask px is many slide px
    outline = annotator.evaluate("""() => {
        const ring = Array.from({length: 360}, (_, i) => [Math.round(3300 + 30000 * Math.cos(i * Math.PI / 180)), Math.round(2000 + 30000 * Math.sin(i * Math.PI / 180))]);
        changeDrawn(drawnLayer(), [ring]);
        return ring;
    }""")
    for _ in range(5):
        annotator.evaluate("applyStroke({points: [[3300, 2000]], radius: 40, erase: true})")
    drawn = annotator.evaluate("drawnLayer().drawn")
    assert len(drawn) == 2 and outline in drawn
    hole = next(ring for ring in drawn if ring != outline)
    assert annotator.evaluate("ringArea", hole) == pytest.approx(np.pi * 40 ** 2, rel=0.25)  # coarse: a mask px is some 15 slide px here
    undos = len(annotator.evaluate("drawnLayer().history"))
    annotator.evaluate("applyStroke({points: [[3290, 2000], [3310, 2000]], radius: 15, erase: true})")  # well within the zone
    assert len(annotator.evaluate("drawnLayer().history")) == undos and annotator.evaluate("drawnLayer().drawn") == drawn


def test_class_colour_width_and_merge(annotator: Page, tiffs, nuclei):
    """A class takes a new colour, width and fill; merged into another, its nuclei count there, and a rename covers both."""
    path, centres = nuclei
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.set_input_files("#files", path)
    wait_for(annotator, "$('layers').querySelectorAll('label').length >= 2")
    tumor = annotator.locator("#layers label", has_text="Tumor")
    tumor.locator(".swatch-edit").click()
    annotator.locator(".class-edit input[type=color]").evaluate("input => { input.value = '#00ff00'; input.dispatchEvent(new Event('input')); input.dispatchEvent(new Event('change')); }")
    annotator.locator(".class-edit input[type=range]").first.fill("4")
    annotator.locator(".class-edit input[type=range]").nth(1).fill("0.3")
    assert annotator.evaluate("(({color, width, fill}) => [color, width, fill])(layers[0].classes.get('Tumor'))") == [[0, 255, 0], 4, 0.3]

    annotator.locator(".class-edit select").select_option("Lymphocyte")
    assert class_counts(annotator) == {"Lymphocyte": sum(len(c) for c in centres.values())}
    annotator.locator(".class-edit input[type=text]").fill("all nuclei")  # the editor stays open on the merged class
    annotator.locator(".class-edit input[type=text]").press("Enter")
    assert class_counts(annotator) == {"all nuclei": sum(len(c) for c in centres.values())}
    assert sorted(annotator.evaluate("[...layers[0].renamed]")) == [["Lymphocyte", "all nuclei"], ["Tumor", "all nuclei"]]
