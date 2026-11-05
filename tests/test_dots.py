"""Annotator: dots placed by hand with the Dots tool are counted as nuclei, inside the ROIs shown, change class or go with a click, undo, and save as QuPath points that load back."""
import json

from playwright.sync_api import Page

from conftest import open_files, wait_for
from test_annotator import class_counts, feature_collection, ring, stats_note


def screen(page: Page, x: float, y: float) -> tuple[float, float]:
    """Page px of slide px (x, y)."""
    return tuple(page.evaluate("([x, y]) => { const point = viewer.viewport.imageToViewerElementCoordinates(new OpenSeadragon.Point(x, y)), rect = viewer.container.getBoundingClientRect(); return [rect.left + point.x, rect.top + point.y]; }", [x, y]))


def dot_counts(page: Page) -> dict[str, int]:
    """The Dots layer's classes that have dots, by their counts in the layers."""
    names = page.evaluate("dotClasses(layers.find(layer => layer.dotTool)).map(cls => cls.name)")
    return {name: count for name, count in class_counts(page).items() if count and name in names}


def test_dots_are_placed_changed_and_undone(annotator: Page, tiffs):
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(0.25), viewer.viewport.imageToViewportCoordinates(3000, 2000), true)")
    annotator.click("#dots")
    assert annotator.locator("#dot-classes button.active").inner_text().split("\n")[1] == "Tumor"
    for x in (2000, 2400, 2800):
        annotator.mouse.click(*screen(annotator, x, 2000))
    annotator.keyboard.press("2")
    for x in (2000, 2400, 2800, 3200):
        annotator.mouse.click(*screen(annotator, x, 2400))
    assert dot_counts(annotator) == {"Tumor": 3, "Lymphocyte": 4}
    assert "Lymphocyte / (Lymphocyte + Tumor): 57.1%" in stats_note(annotator)

    annotator.mouse.click(*screen(annotator, 2010, 2005))  # on a tumour dot: it becomes a lymphocyte
    assert dot_counts(annotator) == {"Tumor": 2, "Lymphocyte": 5}
    annotator.mouse.click(*screen(annotator, 3200, 2400))  # on a lymphocyte dot: it goes
    assert dot_counts(annotator) == {"Tumor": 2, "Lymphocyte": 4}
    assert annotator.evaluate("viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom())") == 0.25  # clicks did not zoom

    annotator.keyboard.press("Backspace")
    assert dot_counts(annotator) == {"Tumor": 2, "Lymphocyte": 5}
    annotator.click("#dots-undo")
    assert dot_counts(annotator) == {"Tumor": 3, "Lymphocyte": 4}
    tumor = annotator.evaluate("layers.find(layer => layer.dotTool).classes.get('Tumor').centres")
    assert sorted(zip(tumor[::2], tumor[1::2])) == [(2000, 2000), (2400, 2000), (2800, 2000)]

    annotator.fill("#dot-class-name", "Mitosis")
    annotator.press("#dot-class-name", "Enter")
    assert annotator.locator("#dot-classes button.active").inner_text().split("\n")[1] == "Mitosis"
    annotator.mouse.click(*screen(annotator, 3000, 2800))
    assert dot_counts(annotator) == {"Tumor": 3, "Lymphocyte": 4, "Mitosis": 1}
    annotator.keyboard.press("Escape")
    assert annotator.evaluate("!dotting")


def test_dots_counted_inside_rois(annotator: Page, tiffs, tmp_path):
    rois = tmp_path / "rois.geojson"
    rois.write_text(feature_collection([{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring(1000, 1000, 3000, 3000)]}, "properties": {"objectType": "annotation"}}]))
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.set_input_files("#roi-file", str(rois))
    annotator.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(0.2), viewer.viewport.imageToViewportCoordinates(3000, 2000), true)")
    annotator.click("#dots")
    for x, y in [(1500, 1500), (2500, 2500), (3500, 2000)]:
        annotator.mouse.click(*screen(annotator, x, y))
    assert dot_counts(annotator) == {"Tumor": 2}
    annotator.keyboard.press("2")
    annotator.mouse.click(*screen(annotator, 2000, 2000))
    assert stats_note(annotator).split("\n") == [  # the dot outside the ROI left out; 2000² px at 0.25 µm/px is 0.25 mm²
        "Lymphocyte / (Lymphocyte + Tumor): 33.3%",
        "Lymphocyte / all nuclei: 33.3%",
        "Lymphocyte: 4 per mm² (0.3 mm² of ROI)",
        "counted inside the ROIs and annotations shown",
    ]


def test_dots_save_as_points_and_load_back(annotator: Page, tiffs, tmp_path):
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(0.25), viewer.viewport.imageToViewportCoordinates(3000, 2000), true)")
    annotator.click("#dots")
    for x in (2000, 2400):
        annotator.mouse.click(*screen(annotator, x, 2000))
    annotator.keyboard.press("2")
    annotator.mouse.click(*screen(annotator, 2400, 2400))
    assert "(not saved)" in annotator.locator("#layers h4").inner_text()
    with annotator.expect_download() as download:
        annotator.locator("#layers h4 button").first.click()
    assert download.value.suggested_filename.endswith("_dots.geojson")
    path = tmp_path / download.value.suggested_filename
    download.value.save_as(path)
    saved = json.loads(path.read_text())
    assert [(f["properties"]["classification"]["name"], f["geometry"]) for f in saved["features"]] == [
        ("Tumor", {"type": "MultiPoint", "coordinates": [[2000, 2000], [2400, 2000]]}),
        ("Lymphocyte", {"type": "MultiPoint", "coordinates": [[2400, 2400]]}),
    ]
    assert "(not saved)" not in annotator.locator("#layers h4").inner_text()

    # on a fresh page the file is the slide's dots again, and placing more goes on with it
    annotator.reload()
    open_files(annotator, tiffs["ome_zlib"][0])
    annotator.set_input_files("#files", str(path))
    wait_for(annotator, "layers.some(layer => layer.dotTool)")
    assert dot_counts(annotator) == {"Tumor": 2, "Lymphocyte": 1}
    annotator.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(0.25), viewer.viewport.imageToViewportCoordinates(3000, 2000), true)")
    annotator.click("#dots")
    annotator.mouse.click(*screen(annotator, 2800, 2000))
    assert dot_counts(annotator) == {"Tumor": 3, "Lymphocyte": 1}
    assert annotator.evaluate("layers.length") == 1
