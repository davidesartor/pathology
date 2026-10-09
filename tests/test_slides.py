"""Both pages: slides open and read pixel-exact, nothing leaves the device, and the view turns and mirrors consistently."""
import re

import numpy as np
import openslide
import pytest
from playwright.sync_api import Page

from conftest import ROOT, PAGES, SIZE, TILE, open_files, open_folder, pointer_at, read_level, read_tile, set_view, settle

ALLOWED = re.compile(r"https://cdn\.jsdelivr\.net/npm/(openseadragon@5\.0\.1|geotiff@3\.0\.5|@cornerstonejs/codec-openjpeg@1\.3\.6|onnxruntime-web@1\.30\.0)/")


@pytest.mark.parametrize("name", ["ome_zlib", "pages_lzw"])
def test_tiff_tiles_are_exact(app: Page, tiffs, name):
    path, levels = tiffs[name]
    open_files(app, path)
    assert app.evaluate("[slide.width, slide.height, slide.mpp]") == [6000, 4000, pytest.approx(0.25)]
    for downsample, level in zip([1, 2, 4, 8], levels):
        columns, rows = -(-level.shape[1] // TILE), -(-level.shape[0] // TILE)
        for x, y in {(0, 0), (columns // 2, rows // 2), (columns - 1, rows - 1)}:  # the last is clipped by the slide's edge
            got = read_tile(app, downsample, x, y)
            np.testing.assert_array_equal(got[..., :3], level[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE], err_msg=f"1/{downsample} tile {x},{y}")


@pytest.mark.parametrize("name", ["grid", "stitched"])
def test_mrxs_matches_openslide(app: Page, mrxs, name):
    folder = mrxs[name]
    reference = openslide.OpenSlide(folder.parent / f"{folder.name}.mrxs")
    open_folder(app, folder)
    assert app.evaluate("[slide.width, slide.height]") == list(reference.dimensions)
    for level, downsample in enumerate([1, 2, 4]):
        want = np.asarray(reference.read_region((0, 0), level, reference.level_dimensions[level])).astype(int)
        want = want[..., :3] * want[..., 3:] // 255 + 255 - want[..., 3:]  # unscanned areas show the slide's white fill
        got = read_level(app, downsample)[:want.shape[0], :want.shape[1], :3]
        np.testing.assert_array_equal(got, want, err_msg=f"level {level}")


def test_requests_stay_on_pinned_cdn(app: Page, server, tiffs, mrxs):
    """Slides never leave the device: after opening and browsing slides, every request went to this site or the pinned libraries."""
    requests = []
    app.on("request", lambda request: requests.append(request.url))
    open_files(app, tiffs["ome_zlib"][0])
    open_folder(app, mrxs["stitched"])
    app.evaluate("viewer.viewport.zoomTo(viewer.viewport.getMaxZoom(), null, true)")
    settle(app)
    outside = [url for url in requests if not url.startswith(server) and not ALLOWED.match(url) and not url.startswith(("blob:", "data:"))]
    assert not outside


@pytest.mark.parametrize("name", PAGES)
def test_csp_allows_no_other_hosts(name):
    html = (ROOT / name).read_text()
    policy = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html).group(1)
    directives = dict(part.strip().split(" ", 1) for part in policy.split(";"))
    assert directives["default-src"] == "'none'"
    assert directives["form-action"] == "'none'"
    for source in directives["connect-src"].split():
        assert source == "blob:" or ALLOWED.match(source) or source == "https://zenodo.org/api/records/", source


@pytest.mark.parametrize("rotation, flip", [(0, False), (90, False), (-30, False), (0, True), (135, True)])
def test_pointer_reads_the_pixel_under_it(app: Page, tiffs, rotation, flip):
    """At 4 screen px per slide px, the readout under the pointer names the slide px drawn there."""
    path, (image, *_) = tiffs["ome_zlib"]
    open_files(app, path)
    app.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(4), null, true)")
    set_view(app, rotation, flip)
    screen = screenshot(app)
    checked = 0
    for x, y in np.random.default_rng(abs(rotation) + flip).integers([0, 0], [SIZE["width"], SIZE["height"]], (80, 2)):
        if app.evaluate("([x, y]) => !document.elementFromPoint(x, y).closest('.openseadragon-canvas')", [int(x), int(y)]):
            continue  # under the panels or the navigator
        slide_x, slide_y = pointer_at(app, int(x), int(y))
        # the readout rounds and the screen blends neighbours, so skip points near block edges or where blue wraps
        if np.ptp(image[slide_y - 2:slide_y + 3, slide_x - 2:slide_x + 3].reshape(-1, 3).astype(int), axis=0).max() > 12:
            continue
        candidates = image[slide_y - 1:slide_y + 2, slide_x - 1:slide_x + 2].reshape(-1, 3).astype(int)
        # a blend of the px around the readout lies within their range
        assert np.all(candidates.min(0) - 3 <= screen[y, x]) and np.all(screen[y, x] <= candidates.max(0) + 3), (rotation, flip, x, y, slide_x, slide_y)
        checked += 1
    assert checked >= 20


def screenshot(page: Page, view_only: bool = False) -> np.ndarray:
    """The page as RGB; view_only hides the panels and navigator drawn over the slide."""
    from io import BytesIO
    from PIL import Image
    hide = "for (const element of document.querySelectorAll('#panel, #info, #tabs, .navigator')) element.style.visibility = '%s'"
    if view_only:
        page.evaluate(hide % "hidden")
    png = page.screenshot()
    if view_only:
        page.evaluate(hide % "")
    return np.asarray(Image.open(BytesIO(png)).convert("RGB")).astype(int)
