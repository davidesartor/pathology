import base64
import functools
import re
import http.server
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from playwright.sync_api import Page

from slides import pattern, write_mrxs, write_tiff

ROOT = Path(__file__).parent.parent
PAGES = ["slide-viewer.html", "slide-annotator.html"]
SIZE = {"width": 1000, "height": 800}


@pytest.fixture(scope="session")
def server():
    """The repo served over HTTP, as GitHub Pages serves it."""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass
    handler = functools.partial(Quiet, directory=ROOT)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()


@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    return {**browser_context_args, "viewport": SIZE, "device_scale_factor": 1, "accept_downloads": True}


def wait_for(page: Page, expression: str, timeout: float = 30):
    """Poll a JS expression until truthy; wait_for_function needs eval, which the page's CSP forbids in WebKit."""
    deadline = time.monotonic() + timeout
    while not (value := page.evaluate(expression)):
        if time.monotonic() > deadline:
            raise TimeoutError(f"{expression} still falsy; loading: {page.evaluate('$(\"loading\").textContent')!r}")
        time.sleep(0.05)
    return value


def load(page: Page, server: str, name: str) -> Page:
    # the save picker would block; without it the pages save as downloads
    page.add_init_script("window.showSaveFilePicker = undefined")
    page.goto(f"{server}/{name}")
    wait_for(page, "typeof viewer !== 'undefined' && viewer.isOpen !== undefined")
    return page


def settle(page: Page):
    """Wait until the slide is open and every tile in view is drawn."""
    wait_for(page, "slide?.source && viewer.world.getItemCount() > 0 && viewer.world.getItemAt(0).getFullyLoaded()")
    page.evaluate("new Promise(requestAnimationFrame)")


def open_files(page: Page, *paths: Path):
    page.set_input_files("#slide-file", [str(path) for path in paths])
    settle(page)


def open_folder(page: Page, folder: Path):
    page.set_input_files("#slide-folder", str(folder))
    settle(page)


def pointer_at(page: Page, x: int, y: int) -> tuple[int, int]:
    """Move the mouse to screen (x, y) and return the slide px the page's readout names there."""
    page.mouse.move(x, y)
    page.evaluate("new Promise(requestAnimationFrame)")
    status = page.evaluate("$('status').textContent")
    slide_x, slide_y = map(int, re.search(r"x (-?\d+)  y (-?\d+)", status).groups())
    return slide_x, slide_y


def set_view(page: Page, rotation: float, flip: bool):
    """Turn and mirror the view with the page's own controls."""
    page.locator("#rotation").fill(str(rotation))
    if page.evaluate("viewer.viewport.getFlip()") != flip:
        page.click("#flip")
    page.evaluate("new Promise(requestAnimationFrame)")
    settle(page)


TILE = 512


def read_tile(page: Page, downsample: int, x: int, y: int) -> np.ndarray:
    """RGBA of the screen tile at column x, row y of the 1/downsample level, as the page's tile source draws it."""
    data = page.evaluate("""async ([downsample, x, y]) => {
        const {source} = slide;
        const src = source.getTileUrl(source.maxLevel - Math.log2(downsample), x, y);
        const bitmap = await new Promise((resolve, reject) =>
            source.downloadTileStart({src, finish: (data, request, error) => error ? reject(new Error(error)) : resolve(data)}));
        const context = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d");
        context.drawImage(bitmap, 0, 0);
        const bytes = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
        let text = "";
        for (let i = 0; i < bytes.length; i += 0x8000) text += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
        return [bitmap.width, bitmap.height, btoa(text)];
    }""", [downsample, x, y])
    width, height, pixels = data
    return np.frombuffer(base64.b64decode(pixels), np.uint8).reshape(height, width, 4)


def read_level(page: Page, downsample: int) -> np.ndarray:
    """The whole slide at 1/downsample, stitched from its screen tiles."""
    width, height = page.evaluate("[slide.width, slide.height]")
    width, height = -(-width // downsample), -(-height // downsample)
    rows = [np.concatenate([read_tile(page, downsample, x, y) for x in range(-(-width // TILE))], axis=1) for y in range(-(-height // TILE))]
    return np.concatenate(rows, axis=0)


@pytest.fixture(params=PAGES)
def app(request, page: Page, server: str) -> Page:
    return load(page, server, request.param)


@pytest.fixture
def annotator(page: Page, server: str) -> Page:
    return load(page, server, "slide-annotator.html")


@pytest.fixture
def viewer(page: Page, server: str) -> Page:
    return load(page, server, "slide-viewer.html")


@pytest.fixture(scope="session")
def tiffs(tmp_path_factory) -> dict[str, tuple[Path, list[np.ndarray]]]:
    """name -> (path, stored levels); level 0 above the pages' 64 MB whole-image decode, so both tile paths run."""
    folder = tmp_path_factory.mktemp("tiff")
    image = pattern(4000, 6000)
    return {
        "ome_zlib": (folder / "ome_zlib.ome.tiff", write_tiff(folder / "ome_zlib.ome.tiff", image, 4, subifds=True, compression="zlib")),
        "pages_lzw": (folder / "pages_lzw.tiff", write_tiff(folder / "pages_lzw.tiff", image, 4, subifds=False, compression="lzw")),
    }


@pytest.fixture(scope="session")
def mrxs(tmp_path_factory) -> dict[str, Path]:
    """name -> MRXS slide folder: a plain camera grid, and stitched overlapping photos with one never scanned."""
    root = tmp_path_factory.mktemp("mrxs")
    photo, grid, overlap = (128, 160), (5, 6), 20
    rows, cols = np.meshgrid(np.arange(grid[0]) * (photo[0] - overlap), np.arange(grid[1]) * (photo[1] - overlap), indexing="ij")
    jitter = np.random.default_rng(0).integers(0, 3, (*grid, 2)) * 4  # multiples of 4, so every level's photos sit on whole px
    return {
        "grid": write_mrxs(root / "grid", "grid", photo, grid, 3, None, 0),
        "stitched": write_mrxs(root / "stitched", "stitched", photo, grid, 3, np.stack([rows, cols], -1) + jitter, overlap, missing={8}),
    }
