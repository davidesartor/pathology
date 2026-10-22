"""Both pages: zoom presets as through an objective, the microscope field, measurements, and the capture frame's shape and size."""
import numpy as np
from PIL import Image
from playwright.sync_api import Page

from conftest import drag, open_files, open_settings, open_folder, set_view, settle, wait_for
from slides import write_mrxs
from test_slides import screenshot


def test_capture_frame_keeps_its_shape_and_settings(app: Page, tiffs):
    """A fixed shape holds while resizing; shape and corner come back on the next visit."""
    open_files(app, tiffs["ome_zlib"][0])
    app.click("#capture")
    app.click("#capture-shape [data-value='4:3']")
    drag(app, (800, 640), (900, 700))
    app.click("#capture-corner [data-value='top-right']")
    assert app.evaluate("JSON.parse(localStorage.getItem('capture-settings')).size") is None  # kept only once a capture is saved
    app.reload()
    open_files(app, tiffs["ome_zlib"][0])
    app.click("#capture")
    assert app.locator("#capture-shape .active").get_attribute("data-value") == "4:3"
    assert app.locator("#capture-corner .active").get_attribute("data-value") == "top-right"
    size = app.locator("#capture-size").text_content()
    width, height = (int(n.replace(",", "")) for n in size.split(" px")[0].split(" × "))
    assert abs(width / height - 4 / 3) < 0.01, size


def test_capture_format_and_resolution(app: Page, tiffs):
    """Format and resolution are kept settings: half resolution halves each side, JPEG saves a .jpg."""
    open_files(app, tiffs["ome_zlib"][0])
    app.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")
    app.click("#capture")
    app.click("#capture-shape [data-value='4:3']")
    app.fill("#capture-width", "500")
    app.press("#capture-width", "Enter")
    app.click("#capture-resolution [data-value='2']")
    app.click("#capture-format [data-value='jpeg']")
    assert app.locator("#capture-size").text_content().startswith("1,000 × 750 px")
    with app.expect_download() as download:
        app.click("#capture-save")
    assert download.value.suggested_filename.endswith("_1to2.jpg")
    image = Image.open(download.value.path())
    assert image.format == "JPEG" and image.size == (1000, 750)
    app.reload()
    open_files(app, tiffs["ome_zlib"][0])
    app.click("#capture")
    assert app.locator("#capture-format .active").get_attribute("data-value") == "jpeg"
    assert app.locator("#capture-resolution .active").get_attribute("data-value") == "2"


def test_capture_has_no_seams_between_mrxs_photos(app: Page, tmp_path):
    """Turned, a capture of a flat gray MRXS slide is flat gray inside: the white fill never shows between its camera images."""
    open_folder(app, write_mrxs(tmp_path, "flat", (128, 160), (6, 6), 2, None, 0, gray=200))
    set_view(app, 30, False)
    app.click("#capture")
    app.click("#capture-corner [data-value='none']")
    with app.expect_download() as download:
        app.click("#capture-save")
    got = np.asarray(Image.open(download.value.path()).convert("RGB")).astype(int)
    height, width = got.shape[:2]
    inside = got[height // 3:2 * height // 3, width // 3:2 * width // 3]
    assert np.abs(inside - 200).max() <= 2, np.abs(inside - 200).max()
    # unturned, GPU canvases antialias too: a region read from an odd px at half size, as a downsampled capture reads it,
    # still places every camera image on whole px
    rects = app.evaluate("""async () => {
        const rects = [];
        await slide.drawRegion({drawImage: (...args) => rects.push(args.slice(5))}, 101, 101, 1101, 701, 500, 300);
        return rects;
    }""")
    assert rects and all(value == round(value) for rect in rects for value in rect), rects


def test_capture_size_in_microns(app: Page, tiffs):
    """A typed side spans that much tissue, a fixed shape carrying the other along and the view zooming out to fit; the boxes follow freehand resizing."""
    open_files(app, tiffs["ome_zlib"][0])
    app.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")
    app.click("#capture")
    app.click("#capture-shape [data-value='4:3']")
    app.fill("#capture-width", "500")
    app.press("#capture-width", "Enter")
    assert app.locator("#capture-size").text_content().startswith("2,000 × 1,500 px")
    assert app.input_value("#capture-height") == "375"
    assert app.evaluate("viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true))") <= 0.9 * 1000 / 2000 + 1e-3
    app.click("#capture-shape [data-value='free']")
    app.fill("#capture-height", "100")
    app.press("#capture-height", "Enter")
    assert app.locator("#capture-size").text_content().startswith("2,000 × 400 px")
    before = app.input_value("#capture-width")
    rect = app.evaluate("(() => { const s = viewer.viewport.getContainerSize(); return [s.x, s.y]; })()")
    scale = app.evaluate("viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true))")
    corner = (rect[0] / 2 + 1000 * scale, rect[1] / 2 - 200 * scale)  # top right, clear of the tool rail
    drag(app, corner, (corner[0] - 100, corner[1]))
    assert app.input_value("#capture-width") != before


def test_zoom_presets(app: Page, tiffs):
    """Through a 10× eyepiece, 10× shows the slide (0.25 µm per px) at its true size times 100 on screen."""
    open_files(app, tiffs["ome_zlib"][0])
    app.click("#zoom [data-zoom='10']")
    app.wait_for_timeout(1500)  # the zoom animates
    want = 0.25e-3 * 100 / app.evaluate("screenMmPerPx()")  # screen px per slide px
    assert abs(app.evaluate("viewer.viewport.viewportToImageZoom(viewer.viewport.getZoom(true))") / want - 1) < 0.02
    assert app.locator("#zoom .active").get_attribute("data-zoom") == "10"


def test_measure_line(app: Page, tiffs):
    """A dragged line is kept with its length in µm; shift keeps it level."""
    open_files(app, tiffs["ome_zlib"][0])
    app.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")
    app.click("#measure")
    drag(app, (300, 300), (500, 300))
    app.keyboard.down("Shift")
    drag(app, (300, 400), (500, 420))
    app.keyboard.up("Shift")
    first, second = app.evaluate("slide.areas")
    assert first["kind"] == "line" and abs(np.hypot(*np.diff(first["ring"], axis=0)[0]) - 200) <= 1
    assert second["ring"][0][1] == second["ring"][1][1]  # level
    assert app.locator("#areas .area .count").first.text_content() == "50 µm"
    app.click("#measure-undo")  # touch screens have no Backspace
    assert len(app.evaluate("slide.areas")) == 1


def test_fingers_pan_while_measuring_and_only_the_pen_draws_once_used(app: Page, tiffs, browser_name):
    """Two fingers pan without leaving a line end behind; once a pen is used, a palm on the screen neither draws nor moves the slide."""
    if browser_name != "chromium":
        return  # touch and pen input are sent through the DevTools protocol
    open_files(app, tiffs["ome_zlib"][0])
    app.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")
    app.click("#measure")
    cdp = app.context.new_cdp_session(app)

    def touch(kind, *points):
        cdp.send("Input.dispatchTouchEvent", {"type": kind, "touchPoints": [{"x": x, "y": y, "id": i} for i, (x, y) in points]})

    centre = app.evaluate("viewer.viewport.getCenter(true).x")
    touch("touchStart", (0, (300, 300)))
    touch("touchMove", (0, (303, 300)))
    touch("touchStart", (0, (303, 300)), (1, (400, 400)))
    for step in range(1, 11):
        touch("touchMove", (0, (303 + 20 * step, 300)), (1, (400 + 20 * step, 402)))
    touch("touchEnd", (1, (600, 402)))
    touch("touchMove", (0, (530, 300)))  # the finger left on the screen draws nothing
    touch("touchEnd")
    assert app.evaluate("viewer.viewport.getCenter(true).x") < centre  # panned right
    assert app.evaluate("slide.areas") == []
    assert app.locator("#measure").get_attribute("class") == "active"

    def pen(kind, x, y):
        cdp.send("Input.dispatchMouseEvent", {"type": kind, "x": x, "y": y, "button": "left", "buttons": 0 if kind == "mouseReleased" else 1, "clickCount": 1, "pointerType": "pen"})

    # a pencil's touches are cancelled, or iPadOS takes them for Scribble; a pen press it cancels leaves fingers drawing
    assert app.evaluate("""() => {
        const touch = Object.defineProperty(new Touch({identifier: 9, target: viewer.canvas, clientX: 300, clientY: 300}), "touchType", {value: "stylus"});  // Chrome has no touchType
        const event = new TouchEvent("touchstart", {touches: [touch], changedTouches: [touch], cancelable: true, bubbles: true});
        viewer.canvas.dispatchEvent(event);
        return event.defaultPrevented;
    }""")
    app.evaluate("""() => {
        viewer.container.dispatchEvent(new PointerEvent("pointerdown", {pointerType: "pen", pointerId: 9, bubbles: true}));
        window.dispatchEvent(new PointerEvent("pointercancel", {pointerType: "pen", pointerId: 9}));
    }""")
    touch("touchStart", (0, (300, 300)))
    for step in range(1, 11):
        touch("touchMove", (0, (300 + 10 * step, 300)))
    touch("touchEnd")
    assert len(app.evaluate("slide.areas")) == 1
    app.click("#measure-undo")

    app.evaluate("viewer.viewport.zoomTo(viewer.viewport.imageToViewportZoom(1), null, true)")  # the pinch zoomed a little
    centre = app.evaluate("viewer.viewport.getCenter(true)")
    pen("mousePressed", 300, 500)
    touch("touchStart", (0, (700, 600)))  # the palm
    for step in range(1, 11):
        pen("mouseMoved", 300 + 20 * step, 500)
        touch("touchMove", (0, (700 + 5 * step, 600 + 5 * step)))
    touch("touchEnd")
    pen("mouseReleased", 500, 500)
    assert app.evaluate("viewer.viewport.getCenter(true)") == centre
    [line] = app.evaluate("slide.areas")
    assert abs(np.hypot(*np.diff(line["ring"], axis=0)[0]) - 200) <= 1

    # the pen has drawn, so one finger now pans
    touch("touchStart", (0, (300, 300)))
    for step in range(1, 11):
        touch("touchMove", (0, (300 + 20 * step, 300)))
    touch("touchEnd")
    assert app.evaluate("viewer.viewport.getCenter(true).x") < centre["x"]
    assert len(app.evaluate("slide.areas")) == 1


def test_microscope_field_sets_magnification(app: Page, tiffs):
    """With the field on, 40× through a 10×/22 eyepiece shows a 0.55 mm circle, the slide dimmed outside it."""
    open_files(app, tiffs["ome_zlib"][0])
    app.click("#zoom [data-zoom='40']")
    settle(app)
    plain = screenshot(app, view_only=True).astype(int)
    app.click("#field")
    open_settings(app)
    app.click("#field-number [data-value='22']")
    app.click("#zoom [data-zoom='40']")
    wait_for(app, "$('status').textContent.includes('40.0×, field ⌀ 550 µm')")
    settle(app)
    assert app.locator("#zoom .active").get_attribute("data-zoom") == "40"
    # the circle is 90% of the 800 px height: the centre is lit, the left edge dimmed
    dimmed = screenshot(app, view_only=True).astype(int)
    assert dimmed[390:410, 490:510].mean() > 100 and dimmed[390:410, 20:60].mean() < 0.7 * plain[390:410, 20:60].mean()
    app.reload()
    assert app.evaluate("eyepiece.field") and "active" in app.locator("#field").get_attribute("class").split()
