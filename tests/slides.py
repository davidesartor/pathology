"""Synthetic TIFF and MIRAX slides with known pixels, small enough to write per test session."""
import struct
from pathlib import Path

import numpy as np
import tifffile
from einops import rearrange, reduce
from jaxtyping import UInt8
from PIL import Image


def pattern(height: int, width: int, seed: int = 0) -> UInt8[np.ndarray, "H W 3"]:
    """Gradients that identify a pixel's position, plus noise blocks so misplaced tiles show."""
    y, x = np.mgrid[:height, :width]
    blocks = np.random.default_rng(seed).integers(0, 64, (height // 32 + 1, width // 32 + 1))
    noise = blocks[y // 32, x // 32]
    return np.stack([x * 191 // width + noise, y * 191 // height + noise, (x + y) % 256], axis=-1).astype(np.uint8)


def halve(image: UInt8[np.ndarray, "H W 3"]) -> UInt8[np.ndarray, "h w 3"]:
    """2x2 box downsample, as a scanner or tiff writer would store the next level."""
    height, width = image.shape[0] // 2 * 2, image.shape[1] // 2 * 2
    return reduce(image[:height, :width].astype(np.float32), "(h a) (w b) c -> h w c", "mean", a=2, b=2).round().astype(np.uint8)


def write_tiff(path: Path, image: UInt8[np.ndarray, "H W 3"], levels: int, subifds: bool, compression: str, mpp: float = 0.25) -> list[np.ndarray]:
    """Tiled pyramid as OME-style SubIFDs or SVS-style further pages; returns the stored levels."""
    pyramid = [image]
    for _ in range(levels - 1):
        pyramid.append(halve(pyramid[-1]))
    resolution = (1e4 / mpp, 1e4 / mpp)
    options = dict(tile=(256, 256), photometric="rgb", compression=compression)
    with tifffile.TiffWriter(path, bigtiff=False) as tiff:
        tiff.write(pyramid[0], subifds=levels - 1 if subifds else 0, resolution=resolution, resolutionunit="CENTIMETER", **options)
        for level in pyramid[1:]:
            tiff.write(level, subfiletype=1, **options)
    return pyramid


def write_mrxs(root: Path, name: str, photo: tuple[int, int], grid: tuple[int, int], levels: int, positions: np.ndarray | None,
               overlap: int, missing: set[int] = frozenset(), seed: int = 1) -> Path:
    """MIRAX slide: root/name.mrxs and root/name/{Slidedat.ini, Index.dat, Data0000.dat}, laid out as OpenSlide reads it.

    positions (Y X 2, level-0 px) go in a VIMSLIDE_POSITION_BUFFER; None leaves the nominal grid. Photos in `missing` are not scanned.
    """
    (photo_height, photo_width), (images_y, images_x) = photo, grid
    folder = root / name
    folder.mkdir(parents=True)
    (root / f"{name}.mrxs").write_bytes(b"")
    photos = rearrange(pattern(images_y * photo_height, images_x * photo_width, seed), "(y h) (x w) c -> y x h w c", h=photo_height, w=photo_width)

    data = bytearray()
    def store(blob: bytes) -> tuple[int, int]:
        data.extend(blob)
        return len(data) - len(blob), len(blob)
    def png(image: np.ndarray) -> bytes:
        path = folder / "photo.png"
        Image.fromarray(image).save(path)
        blob = path.read_bytes()
        path.unlink()
        return blob

    # level n images are 2^n x 2^n photos, each halved n times into its own quadrant, indexed by their top-left photo
    level_entries = []
    for level in range(levels):
        span = 2 ** level
        entries = []
        for y in range(0, images_y, span):
            for x in range(0, images_x, span):
                covered = [(yy, xx) for yy in range(y, min(y + span, images_y)) for xx in range(x, min(x + span, images_x))]
                if all(yy * images_x + xx in missing for yy, xx in covered):
                    continue
                canvas = np.full((photo_height, photo_width, 3), 255, np.uint8)
                for yy, xx in covered:
                    if yy * images_x + xx in missing:
                        continue
                    small = photos[yy, xx]
                    for _ in range(level):
                        small = halve(small)
                    top, left = (yy - y) * photo_height // span, (xx - x) * photo_width // span
                    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small
                entries.append((y * images_x + x, *store(png(canvas)), 0))
        level_entries.append(entries)

    nonhier = []
    if positions is not None:
        records = b"".join(struct.pack("<bii", 1, int(px), int(py)) for py, px in rearrange(positions, "y x c -> (y x) c"))
        nonhier.append(("VIMSLIDE_POSITION_BUFFER", "default", *store(records), 0))
    (folder / "Data0000.dat").write_bytes(bytes(data))

    # Index.dat: version, slide id, then pointers to the level and non-hierarchical record tables; each record is (0, page) and
    # each page is (entry count, next page, entries)
    slide_id = "SYNTHETIC0000"
    index = bytearray(b"01.02" + slide_id.encode() + bytes(8))
    def append(*ints: int) -> int:
        index.extend(struct.pack(f"<{len(ints)}i", *ints))
        return len(index) - 4 * len(ints)
    hier_table = append(*[0] * levels)
    nonhier_table = append(*[0] * len(nonhier)) if nonhier else len(index)
    for level, entries in enumerate(level_entries):
        page = append(len(entries), 0, *[value for entry in entries for value in entry])
        struct.pack_into("<i", index, hier_table + 4 * level, append(0, page))
    for i, (_, _, offset, length, file_number) in enumerate(nonhier):
        page = append(1, 0, 0, 0, offset, length, file_number)
        struct.pack_into("<i", index, nonhier_table + 4 * i, append(0, page))
    struct.pack_into("<ii", index, 5 + len(slide_id), hier_table, nonhier_table)
    (folder / "Index.dat").write_bytes(bytes(index))

    lines = ["[GENERAL]", f"SLIDE_ID = {slide_id}", f"IMAGENUMBER_X = {images_x}", f"IMAGENUMBER_Y = {images_y}", "CameraImageDivisionsPerSide = 1",
             "[HIERARCHICAL]", f"HIER_COUNT = 1", "HIER_0_NAME = Slide zoom level", f"HIER_0_COUNT = {levels}",
             *[f"HIER_0_VAL_{level} = ZoomLevel_{level}\nHIER_0_VAL_{level}_SECTION = LAYER_0_LEVEL_{level}_SECTION" for level in range(levels)],
             f"NONHIER_COUNT = {len(nonhier)}",
             *[f"NONHIER_{i}_NAME = {record}\nNONHIER_{i}_COUNT = 1\nNONHIER_{i}_VAL_0 = {value}" for i, (record, value, *_) in enumerate(nonhier)],
             "INDEXFILE = Index.dat", "[DATAFILE]", "FILE_COUNT = 1", "FILE_0 = Data0000.dat"]
    for level in range(levels):
        lines += [f"[LAYER_0_LEVEL_{level}_SECTION]", f"OVERLAP_X = {overlap if level == 0 else 0}", f"OVERLAP_Y = {overlap if level == 0 else 0}",
                  f"MICROMETER_PER_PIXEL_X = {0.25 * 2 ** level}", f"MICROMETER_PER_PIXEL_Y = {0.25 * 2 ** level}",
                  "IMAGE_FILL_COLOR_BGR = 16777215", f"DIGITIZER_WIDTH = {photo_width}", f"DIGITIZER_HEIGHT = {photo_height}",
                  "IMAGE_FORMAT = PNG", f"IMAGE_CONCAT_FACTOR = {0 if level == 0 else 1}"]
    (folder / "Slidedat.ini").write_text("\r\n".join(lines) + "\r\n")
    return folder
