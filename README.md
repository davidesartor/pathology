# Pathology slide viewer

Whole-slide image viewer and annotator that runs entirely in the browser.

- **Viewer:** https://davidesartor.github.io/pathology/slide-viewer.html
  Open TIFF/SVS/MRXS slides, rotate/flip, highlight areas, measure and capture images with a scale bar.
- **Annotator:** https://davidesartor.github.io/pathology/slide-annotator.html
  Everything above, plus GeoJSON overlays, ROIs and in-browser nucleus detection.

Each page is a single HTML file. You can also download it and open it locally by double-clicking; there is nothing to install.

## Disclaimer

**Research and educational use only. This is not a medical device.** It has no CE marking, FDA clearance or any other regulatory approval. Do not use it for diagnosis, treatment or any other clinical decision.

Displayed images, measurements, scale bars, detections and classifications may be inaccurate or incomplete. Check anything that matters against a validated system.

The software is provided "as is", without warranty of any kind, and the authors accept no liability for its use (see [LICENSE](LICENSE)). You are responsible for using it in line with the data-protection law (e.g. GDPR) and institutional rules that apply to your slides.

## Privacy

Slides are read from your device with the browser's File API. They are never uploaded. The page's Content-Security-Policy limits network access to the pinned, integrity-checked libraries listed below and to the Zenodo model downloads, and it blocks form submissions.

This is a technical safeguard, not a guarantee. Browser extensions, a modified copy of the page or your own network setup are outside its control.

## Third-party components

These are loaded at runtime from their original sources and are not redistributed here. Each is under its own licence.

| Component | Source | Licence |
|---|---|---|
| OpenSeadragon 5.0.1 | jsDelivr | BSD-3-Clause |
| geotiff.js 3.0.5 | jsDelivr | MIT |
| @cornerstonejs/codec-openjpeg 1.3.6 (OpenJPEG) | jsDelivr | MIT (OpenJPEG: BSD-2-Clause) |
| onnxruntime-web 1.30.0 (annotator only) | jsDelivr | MIT |
| HoVer-NeXt weights, Lizard/PanNuke ([Zenodo 10635618](https://zenodo.org/records/10635618)) | Zenodo | CC BY-NC-SA 4.0 (non-commercial) |
| HoVer-NeXt weights, PUMA melanoma ([Zenodo 15526308](https://zenodo.org/records/15526308)) | Zenodo | CC BY 4.0 |
| CellViT++ weights ([Zenodo 15024474](https://zenodo.org/records/15024474)) | Zenodo | Set by the CellViT authors; trained on PanNuke (CC BY-NC-SA 4.0) |

The model weights are downloaded into your browser only when you choose to run detection. Their licences (some non-commercial) apply to that use. If you use their outputs in a publication, cite the original model papers.

## Tests

```bash
uv run pytest --browser-channel chrome
uv run pytest --browser webkit
```
