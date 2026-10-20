# Inspectra AI
## GitHub downloads and repository layout

This repository preserves the supplied project layout. Large assets are published in [GitHub Releases](https://github.com/ArviZuko/Inspectra-AI/releases/tag/project-assets-v1); see [DOWNLOADS.md](DOWNLOADS.md) for the complete list and restoration instructions. The installer and portable application are provided there, as are all trained checkpoints and the restoration/super-resolution datasets. Download notes also appear in their original folders.

```text
Dataset/                       Defect data in Git; large dataset download notes
Documentation/                 Project report and presentation
Screenshot/                    Application screenshots
SRC/
  Application/                 Portable executable download instructions
  Installer/                   Installer download instructions
  Models/                      Research notebooks, plots, metrics, weight links
  Source Code/
    app.py                     Consolidated desktop application
    Models/                    Runtime checkpoint download instructions
requirements.txt               Supplied pinned runtime dependencies
```

For this layout, download and extract the model archives into the repository root, then run from the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe 'SRC\Source Code\app.py'
```

The sections below retain the supplied application's usage and technical documentation. References there to a flat `app.py`/`models` directory describe the source app directory above. Reported build/test results are from the supplied documentation and were not independently reproduced during this repository upload. One embedded API-key value in a research notebook was replaced with a placeholder before publication.

Inspectra AI is a Windows desktop application for assembly-line image inspection. It combines image restoration, 2x super-resolution, and detection of Blowhole, Crack, Fray, and Break defects. Inspection logic, model architectures, quality scoring, reporting, and the window icon are integrated into `app.py`.

## 1. Ways to run the application

### Recommended: installer

1. Run `installer.exe`.
2. Choose the installation folder. The default is `%LOCALAPPDATA%\Programs\Inspectra AI`.
3. Optionally enable the desktop shortcut.
4. Launch **Inspectra AI** from the Start menu or shortcut.

The installer includes the application, all 22 trained checkpoints, and the packaged Python libraries. An additional Python installation or separate models directory is not needed for this packaged version. Installation is per user and does not request administrator privileges. The current installer targets Windows 10/11 x64-compatible systems.

The installer is approximately 1.25 GB because it includes the models and runtime. First launch may take longer while the single-file application extracts its bundled runtime. Allow several GB of free space for installation, temporary extraction, and inspection data. No exact minimum RAM or production throughput has been benchmarked; start with small sample images when assessing a new computer.

### Developer/source version

Tested environment: **Python 3.10.20, 64-bit, on Windows**. Use Python 3.10 x64 to reproduce the tested environment. Other Python versions have not been verified.

Place the following in the same project directory:

```text
InspectraAI/
    app.py
    requirements.txt
    models/
        restoration/
        super_resolution/
        Blowhole/
        crack/
        fray/
        break/
```

Open PowerShell in that directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

These commands do not require activation of the environment or a PowerShell execution-policy change. Use `python` in place of `py -3.10` only if it refers to the intended Python 3.10 x64 installation. Internet access is needed to download dependencies during setup; the trained checkpoints must already be present locally.

`app.py` plus the `models` directory are the application-specific runtime files. `requirements.txt` installs external libraries; it does not include the checkpoints. The previous helper files `batch_reporting.py`, `inspection_reporting.py`, and `automatic_quality.py` are integrated and are not required separately.

The old `Run_Studio.cmd`, if present, may contain a computer-specific Python path. Use the command above when running on another computer.

## 2. Python libraries

`requirements.txt` pins the runtime versions from the environment used for the tested build:

| Library | Purpose |
| --- | --- |
| PyQt6 | Desktop interface, camera preview, image display, zoom, and controls |
| NumPy | Image arrays and quality calculations |
| opencv-python | Camera capture, image operations, and embedded BRISQUE SVM inference |
| Pillow | Image loading, conversion, and saving |
| torch | Neural-network model loading and inference |
| torchvision | Vision backbones and dependencies used by segmentation models |
| segmentation-models-pytorch | DeepLabV3+ and U-Net++ model construction |
| transformers | SegFormer model construction and inference |
| timm | Vision encoder/model support |
| SciPy | Numerical support used by the image/model stack |
| scikit-image | Structural similarity (SSIM) calculations |
| ReportLab | PDF report generation |
| pypdf | PDF validation in the packaged self-test |

Pip installs their transitive dependencies, such as Qt runtime packages, automatically. Python standard-library modules (`sqlite3`, `json`, `pathlib`, `base64`, and others) do not need separate installation. Do not try to install `cv2`, `PIL`, or `skimage` by those import names; their distribution names are listed above.

`requirements-build.txt` adds PyInstaller for rebuilding the executable. Inno Setup is a separate Windows program and is not installed by pip. The source app does not require `opencv-contrib-python`, external BRISQUE model files, TensorFlow, or Jupyter notebooks. Avoid installing multiple OpenCV distributions into the same environment.

The packaged build was verified with CPU inference. The source detects CUDA when a compatible GPU and PyTorch installation are available, but a CUDA configuration has not been validated as part of this release.

## 3. Trained model files

The current source expects these checkpoints under its sibling `models` directory. Preserve their filenames and subdirectories:

```text

```

There are 3 restoration models, 4 super-resolution models, and 15 defect models. Restoration supports MIRNet v2, MPRNet, and NAFNet. Super-resolution supports ESRGAN, RCAN, SRCNN, and SwinIR. Defect detection supports SegFormer, DeepLabV3+, custom HRNet, and U-Net++, subject to available checkpoints for each defect. This collection does not contain a U-Net++ Blowhole model.

Model architecture must match its checkpoint. Renaming an unrelated checkpoint does not make it compatible. Use **Refresh Models** after adding compatible model files. Load only checkpoints from sources you trust, because checkpoint loading may deserialize Python objects.

## 4. Inspection workflow

1. Enter **Worker ID**, **Batch Number**, **Product ID**, and a positive **Samples in Batch** value. Worker ID, batch number, and product ID are required before inspection; batch capacity is required for sample collection.
2. Choose restoration and super-resolution options. **Choose Best Model (Automatic)** is the initial option; choose one model or skip a stage as needed.
3. Select one or more defect types. Use **Choose All** or **Clear All** below the four defect choices.
4. Choose **Choose Best Model (Automatic)** or a single detection model family. A single family is applied to the selected defect types where its checkpoints are available.
5. Upload an image or capture an image from the camera. A captured/uploaded sample is registered in the current batch once its metadata is valid. Sample numbering is managed by the batch collection.
6. Click **Run Inspection**. The progress bar shows processing progress. Automatic selection can take longer because every available candidate must run.
7. Review the four panels: original, restored, super-resolution, and combined defect detection. Use **Zoom in**, **Zoom out**, and **Fit** on each panel; drag to pan a zoomed image.
8. Click the defect legend buttons to display one defect type or a combination. These are display filters; they do not rerun inference or remove detections from saved batch statistics.
9. After inspection finishes, record **Accepted**, **Rejected**, or **Needs Review**. Decisions are manual and are not inferred from model scores. The decision control is disabled before/during inspection.
10. Open **View Metrics** for image-quality measurements and detection confidence. Use **Save Report** for the current batch summary.

Detection always uses the latest successful processing stage: super-resolution when available, otherwise restoration, otherwise the original. No detection-input choice is required. A processing warning may indicate a failed candidate or stage even when another candidate produced a usable result; review the warning and actual displayed results.

## 5. What automatic model selection means

### Restoration and super-resolution

Each available candidate runs. With an aligned clean reference, the highest SSIM wins. Without a reference, the lowest BRISQUE score wins. Only the winning output is displayed for each stage. Super-resolution operates on the restored image if restoration succeeded, or on the original otherwise.

BRISQUE is a no-reference perceptual quality estimate. SSIM measures agreement with the provided reference. Neither guarantees the best defect-detection accuracy for every surface or camera condition.

### Defect detection

For each selected defect type, available candidates run and the result with the highest **area-weighted detected-region confidence** is selected. This is an estimated ranking based on the model outputs; it is not an accuracy comparison against labeled ground truth. Confidence scales can differ across architectures. A model with no detected regions scores zero for this selection rule.

## 6. Metrics and clean reference images

| Metric | Meaning | Interpretation |
| --- | --- | --- |
| PSNR (dB) | Agreement with a clean reference based on squared error | Higher agreement is better; perfect matches have infinite PSNR |
| SSIM | Structural similarity to a clean reference | Higher agreement is better |
| MSE | Mean squared pixel error against a reference | Lower is better |
| MAE | Mean absolute pixel error against a reference | Lower is better |
| BRISQUE | Image-quality estimate without a reference | Lower is the selection preference |
| Sharpness | Variance of the Laplacian | Descriptive; higher does not always mean better quality |
| Entropy | Grayscale information content in bits | Descriptive; noise may also increase it |
| Contrast | Standard deviation of grayscale intensity | Descriptive; depends on image content |
| Stage time | Processing time in seconds | Compare on consistent hardware and image sizes |

**Why PSNR/SSIM/MSE/MAE can show N/A:** a suitable reference has not been loaded, dimensions do not match the output, or the measurement cannot be computed. Source metrics compare grayscale values normalized to `[0, 1]`. SSIM also requires an adequately sized image.

A reference should depict the same scene, aligned to the output, with the intended clean/high-resolution appearance. The degraded input is not automatically a clean reference. Comparing against the original input measures similarity to that input, not recovery of unknown ground truth. For super-resolution, use a reference matching the final output resolution; a low-resolution original will not match the dimensions of a 2x output.

In **View Metrics**, use **Load clean reference** for each stage. References are cleared when a new sample image is accepted. After changing a reference, rerun automatic selection to choose the winner using that reference. Detection confidence is a model score, not a statistical confidence interval or a calibrated probability of correctness.

## 7. Batch collection and sample decisions

- Counts start fresh every time the application launches. Existing saved files do not resume the previous session's counters.
- **Samples in Batch** sets the maximum captured sample count for that batch. Once samples are collected, its capacity cannot be changed.
- Captured and inspected counts differ: a captured sample may still be pending inspection.
- Re-inspecting the current sample updates its result without adding another captured sample.
- When the configured number of samples have saved inspection results, a completed-batch PDF is written automatically and the next batch is selected.
- A trailing batch number is incremented while retaining its numeric width, for example `LOT001` becomes `LOT002`. A batch without trailing digits receives a numbered suffix.
- The next batch retains the configured capacity and starts sample numbering again.
- A completed batch may contain samples awaiting an inspector decision; completion is based on saved inspection results. Recording the current sample's decision after rollover updates its saved record and completed PDF.

For a 100-sample batch, collect and inspect 100 distinct samples. The app then writes the completed summary and advances to the next batch. This application is an interactive capture/inspection tool; no PLC trigger, conveyor integration, or guaranteed line-rate throughput is configured in this version.

## 8. Reports and storage

### PDF-only output directory

New report exports go directly into the application's `outputs` directory as PDFs. Images, JSON, text files, SQLite databases, and session folders are stored separately.

- **Source:** `outputs` next to `app.py`.
- **Installed EXE:** `outputs` next to the installed `app.exe`, normally `%LOCALAPPDATA%\Programs\Inspectra AI\outputs`.
- **Packaged fallback:** if the EXE directory is not writable, `%LOCALAPPDATA%\InspectraAI\outputs`.
- **Internal data:** `%LOCALAPPDATA%\InspectraAI\data`.

Internal data includes captured sample images, session SQLite databases, per-inspection images, JSON metadata, and supporting text records. It enables collective reports and decision updates; do not remove it while the application is running. Resetting startup counters does not delete this archive. Uninstallation preserves reports and internal inspection data. Plan retention and backups separately for production use.

The development/distribution folder named `outputs` also holds deliverables such as `app.exe` and `installer.exe`. This is distinct from the runtime report folder created beside the installed app.

### Two-page batch summary

**Save Report** exports a two-page aggregate summary for the current batch; completed batches also export automatically. The PDF omits images and individual sample appendices.

- **Page 1:** planned/captured/inspected/pending counts, decision totals, warning/failure counts, defect region totals, inspected samples per class, mean regions per inspected sample, mean detected-region confidence, and processing-time averages.
- **Page 2:** average restoration/super-resolution metrics, valid measurement counts, processing coverage, average scale factors, and average increase in pixel count.

Averages use the latest saved result of each captured sample. A metric's **Valid N** counts available finite measurements; missing values are excluded. Infinite PSNR values are reported separately and excluded from finite averages. Region counts count connected regions produced by the model, which may not equal physical defect counts if a defect is split or nearby defects merge. The final summary focuses on averages rather than individual images or best/worst sample listings.

## 9. Rebuild the executable and installer

Only needed for application development. Users of `installer.exe` can skip this section.

### Executable

Install the build dependencies in the development environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
```

The supplied build configuration lives at `work\InspectraAI.spec`. It includes all checkpoints, dynamic model imports, required package data, and the icon. Its supporting hooks are in `work\packaging_hooks`. Keep these with the spec. Before building on another computer, edit the spec's `root` to the folder containing `app.py`/`models`, and `project` to the distribution workspace containing `work` and `assets`. The recorded paths are specific to the development computer.

From that distribution workspace:

```powershell
$env:INSPECTRA_TEST_DIR = '0'
$env:USE_TF = '0'
$env:USE_FLAX = '0'
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --distpath outputs --workpath work\exe_build work\InspectraAI.spec
```

The result is `outputs\app.exe`. The build uses `assets\inspectra-ai-named.ico`. Source changes require an EXE rebuild; editing `app.py` does not update an already compiled EXE.

### Inno Setup installer

Install Inno Setup 6 separately. Retain this relative layout:

```text
distribution/
    assets/inspectra-ai-named.ico
    outputs/app.exe
    installer/InspectraAI.iss
```

Open `installer\InspectraAI.iss` in Inno Setup and select **Compile**, or run:

```powershell
& "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe" .\installer\InspectraAI.iss
```

Adjust the compiler path if Inno Setup is installed elsewhere. The output is `outputs\installer.exe`. The script installs the bundled EXE and icon, creates a Start menu shortcut, offers a desktop shortcut, and creates an uninstaller. The standalone `installer.exe` is the file to distribute to end users. Keep the `.iss`, icon, and input EXE for future rebuilds. Compiler/package tools are not required on end-user computers.

### Built-in self-test

Use a dedicated test machine or development environment:

```powershell
.\.venv\Scripts\python.exe app.py --self-test "$PWD\source-self-test.json"
# Or test the compiled executable; wait for the process to exit:
Start-Process -FilePath .\outputs\app.exe -ArgumentList @('--self-test', ('"' + "$PWD\exe-self-test.json" + '"')) -Wait
```

The self-test loads and performs a small forward pass through all discovered supported checkpoints, checks BRISQUE and SSIM, runs a synthetic inspection, validates a two-page PDF, batch rollover, and PDF-only runtime output. It uses temporary test storage. The JSON log reports `passed` and any error. This is a functional smoke test; it does not measure accuracy on a labeled validation dataset or establish production throughput.

The latest installer was tested by installing to a separate folder, verifying the installed EXE hash, running this self-test with all **22** checkpoints, and uninstalling successfully. PDF generation, batch rollover, and PDF-only runtime output passed.

## 10. Troubleshooting

| Symptom | Action |
| --- | --- |
| `ModuleNotFoundError` | Install `requirements.txt` with the same Python executable used to launch the app. |
| No usable model / checkpoint mismatch | Restore the expected model files and names; confirm that the architecture matches the trained checkpoint. |
| Missing `cv2.quality` warning | Use the latest integrated `app.py` or rebuilt installer. Its BRISQUE implementation uses embedded data and does not require `cv2.quality`. |
| PSNR/SSIM/MSE/MAE are N/A | Load an aligned, same-size clean reference for the relevant output stage. |
| Automatic selection is slow | It runs every available candidate; choose one model or skip an enhancement stage if needed. Assess throughput using representative images on the deployment computer. |
| Camera cannot open | Check the connection, Windows camera permissions, and whether another application is using the camera. Upload Image remains available. |
| Inspection cannot start | Supply required IDs, a positive batch capacity, and an image; review any validation dialog. |
| No defect overlay | Confirm the desired defect types are selected and their legend filters are enabled. An image can also have no predicted regions. |
| Reports appear in an unexpected location | Check the runtime paths in section 8, including the writable fallback. |
| EXE takes time before the window appears | The single-file package extracts its runtime and models on startup. |
| PyTorch/Qt DLL import error in source mode | Use a clean Python 3.10 x64 environment with the pinned requirements; avoid mixing unrelated Qt/OpenCV distributions. |

For an inference failure, retain the displayed warning, the model filename, image dimensions, and the Python package versions when investigating. Confirm performance and decision criteria with representative labeled assembly-line data before relying on automatic ranking for quality decisions.
