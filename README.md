# FrameNth Exporter (Multi-GPU)

Export every Nth frame from a video at max speed with GPU decode and smart parallelism. Perfect for prepping drone and 360 footage for photogrammetry or Gaussian splatting. Keeps original dimensions. Filenames like `myclip_[0001].jpeg`.

## Why

* Drone and 360 rigs produce huge files.
* Photogrammetry and splat trainers often need sparse, evenly sampled frames.
* This app slices timelines across multiple GPUs, spills to RAM if VRAM gets tight.

## Features

* Multi-GPU NVDEC with per-device checkboxes
* Parallel chunking across selected GPUs
* VRAM-safe spill to RAM with auto CPU fallback on OOM
* JPEG 4:4:4 q=1 by default. PNG option for true lossless
* Constant naming: `[basename]_[####].jpeg`
* Original resolution preserved

## Ideal for

* Agisoft Metashape, RealityCapture, COLMAP
* Gaussian splats pipelines (gsplat, SIBR, PostShot, custom trainers)
* Aerial and 360 capture workflows

## Quick start

1. Download the release ZIP that includes:

   * `FrameNthExporter.exe`
   * `ffmpeg.exe`
   * `ffprobe.exe`
     Unzip and run `FrameNthExporter.exe`.

2. Select:

   * **Input video**
   * **Output folder**
   * **Every Nth** frame
   * **GPUs** to use
   * Optional **VRAM-safe mode**
   * Choose **JPEG** or **PNG**

3. Click **Export**. Files are written as `yourfile_[0001].jpeg` etc.

## If you need to install FFmpeg yourself

Windows CMD only (elevated):

```cmd
winget install -e --id FFmpeg.FFmpeg --accept-source-agreements --accept-package-agreements
ffmpeg -version
ffprobe -version
```

If `ffmpeg` is not found, add its bin to PATH for this session:

```cmd
set "PATH=C:\Program Files\ffmpeg\bin;%PATH%"
```

Chocolatey alternative:

```cmd
powershell -NoProfile -ExecutionPolicy Bypass -Command "Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol=3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"
choco install -y ffmpeg
```

## Build from source

```bash
python -m pip install PySide6 pyinstaller
pyinstaller --onefile --windowed --name FrameNthExporter frame_nth_exporter_multi_gpu.py
```

Copy `ffmpeg.exe` and `ffprobe.exe` next to the built EXE.

## Notes for image formats

* JPEG has no true zero compression. This app uses q=1 and 4:4:4 for minimal loss.
* Choose PNG for lossless if your pipeline demands it.

## Performance tips

* Start with 2 chunks per GPU. Increase if disks are fast NVMe.
* For variable frame rate sources, use 1 chunk per GPU for mathematically exact selection.
* Output to a fast local drive. Network shares bottleneck quickly.

## Troubleshooting

* OOM or stutters: enable VRAM-safe mode. The app reduces in-flight frames and spills to RAM. It will auto retry chunks on CPU if needed.
* No GPU listed: update NVIDIA driver. The app falls back to CPU if NVDEC is unavailable.
* Wrong file numbering when slicing tiny clips: reduce chunks.

## License

MIT. Use at your own risk.
