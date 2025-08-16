# frame_nth_exporter_multi_gpu.py
# Export every Nth frame with GPU parallelism across selected NVIDIA GPUs.
# GUI: choose input, output, N, chunks per GPU, and which GPUs to use.
# VRAM-safe mode spills early to RAM and retries on OOM with CPU.
#
# Requires: Python 3.10+, PySide6
# Tools: ffmpeg.exe, ffprobe.exe in PATH or next to the EXE
# Optional: NVIDIA drivers with nvidia-smi for GPU enumeration
#
# Build EXE:
#   pyinstaller --onefile --windowed --name FrameNthExporter frame_nth_exporter_multi_gpu.py

import os, sys, json, math, shutil, subprocess, threading, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QSpinBox, QCheckBox, QPlainTextEdit, QGroupBox,
    QRadioButton, QScrollArea, QWidget as QtWidget, QGridLayout
)
from PySide6.QtCore import Qt

def run(cmd, env=None):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    out, err = proc.communicate()
    return proc.returncode, out, err

def find_ffmpeg_like(binary):
    here = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    candidates = [
        here / f"{binary}.exe",
        Path(sys.executable).with_name(f"{binary}.exe"),
        Path(__file__).with_name(f"{binary}.exe"),
        shutil.which(binary)
    ]
    for c in candidates:
        if c and Path(c).exists():
            return str(c)
    return None

def ffprobe_stream_meta(ffprobe, inp):
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,avg_frame_rate,nb_frames",
        "-show_entries", "format=duration",
        "-of", "json", inp
    ]
    rc, out, err = run(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed: {err.strip()}")
    data = json.loads(out or "{}")
    stream = data.get("streams", [{}])[0] if data.get("streams") else {}
    codec = stream.get("codec_name", "")
    afr = stream.get("avg_frame_rate", "0/1")
    nb = stream.get("nb_frames", "0")
    dur = data.get("format", {}).get("duration", "0")
    try:
        num, den = afr.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 0.0
    except Exception:
        fps = 0.0
    try:
        total_frames = int(nb)
    except Exception:
        total_frames = 0
    try:
        duration = float(dur)
    except Exception:
        duration = 0.0
    return codec, fps, total_frames, duration

def nvdec_decoder_for(codec_name: str) -> str | None:
    m = {
        "h264": "h264_cuvid",
        "hevc": "hevc_cuvid",
        "h265": "hevc_cuvid",
        "vp9":  "vp9_cuvid",
        "av1":  "av1_cuvid"
    }
    return m.get((codec_name or "").lower())

def list_nvidia_gpus():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []  # Unknown, user can still run CPU or single GPU by typing device 0
    q = ["index","name","memory.total"]
    cmd = [exe, f"--query-gpu={','.join(q)}", "--format=csv,noheader,nounits"]
    rc, out, err = run(cmd)
    if rc != 0 or not out.strip():
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                idx = int(parts[0])
            except:
                continue
            name = parts[1]
            try:
                mem = int(parts[2])
            except:
                mem = None
            gpus.append({"index": idx, "name": name, "mem_total_mb": mem})
    return gpus

def build_select_filter(n: int, offset: int = 0, use_gpu: bool = True, vram_safe: bool = False) -> str:
    sel = f"select='not(mod(n+{offset}\\,{n}))'"
    if use_gpu:
        # Always download to system RAM before encode
        if vram_safe:
            # Keep GPU surfaces minimal, convert to nv12 then drop from GPU asap
            return f"{sel},hwdownload,format=nv12"
        else:
            return f"{sel},hwdownload,format=nv12"
    else:
        return sel

def ffmpeg_cmd(ffmpeg, inp, out_dir, base, n, use_gpu, decoder, gpu_index=None,
               start_time=None, end_time=None, start_number=None, jpeg=True,
               vram_safe=False):
    args = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-probesize", "64M", "-analyzeduration", "64M"]

    if use_gpu:
        # Tell ffmpeg which device to use
        if gpu_index is not None:
            args += ["-hwaccel", "cuda", "-hwaccel_device", str(gpu_index), "-hwaccel_output_format", "cuda"]
        else:
            args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        if decoder:
            args += ["-c:v", decoder]
            if gpu_index is not None:
                # cuvid accepts -gpu index via -extra_hw_frames or by setting -gpu
                args += ["-gpu", str(gpu_index)]
        # Keep in-flight frames small in VRAM safe mode
        if vram_safe:
            args += ["-extra_hw_frames", "2"]
        else:
            args += ["-extra_hw_frames", "8"]

    if start_time is not None:
        args += ["-ss", f"{start_time:.6f}"]
    if end_time is not None:
        args += ["-to", f"{end_time:.6f}"]

    args += ["-i", inp, "-map", "0:v:0", "-vsync", "vfr", "-threads", "0"]

    # Output
    if jpeg:
        out_pat = str(Path(out_dir) / f"{base}_[%04d].jpeg")
        args += ["-q:v", "1", "-pix_fmt", "yuvj444p", out_pat]
    else:
        out_pat = str(Path(out_dir) / f"{base}_[%04d].png")
        args += [out_pat]

    if start_number is not None:
        pre = args[:-1]
        pre += ["-start_number", str(start_number)]
        pre += [args[-1]]
        args = pre

    return args

OOM_PATTERNS = [
    "out of memory",
    "Cannot allocate memory",
    "CUDA_ERROR_OUT_OF_MEMORY",
    "failed to map segment",
    "device memory allocation"
]

def looks_like_oom(stderr: str) -> bool:
    s = (stderr or "").lower()
    return any(pat in s for pat in [p.lower() for p in OOM_PATTERNS])

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FrameNth Exporter Multi-GPU")
        self.ffmpeg = find_ffmpeg_like("ffmpeg")
        self.ffprobe = find_ffmpeg_like("ffprobe")

        lay = QVBoxLayout(self)

        # Input
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Input video:"))
        self.inp = QLineEdit()
        btn_in = QPushButton("Browse")
        btn_in.clicked.connect(self.pick_input)
        row1.addWidget(self.inp, 1)
        row1.addWidget(btn_in)
        lay.addLayout(row1)

        # Output
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Output folder:"))
        self.outp = QLineEdit()
        btn_out = QPushButton("Browse")
        btn_out.clicked.connect(self.pick_output)
        row2.addWidget(self.outp, 1)
        row2.addWidget(btn_out)
        lay.addLayout(row2)

        # Options
        opt_box = QGroupBox("Options")
        opt = QGridLayout()

        opt.addWidget(QLabel("Every Nth:"), 0, 0)
        self.nspin = QSpinBox(); self.nspin.setRange(1, 100000); self.nspin.setValue(10)
        opt.addWidget(self.nspin, 0, 1)

        opt.addWidget(QLabel("Chunks per GPU:"), 0, 2)
        self.chunks_per_gpu = QSpinBox(); self.chunks_per_gpu.setRange(1, 64); self.chunks_per_gpu.setValue(2)
        opt.addWidget(self.chunks_per_gpu, 0, 3)

        self.vram_safe = QCheckBox("VRAM-safe mode spill to RAM")
        self.vram_safe.setChecked(True)
        opt.addWidget(self.vram_safe, 1, 0, 1, 2)

        self.gpu_decode = QCheckBox("Use GPU decode")
        self.gpu_decode.setChecked(True)
        opt.addWidget(self.gpu_decode, 1, 2, 1, 2)

        self.jpg_radio = QRadioButton("JPEG 4:4:4 q=1")
        self.png_radio = QRadioButton("PNG lossless")
        self.jpg_radio.setChecked(True)
        opt.addWidget(self.jpg_radio, 2, 0)
        opt.addWidget(self.png_radio, 2, 1)

        opt_box.setLayout(opt)
        lay.addWidget(opt_box)

        # GPU list
        self.gpu_box = QGroupBox("Select GPUs")
        v = QVBoxLayout()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.gpu_list_widget = QtWidget()
        self.gpu_layout = QVBoxLayout(self.gpu_list_widget)
        self.scroll.setWidget(self.gpu_list_widget)
        v.addWidget(self.scroll)
        self.gpu_box.setLayout(v)
        lay.addWidget(self.gpu_box, 1)

        # Buttons
        runrow = QHBoxLayout()
        self.refresh = QPushButton("Refresh GPUs")
        self.refresh.clicked.connect(self.populate_gpus)
        self.runbtn = QPushButton("Export")
        self.runbtn.clicked.connect(self.run_export)
        runrow.addWidget(self.refresh)
        runrow.addStretch(1)
        runrow.addWidget(self.runbtn)
        lay.addLayout(runrow)

        # Log
        self.log = QPlainTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log, 2)

        self.resize(980, 680)
        self.populate_gpus()

    def populate_gpus(self):
        # Clear
        while self.gpu_layout.count():
            item = self.gpu_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.gpu_checks = []
        gpus = list_nvidia_gpus()
        if not gpus:
            lbl = QLabel("No NVIDIA devices found with nvidia-smi. You can still run CPU or single GPU by selecting device 0 implicitly.")
            self.gpu_layout.addWidget(lbl)
        else:
            for g in gpus:
                cb = QCheckBox(f"GPU {g['index']}: {g['name']}  {g['mem_total_mb']} MB")
                cb.setChecked(True)
                cb.gpu_index = g['index']
                self.gpu_checks.append(cb)
                self.gpu_layout.addWidget(cb)
        self.gpu_layout.addStretch(1)

    def pick_input(self):
        f, _ = QFileDialog.getOpenFileName(self, "Choose input video", "", "Video Files (*.mp4 *.mov *.mkv *.mxf *.avi *.mpg *.m4v *.webm);;All Files (*.*)")
        if f:
            self.inp.setText(f)
            outdir = str(Path(f).with_suffix("").parent)
            self.outp.setText(outdir)

    def pick_output(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self.outp.setText(d)

    def logln(self, s):
        self.log.appendPlainText(s)
        self.log.ensureCursorVisible()
        QApplication.processEvents()

    def run_export(self):
        inp = self.inp.text().strip().strip('"')
        outd = self.outp.text().strip().strip('"')
        n = int(self.nspin.value())
        c_per_gpu = int(self.chunks_per_gpu.value())
        vram_safe = self.vram_safe.isChecked()
        want_gpu = self.gpu_decode.isChecked()
        jpeg = self.jpg_radio.isChecked()

        if not self.ffmpeg or not self.ffprobe:
            self.logln("ERROR: ffmpeg/ffprobe not found. Place ffmpeg.exe and ffprobe.exe next to this app or in PATH.")
            return
        if not inp or not Path(inp).exists():
            self.logln("ERROR: Input video not found.")
            return
        if not outd:
            self.logln("ERROR: Choose an output folder.")
            return
        Path(outd).mkdir(parents=True, exist_ok=True)
        base = Path(inp).stem

        try:
            codec, fps, total_frames, duration = ffprobe_stream_meta(self.ffprobe, inp)
        except Exception as e:
            self.logln(f"ffprobe error: {e}")
            return

        # Selected GPUs
        gpu_indices = [cb.gpu_index for cb in getattr(self, "gpu_checks", []) if cb.isChecked() and hasattr(cb, "gpu_index")]
        if want_gpu and not gpu_indices:
            # Assume at least GPU 0 exists if user insists on GPU
            gpu_indices = [0]

        dec = nvdec_decoder_for(codec) if want_gpu else None
        self.logln(f"Codec {codec} | fps {fps:.3f} | frames {total_frames} | dur {duration:.3f}s | NVDEC {bool(dec)} | GPUs {gpu_indices if want_gpu else 'CPU'}")

        # If only 1 chunk in total, run single pass for perfect VFR handling
        total_chunks = max(1, (len(gpu_indices) if want_gpu else 1) * max(1, c_per_gpu))
        if total_chunks == 1 or fps <= 0.1 or duration <= 0.1:
            filt = build_select_filter(n=n, offset=0, use_gpu=want_gpu and bool(dec), vram_safe=vram_safe)
            cmd = ffmpeg_cmd(self.ffmpeg, inp, outd, base, n, want_gpu and bool(dec), dec,
                             gpu_index=(gpu_indices[0] if want_gpu and gpu_indices else None),
                             jpeg=jpeg, vram_safe=vram_safe)
            insert_at = cmd.index("-vsync")
            cmd = cmd[:insert_at] + ["-vf", filt] + cmd[insert_at:]
            self.logln("Single pass command")
            self.logln(" ".join(f'"{c}"' if " " in c and not c.startswith("-") else c for c in cmd))
            rc, out, err = run(cmd)
            if rc != 0:
                if looks_like_oom(err) and want_gpu:
                    self.logln("OOM detected. Retrying with CPU decode on same range.")
                    cmd_cpu = ffmpeg_cmd(self.ffmpeg, inp, outd, base, n, False, None, jpeg=jpeg)
                    insert_at = cmd_cpu.index("-vsync")
                    cmd_cpu = cmd_cpu[:insert_at] + ["-vf", build_select_filter(n, 0, use_gpu=False)] + cmd_cpu[insert_at:]
                    self.logln(" ".join(cmd_cpu))
                    rc, out, err = run(cmd_cpu)
            self.logln("Done." if rc == 0 else f"Failed: {err.strip()}")
            return

        # Build chunk timeline
        step = duration / total_chunks
        chunks = []
        for i in range(total_chunks):
            start = max(0.0, i * step)
            end = duration if i == total_chunks - 1 else (i + 1) * step
            chunks.append((start, end))

        # Assign chunks round-robin to GPUs
        assignments = []
        for i, (start, end) in enumerate(chunks):
            gpu = gpu_indices[i % len(gpu_indices)] if want_gpu else None
            assignments.append((i, start, end, gpu))

        # Launch
        ok = True
        futures = []
        with ThreadPoolExecutor(max_workers=total_chunks) as ex:
            for i, start, end, gpu in assignments:
                # Estimate frame index to set select offset and global numbering
                start_frame_idx = int(round(start * fps))
                offset = start_frame_idx % n
                start_num = (start_frame_idx // n) + 1

                use_gpu = want_gpu and bool(dec)
                filt = build_select_filter(n=n, offset=offset, use_gpu=use_gpu, vram_safe=vram_safe)
                cmd = ffmpeg_cmd(self.ffmpeg, inp, outd, base, n, use_gpu, dec,
                                 gpu_index=gpu, start_time=start, end_time=end,
                                 start_number=start_num, jpeg=jpeg, vram_safe=vram_safe)
                insert_at = cmd.index("-vsync")
                cmd = cmd[:insert_at] + ["-vf", filt] + cmd[insert_at:]

                self.logln(f"[Chunk {i+1}/{total_chunks}] {start:.3f}s to {end:.3f}s on GPU {gpu if use_gpu else 'CPU'} start_num={start_num} offset={offset}")
                self.logln(" ".join(f'"{c}"' if " " in c and not c.startswith("-") else c for c in cmd))

                futures.append(ex.submit(self._run_with_retry, cmd, use_gpu))

            for fut in as_completed(futures):
                rc, err = fut.result()
                if rc != 0:
                    ok = False
                    self.logln(f"Chunk failed: {err.strip()}")
                else:
                    self.logln("Chunk done.")

        self.logln("All chunks complete." if ok else "Finished with errors. For VFR sources, reduce chunks or set Chunks per GPU = 1.")

    def _run_with_retry(self, cmd, used_gpu):
        rc, out, err = run(cmd)
        if rc != 0 and used_gpu and looks_like_oom(err):
            # Retry on CPU for this chunk
            self.logln("OOM on GPU. Retrying chunk with CPU.")
            # Rebuild CPU command
            cmd_cpu = list(cmd)
            # Remove hwaccel flags and cuvid specifics
            def remove_arg(a, val_next=False):
                while a in cmd_cpu:
                    idx = cmd_cpu.index(a)
                    del cmd_cpu[idx]
                    if val_next and idx < len(cmd_cpu):
                        del cmd_cpu[idx]
            for flag in ["-hwaccel","-hwaccel_device","-hwaccel_output_format","-extra_hw_frames","-gpu","-c:v"]:
                remove_arg(flag, val_next=True)
            # Remove any cuvid decoder presence
            # Nothing else to change. Keep timing, numbering, filter, outputs identical.
            rc2, out2, err2 = run(cmd_cpu)
            return rc2, err2
        return rc, err
        

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = App()
    w.show()
    sys.exit(app.exec())
