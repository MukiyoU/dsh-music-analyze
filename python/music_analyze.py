#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
music_analyze.py — 把声音画成能看的图

由 dsh-music-analyze 插件调用，也可以单独用：

    python music_analyze.py <音频或视频> [--out 目录] [--start 秒] [--end 秒]
                            [--label 标题] [--json]

产物：
    music_report.png   四联图（波形 / 梅尔频谱 / 色度 / 段落能量）
    music_report.txt   文字指标（BPM、主调、频段能量比、最响最静）

为什么是这四个图：
    波形 = 节奏和强弱（眼睛能看出哪里有鼓、哪里安静）
    频谱 = 频率分布（人声、贝斯、镲片的频段位置）
    色度 = 12 个音高类别 → 调性和和弦走向（讨论音乐最关键）
    能量 = 段落结构（前奏/主歌/副歌的分界）
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _writable_cache_dir():
    """找一个能写的 matplotlib 缓存目录

    坑：被 DSH 宿主进程调用时，`tempfile.gettempdir()` 本身就可能抛
    `FileNotFoundError: No usable temporary directory`（沙箱下 %TEMP% 不可用），
    所以它必须放在 try 里，不能直接进候选列表。
    """
    candidates = []
    try:
        candidates.append(Path(tempfile.gettempdir()) / "dsh-music-analyze-mplcache")
    except Exception:
        pass
    candidates += [
        # DSH 自己的缓存区：被宿主进程调用时 %TEMP% 不可用，这个一定可写
        Path.home() / ".dsh" / "cache" / "dsh-music-analyze",
        Path.home() / ".cache" / "dsh-music-analyze",
        # 最后一档：插件目录（会污染安装目录，只在没得选时用）
        HERE / "_mplcache",
        Path.cwd() / ".mplcache",
    ]
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            probe = c / ".write_probe"
            probe.write_text("x", encoding="utf-8")
            probe.unlink()
            return c
        except Exception:
            continue
    return None


_cache = _writable_cache_dir()
if _cache is not None:
    os.environ.setdefault("MPLCONFIGDIR", str(_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（微软雅黑 / 黑体），找不到就退回英文标签
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def find_ffmpeg():
    """按 环境变量 → 脚本旁 bin/ → PATH → imageio_ffmpeg 的顺序找 ffmpeg"""
    env = os.environ.get("FFMPEG_BIN")
    if env and Path(env).exists():
        return env
    local = HERE / "bin" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if local.exists():
        return str(local)
    which = shutil.which("ffmpeg")
    if which:
        return which
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_audio(src: Path, dst: Path, start=None, end=None) -> tuple[bool, str]:
    """用 ffmpeg 统一转成 22050Hz 单声道 wav"""
    ff = find_ffmpeg()
    if not ff:
        return False, (
            "找不到 ffmpeg。装一个，或者设环境变量 FFMPEG_BIN 指向 ffmpeg.exe，"
            "或者 pip install imageio-ffmpeg。"
        )
    cmd = [ff, "-y"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-i", str(src), "-vn", "-ac", "1", "-ar", "22050", "-f", "wav", str(dst)]
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300, creationflags=flags)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg 提取音频超时（300 秒）"
    if r.returncode != 0 or not dst.exists():
        tail = r.stderr.decode("utf-8", "ignore")[-600:]
        return False, f"ffmpeg 提取音频失败：{tail}"
    return True, ""


def analyze(wav: Path, out_dir: Path, label: str = "") -> dict:
    import librosa
    import librosa.display

    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    dur = len(y) / sr
    if dur < 1.0:
        raise ValueError(f"音频太短（{dur:.2f} 秒），至少 1 秒")

    hop = 512
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])
    bpm_reliable = tempo >= 30
    # 无鼓的抒情独奏常被算成真实速度的两倍（八分音符被当拍点）——这不算「不可靠」，
    # 但要明确提示，否则模型会把 123 BPM 当事实描述一首慢曲子。
    if not bpm_reliable:
        bpm_note = "没有明显鼓点，节拍检测不可靠，别把这个数当事实"
    elif 100 <= tempo <= 180:
        bpm_note = f"检测到 {tempo:.0f} BPM；无鼓的抒情曲里这常是真实速度的两倍（约 {tempo / 2:.0f} BPM），引用时请留余地"
    else:
        bpm_note = ""

    S = np.abs(librosa.stft(y, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chroma_mean = chroma.mean(axis=1)
    key_idx = int(np.argmax(chroma_mean))
    key = NOTE_NAMES[key_idx]

    total = S.sum() + 1e-9

    def band(lo, hi):
        idx = (freqs >= lo) & (freqs < hi)
        return float(S[idx].sum() / total)

    bands = {
        "low": band(20, 250),
        "lowmid": band(250, 800),
        "midhigh": band(800, 3000),
        "high": band(3000, 8000),
        "air": band(8000, 20000),
    }
    band_labels = {
        "low": "低频 20-250Hz（贝斯/底鼓）",
        "lowmid": "中低 250-800Hz（人声基频）",
        "midhigh": "中高 800-3000Hz（人声清晰度/吉他）",
        "high": "高频 3000-8000Hz（镲片/齿音）",
        "air": "超高频 8000Hz+（空气感）",
    }

    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    seg_sec = 1.0
    nb = max(1, int(dur / seg_sec))
    seg_rms = np.array(
        [
            rms[(times >= i) & (times < i + seg_sec)].mean()
            if np.any((times >= i) & (times < i + seg_sec))
            else 0.0
            for i in range(nb)
        ]
    )
    dyn = float(20 * np.log10((seg_rms.max() + 1e-9) / (seg_rms.mean() + 1e-9)))

    loudest = [
        [round(float(i * seg_sec), 1), round(float(seg_rms[i]), 4)]
        for i in sorted(np.argsort(seg_rms)[::-1][:5])
    ]
    quietest = [
        [round(float(i * seg_sec), 1), round(float(seg_rms[i]), 4)]
        for i in sorted(np.argsort(seg_rms)[:3])
    ]

    # ---- 画图 ----
    tempo_label = f"{tempo:.0f} BPM" if bpm_reliable else "节拍不明显"
    title_src = f" · {label}" if label else ""
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), constrained_layout=True)
    fig.suptitle(f"音乐分析{title_src} · {dur:.1f}s · {tempo_label} · 主调 {key}", fontsize=15)

    t = np.linspace(0, dur, len(y))
    axes[0].plot(t, y, linewidth=0.3, color="#2b6cb0")
    axes[0].set_title("波形（音量包络：看节奏、强弱、段落）")
    axes[0].set_xlim(0, dur)
    axes[0].set_xlabel("秒")
    axes[0].set_ylabel("振幅")

    librosa.display.specshow(mel_db, sr=sr, hop_length=hop, x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
    axes[1].set_title("频谱（频率随时间分布：下方=低音，上方=高音）")
    axes[1].set_xlabel("秒")

    librosa.display.specshow(chroma, sr=sr, hop_length=hop, x_axis="time", y_axis="chroma", ax=axes[2], cmap="coolwarm")
    axes[2].set_title(f"色度图（12 音高类别，主调 {key}——看和弦走向）")

    axes[3].plot(np.arange(nb) * seg_sec, seg_rms, color="#dd6b20", linewidth=1.2)
    axes[3].fill_between(np.arange(nb) * seg_sec, seg_rms, alpha=0.25, color="#dd6b20")
    axes[3].set_title("段落能量（每秒 RMS：能看出前奏/主歌/副歌的分界）")
    axes[3].set_xlabel("秒")
    axes[3].set_ylabel("能量")
    axes[3].set_xlim(0, dur)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "music_report.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)

    # ---- 文字报告 ----
    lines = []
    lines.append(f"时长      : {dur:.1f} 秒")
    lines.append(f"速度      : {tempo_label}" + (f"（{tempo:.0f} BPM）" if bpm_reliable else ""))
    if bpm_note:
        lines.append(f"            {bpm_note}")
    lines.append(f"主调      : {key}（色度分析最强音级）")
    lines.append(f"节拍检测  : {len(beats)} 个拍点")
    lines.append(f"动态范围  : {dyn:.1f} dB（越大起伏越强）")
    lines.append("")
    lines.append("频段能量分布：")
    for k in bands:
        lines.append(f"  {band_labels[k]:28s} {bands[k]*100:5.1f}%")
    lines.append("")
    lines.append("音量最大的 5 个时刻（可能是副歌/高潮）：")
    for sec, val in loudest:
        lines.append(f"  {sec:6.1f}s   能量 {val:.4f}")
    lines.append("")
    lines.append("音量最小的 3 个时刻（可能是前奏/间奏/结尾）：")
    for sec, val in quietest:
        lines.append(f"  {sec:6.1f}s   能量 {val:.4f}")
    report = "\n".join(lines)

    txt = out_dir / "music_report.txt"
    txt.write_text(report + "\n", encoding="utf-8")

    return {
        "imagePath": str(png.resolve()),
        "reportPath": str(txt.resolve()),
        "report": report,
        "duration": round(float(dur), 2),
        "bpm": round(float(tempo), 1),
        "bpmReliable": bool(bpm_reliable),
        "bpmNote": bpm_note,
        "key": key,
        "bands": {k: round(float(v), 4) for k, v in bands.items()},
        "loudest": loudest,
        "quietest": quietest,
    }


def main():
    # Windows 控制台默认 GBK；插件用 UTF-8 读 stdout，不统一就会得到「ʱ��」这种乱码。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="把音频画成四联图并给出指标")
    ap.add_argument("src", help="音频或视频文件")
    ap.add_argument("--out", default=None, help="产物目录（默认：源文件同级 music/）")
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--label", default="", help="图标题里的名字")
    ap.add_argument("--json", action="store_true", help="最后输出一行 JSON")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    if not src.exists():
        msg = f"文件不存在：{src}"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}, ensure_ascii=False))
        else:
            print(f"[ERR] {msg}")
        return 1

    out_dir = Path(args.out).expanduser().resolve() if args.out else src.parent / "music"

    # 临时 wav 放产物目录下 —— 不依赖 %TEMP%：被 DSH 宿主进程调用时，
    # tempfile.TemporaryDirectory() 会因为「没有可用临时目录」直接抛异常。
    work = out_dir / "_tmp"
    work.mkdir(parents=True, exist_ok=True)
    wav = work / f"audio_{os.getpid()}.wav"
    try:
        ok, err = extract_audio(src, wav, args.start, args.end)
        if not ok:
            if args.json:
                print(json.dumps({"ok": False, "error": err}, ensure_ascii=False))
            else:
                print(f"[ERR] {err}")
            return 1

        try:
            data = analyze(wav, out_dir, args.label)
        except Exception as e:  # noqa: BLE001 — 要把任何分析失败转成一条清楚的错误
            if args.json:
                print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=False))
            else:
                print(f"[ERR] {type(e).__name__}: {e}")
            return 1
    finally:
        try:
            wav.unlink()
        except Exception:
            pass
        try:
            work.rmdir()
        except Exception:
            pass

    data["ok"] = True
    data["source"] = str(src)

    if args.json:
        print(json.dumps(data, ensure_ascii=False))
    else:
        print(data["report"])
        print()
        print(f"[OK] 图  : {data['imagePath']}")
        print(f"[OK] 报告: {data['reportPath']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
