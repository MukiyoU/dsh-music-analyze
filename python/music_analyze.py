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
import librosa

# 中文字体（微软雅黑 / 黑体），找不到就退回英文标签
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


# ---------------------------------------------------------------------------
# BPM：三个独立估计器 + 倍频裁决
#
# 为什么不能只信一个：
#   librosa 的 beat_track 在无鼓的抒情曲上会把八分音符当拍点，报出两倍速度。
#   三个方法交叉验证 + 一个「奇偶拍强度」判据，能把倍频错误挑出来。
#
# 为什么奇偶比要用 RMS 而不是 onset_strength：
#   onset_strength 对振幅不敏感（面板 1.0 vs 0.3 只差十几个百分点），
#   实测合成"强弱交替"信号时 onset 比只有 1.19，而 RMS 比能到 1.4 以上。
# ---------------------------------------------------------------------------


def _est_acf(onset_env, sr, hop, bpm_min=30.0, bpm_max=300.0):
    """onset 强度自相关：看周期性，最强峰对应的 lag 换算成 BPM"""
    x = onset_env - onset_env.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1 :]
    ac = ac / (ac[0] + 1e-9)
    lag_min = max(2, int(np.floor(60.0 * sr / (bpm_max * hop))))
    lag_max = min(len(ac) - 2, int(np.ceil(60.0 * sr / (bpm_min * hop))))
    if lag_max <= lag_min + 1:
        return None
    seg = ac[lag_min:lag_max]
    peaks = [(seg[i], lag_min + i) for i in range(1, len(seg) - 1)
             if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.02]
    if not peaks:
        return None
    peaks.sort(reverse=True)
    return float(60.0 * sr / (peaks[0][1] * hop))


def _est_tempogram(onset_env, sr, hop, bpm_min=30.0, bpm_max=300.0):
    """循环节拍图：频域看拍速的周期结构"""
    try:
        tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop)
    except Exception:
        return None
    mean_tg = tg.mean(axis=1)
    lags = np.arange(len(mean_tg))
    bpms = 60.0 * sr / (lags * hop + 1e-9)
    mask = (bpms >= bpm_min) & (bpms <= bpm_max)
    if not np.any(mask):
        return None
    idx = np.where(mask)[0]
    return float(bpms[idx[np.argmax(mean_tg[idx])]])


def _accent_ratio(strengths):
    """奇偶拍强度比（中位数，抗噪）。>1.35 视为有明显强弱交替"""
    if len(strengths) < 8:
        return 1.0, False
    a = np.median(strengths[0::2])
    b = np.median(strengths[1::2])
    hi, lo = max(a, b), min(a, b)
    mx = float(np.max(strengths))
    if mx <= 0 or hi < mx * 0.02:
        return 1.0, False
    return float(hi / (lo + 1e-9)), True


def _octave_factor(a, b, tol=0.06):
    """b 是 a 的 0.5 倍 / 1 倍 / 2 倍？（用于检查三个估计器是否互相印证）"""
    for k in (0.5, 1.0, 2.0):
        if abs(a - b * k) / max(a, b * k, 1e-9) <= tol:
            return k
    return None


def decide_bpm(y, sr, hop):
    """
    返回 (bpm, beats, info)。
    info 里带三个原始估计、奇偶强度比、候选和判定来源——让模型能看到「为什么是这个数」。
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo_dp, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    tempo_dp = float(np.atleast_1d(tempo_dp)[0])
    beats = np.asarray(beats, dtype=int)

    bpm_acf = _est_acf(onset_env, sr, hop)
    bpm_tg = _est_tempogram(onset_env, sr, hop)

    # 奇偶拍强度比：用 DP 吐出来的**真实拍点位置**取强度。
    # 自己按周期铺网格会丢相位，网格错开半拍时奇偶比永远是 1。
    ratio_onset, ratio_onset_ok = (1.0, False)
    ratio_rms, ratio_rms_ok = (1.0, False)
    if len(beats) >= 8:
        env = onset_env / (onset_env.max() + 1e-9)
        ratio_onset, ratio_onset_ok = _accent_ratio(env[np.clip(beats, 0, len(env) - 1)])
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        ratio_rms, ratio_rms_ok = _accent_ratio(rms[np.clip(beats, 0, len(rms) - 1)])

    acc_ratio = ratio_rms if ratio_rms_ok else ratio_onset
    acc_ok = ratio_rms_ok or ratio_onset_ok

    factors = {}
    for name, val in (("acf", bpm_acf), ("tempogram", bpm_tg)):
        if val:
            factors[name] = _octave_factor(tempo_dp, val)
    agree = [n for n, k in factors.items() if k == 1.0]

    bpm = tempo_dp
    source = "动态规划节拍跟踪"
    halved = False
    note = ""

    if acc_ok and acc_ratio >= 1.35 and tempo_dp / 2 >= 40:
        bpm = tempo_dp / 2
        source = "动态规划节拍跟踪 + 奇偶强度比判半速"
        halved = True
        note = f"奇偶拍强度比 {acc_ratio:.2f}，判定真实速度为一半（重音每两拍一次）"
    elif agree:
        source = f"三估计器一致（DP + {'/'.join(agree)}）"
        if not acc_ok:
            note = "奇偶强度比数据不足，按三估计器共识报"
    else:
        note = "三个估计器未完全一致，按 DP 报，置信度较低"

    candidates = []
    for label, mult in (("半速", 0.5), ("原值", 1.0), ("倍速", 2.0)):
        v = tempo_dp * mult
        if 30 <= v <= 300:
            candidates.append({"bpm": round(v, 1), "relation": label})

    info = {
        "raw": {
            "dp": round(tempo_dp, 2),
            "acf": round(bpm_acf, 2) if bpm_acf else None,
            "tempogram": round(bpm_tg, 2) if bpm_tg else None,
        },
        "accentRatio": round(acc_ratio, 3),
        "accentReliable": bool(acc_ok),
        "halved": halved,
        "source": source,
        "note": note,
        "candidates": candidates,
        "beatCount": int(len(beats)),
    }
    return bpm, beats, info


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
    # BPM 用三估计器 + 倍频裁决（不再是单算法的裸结果）
    tempo, beats, bpm_info = decide_bpm(y, sr, hop)
    bpm_reliable = tempo >= 30
    bpm_note = bpm_info["note"]

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
    lines.append(f"速度      : {tempo_label}")
    _src = bpm_info.get("source")
    if _src:
        lines.append(f"            判定依据：{_src}")
    if bpm_note:
        lines.append(f"            {bpm_note}")
    # 三个估计器的原始值 —— 速度有歧义时，这些数字比结论更有用
    _raw = bpm_info.get("raw", {}) or {}
    _parts = []
    for _k, _lab in (("dp", "动态规划"), ("acf", "自相关"), ("tempogram", "节拍图")):
        _v = _raw.get(_k)
        if _v:
            _parts.append(f"{_lab} {_v:.1f}")
    if _parts:
        lines.append("            三估计器：" + " / ".join(_parts))
    _cands = bpm_info.get("candidates", []) or []
    if _cands:
        lines.append("            候选：" + "，".join(f"{c['relation']} {c['bpm']:.1f}" for c in _cands))
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
        # 三估计器原始值 / 奇偶强度比 / 候选 / 判定来源——让模型看到「为什么是这个数」
        "bpmDetail": bpm_info,
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
