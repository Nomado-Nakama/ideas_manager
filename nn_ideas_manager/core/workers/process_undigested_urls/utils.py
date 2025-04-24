"""
End-to-end ingestion utilities for Instagram Reels / Posts.

• Video download                → yt-dlp
• Images / audio extraction     → ffmpeg / OpenCV
• Text extraction               → faster-whisper + pytesseract
• Metadata pulling              → instaloader (public fields only)
• Embedding & storage           → LangChain + Chroma (bge-small-en, cosine/MRR)

All heavyweight models are initialised once at import
and shared across coroutine calls.
"""

import os
import orjson
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import cv2
import loguru
import yt_dlp
import instaloader

import pytesseract
from torch.cuda import is_available
from langchain_chroma import Chroma
from faster_whisper import WhisperModel
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------------------------------------------------------
# CONSTANTS & ONE-TIME INITIALISATION
# -----------------------------------------------------------------------------

# Let Python discover the full path of `tesseract.exe`
pytesseract.pytesseract.tesseract_cmd = (
    shutil.which("tesseract")          # returns full path if it’s on PATH
    or r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # fallback
)

_META_SUFFIX_JSON = ".meta.json"

_DL_DIR = Path("/ingestion/downloaded_content")
_DL_DIR.mkdir(parents=True, exist_ok=True)
_CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma")).resolve()
_CHROMA_DIR.mkdir(exist_ok=True, parents=True)

# multilingual embeddings
_EMBED_MODEL_NAME = "intfloat/multilingual-e5-base"
_EMBED_DEVICE = "cuda" if is_available() else "cpu"

loguru.logger.info(f"DEVICE: {_EMBED_DEVICE}")

_COMPUTE_TYPE = "float16" if _EMBED_DEVICE == "cuda" else "default"

loguru.logger.info(f"_COMPUTE_TYPE: {_COMPUTE_TYPE}")

_embeddings = HuggingFaceEmbeddings(
    model_name=_EMBED_MODEL_NAME,
    model_kwargs={"device": _EMBED_DEVICE},
    encode_kwargs={"normalize_embeddings": True}
)
_vectorstore = Chroma(persist_directory=str(_CHROMA_DIR), embedding_function=_embeddings)

# singleton whisper model
_whisper = WhisperModel(model_size_or_path="large-v3", device=_EMBED_DEVICE, compute_type=_COMPUTE_TYPE)

# instagram public metadata loader (no login needed for public reels/posts)
_instaloader = instaloader.Instaloader(download_pictures=False,
                                       download_videos=False,
                                       download_comments=False,
                                       save_metadata=False,
                                       quiet=True)


@dataclass(slots=True)
class IngestResult:
    url: str
    doc_ids: List[str]  # vector-store ids
    merged_text: str


# --------------------------------------------------------------------------
# LOW-LEVEL HELPERS
# --------------------------------------------------------------------------
def _dump_info(info: dict, out_dir: Path) -> None:
    """Write both JSON and pickle blobs next to the downloaded media."""
    vid_id = info["id"]
    (out_dir / f"{vid_id}{_META_SUFFIX_JSON}").write_bytes(
        # ⬇️  stringify anything orjson can’t handle natively
        orjson.dumps(
            info,
            option=orjson.OPT_NAIVE_UTC | orjson.OPT_INDENT_2,
            default=lambda o: None,
        )
    )


def _get_escaped_word_timestamps(words: list[dict], detected_language):
    return [
        {
            "token": w['token'],
            "language": detected_language,
            "start": float(round(w['start'], 2)),
            "end": float(round(w['end'], 2)),
        }
        for w in words
    ]


def _extract_content_type_and_content_id_from_url(link: str) -> tuple[str, str]:
    """
    Parse Instagram URLs to get content type and ID.
    """
    if "reel/" in link:
        return "reel", link.split("reel/")[1].split("/")[0]
    if "p/" in link:
        return "post", link.split("p/")[1].split("/")[0]
    raise ValueError(f"Can't parse content type and content id from: {link}")


def _load_cached_info(url: str, out_dir: Path) -> dict | None:
    """
    Try to recover a previously cached info-dict.
    Accepts either a full Instagram URL or a bare shortcode/id.
    """
    # Heuristic: last path fragment before optional '?' is the shortcode.
    _, vid_id = _extract_content_type_and_content_id_from_url(url)
    json_f = out_dir / f"{vid_id}{_META_SUFFIX_JSON}"

    if json_f.exists():
        return orjson.loads(json_f.read_bytes())
    return None


def _run(cmd: list[str]) -> None:
    """Run shell command and raise if it fails."""
    subprocess.run(cmd, check=True, capture_output=True)


def _download_media(url: str, out_dir: Path = _DL_DIR) -> tuple[Path, dict]:
    """
    Download a single Instagram reel/post and return the **merged** MP4 path.

    – Uses yt-dlp’s download-archive to avoid a network hit if we already have it.
    – Creates `out_dir` if missing.
    – Always returns the final path (raise on failure).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    cached = _load_cached_info(url, out_dir)
    if cached:
        final = out_dir / f"{cached['id']}.mp4"

        if final.exists():  # media already on disk
            return final, cached

        final = out_dir / f"{cached['id']}.jpg"

        if final.exists():  # media already on disk
            return final, cached

    # media file missing → fall through and re-download only the file
    ydl_opts = {
        # ----- no console noise -----
        "quiet": True,
        "no_warnings": True,

        # ----- skip if already downloaded -----
        "download_archive": str(out_dir / "downloaded.txt"),

        # ----- output -----
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),  # real name after probe
        "merge_output_format": "mp4",
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",

        # ----- misc -----
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        _dump_info(info, out_dir)
        final_path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")

        if not final_path.exists():
            raise FileNotFoundError(f"yt-dlp reported success but {final_path} is missing")

        return final_path, info


def _extract_audio(video_path: Path) -> Path:
    wav = video_path.with_suffix(".wav")
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
          "-i", str(video_path), "-ar", "16000", "-ac", "1", str(wav)])
    return wav


def _load_cached_whisper_data(words_path: Path) -> tuple[str, list[dict]]:
    """
    Recover `(full_text, words)` from a previously written *.words.json* file.
    """
    words: list[dict] = orjson.loads(words_path.read_bytes())
    # naïve token-join is usually good enough for our purposes
    text = "".join(w["token"] for w in words).strip()
    return text, words


def _whisper_transcribe(media_path: Path, wav: Path, *, chunk_sec: int | None = None) -> tuple[str, list[dict]]:
    """
    Robust speech-to-text.

    • If ``chunk_sec`` is given → use that window size.
    • If it is *None* → let faster-whisper use its own default
      (≈ 30 s for the distil / small checkpoints).
    • If the first run looks like Russian but wasn’t classified
      as “ru”, redo with an explicit language hint.
    """

    if os.path.exists(media_path.with_suffix(".words.json")):
        return _load_cached_whisper_data(media_path.with_suffix(".words.json"))

    # Build the kwargs so we pass *nothing* when we want the default
    kw: dict[str, int] = {}
    if chunk_sec:
        kw["chunk_length"] = max(chunk_sec, 1)   # must be ≥ 1

    def _run(lang: str | None = None) -> tuple[str, str, list[dict]]:
        seg_iter, info = _whisper.transcribe(
            str(wav),
            language=lang,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
            no_speech_threshold=0.0,
            compression_ratio_threshold=float("inf"),
            word_timestamps=True,
            **kw,
        )
        segments = list(seg_iter)
        text = " ".join(s.text.strip() for s in segments)

        # merge word lists from all segments (may be empty)
        words: list[dict] = []

        for s in segments:
            # s.words -> List[Word] (token:str, start:float, end:float)
            words.extend(
                {"token": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                for w in getattr(s, "words", []) or []
            )

        return text, info.language, words

    text, detected_language, words = _run()

    # --- heuristic russian re-run -------------------------------------
    if detected_language != "ru":
        cyr = sum("а" <= ch.lower() <= "я" for ch in text)
        if cyr / max(len(text), 1) > 0.40:
            text, detected_language, words = _run("ru")

    escaped_word_timestamps = _get_escaped_word_timestamps(words, detected_language)
    (
        media_path.with_suffix(".words.json")
        .write_bytes(orjson.dumps(escaped_word_timestamps, option=orjson.OPT_INDENT_2))
    )

    return text, escaped_word_timestamps


def _get_spoken_words(words, t_sec, word_margin: float = 0.4):
    spoken = ""
    if words:
        spoken_tokens = [
            w["token"]
            for w in words
            if (w["start"] - word_margin) <= t_sec <= (w["end"] + word_margin)
        ]
        spoken = "".join(spoken_tokens)
    return spoken


def _ocr_video_frames(
    video_path: Path,
    every_ms: int = 800,
    words: list[dict] | None = None,
) -> str:
    """
    Scan frames, OCR any visible text **and** align it with spoken words.

    Returns a multi-line string like:
        [1.60 s]  OCR: "Bibliothèque Richelieu" | AUDIO: Bibliothèque
        [8.00 s]  OCR: "Galerie Vivienne"       | AUDIO: Galerie Vivienne
    """

    detected_language = words[0]['language']
    detected_language = detected_language + "g" if detected_language == "en" else detected_language + "s"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(fps * every_ms / 1000)

    collected: list[str] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            # current position in seconds (OpenCV keeps this updated)
            t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ocr_txt = pytesseract.image_to_string(gray, lang=detected_language).strip()

            spoken = _get_spoken_words(words, t_sec)

            line = ""
            if ocr_txt:
                line += f"[{t_sec:05.2f}s] Text on video frame: \"{ocr_txt}\"\n"
            if spoken:
                line += f"[{t_sec:05.2f}s] Spoken audio in that moment: {spoken}"
            collected.append(line)

        frame_idx += 1

    cap.release()
    return "\n".join(collected)


def _ocr_image(img_path: Path) -> str:
    img = cv2.imread(str(img_path))
    return pytesseract.image_to_string(img, lang="eng+rus")


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def _embed_and_store(chunks: Iterable[str], url: str) -> List[str]:
    docs = [Document(page_content=ch, metadata={"source_url": url}) for ch in chunks]
    ids = _vectorstore.add_documents(docs)

    return ids


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------
async def ingestion_workflow(url: str) -> IngestResult:
    """
    Full coroutine performing ingestion for a single URL.

    Returns IngestResult and raises on irrecoverable errors.
    """

    # 1. Download
    media_path, meta_data = _download_media(url, _DL_DIR)

    # 2.  Build corpus
    merged_parts: list[str] = [orjson.dumps(meta_data, option=orjson.OPT_INDENT_2).decode()]
    if media_path.suffix == ".mp4":
        audio = _extract_audio(media_path)

        text, words = _whisper_transcribe(media_path, audio)

        merged_parts.append(text)  # speech-to-text
        merged_parts.append(_ocr_video_frames(media_path, words=words))  # frame OCR
    else:
        merged_parts.append(_ocr_image(media_path))  # image OCR

    merged_text = "\n".join(p for p in merged_parts if p)

    # 3.  Chunk ➜ embed ➜ store
    chunks = _chunk_text(merged_text)
    doc_ids = _embed_and_store(chunks, url)

    return IngestResult(url=url, doc_ids=doc_ids, merged_text=merged_text)
