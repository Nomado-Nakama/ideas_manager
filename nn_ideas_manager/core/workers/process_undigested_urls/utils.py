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

import json
import os
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List

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
_whisper = WhisperModel(model_size_or_path="distil-large-v3", device=_EMBED_DEVICE, compute_type=_COMPUTE_TYPE)

# instagram public metadata loader (no login needed for public reels/posts)
_instaloader = instaloader.Instaloader(download_pictures=False,
                                       download_videos=False,
                                       download_comments=False,
                                       save_metadata=False,
                                       quiet=True)


@dataclass(slots=True)
class IngestResult:
    url: str
    doc_ids: List[str]          # vector-store ids
    merged_text: str


# -----------------------------------------------------------------------------
# LOW-LEVEL HELPERS
# -----------------------------------------------------------------------------
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

    ydl_opts = {
        # ----- no console noise -----
        "quiet": True,
        "no_warnings": True,

        # ----- skip if already downloaded -----
        "download_archive": str(out_dir / "downloaded.txt"),

        # ----- output -----
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),      # real name after probe
        "merge_output_format": "mp4",
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",

        # ----- misc -----
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        final_path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")

        if not final_path.exists():
            raise FileNotFoundError(f"yt-dlp reported success but {final_path} is missing")

        return final_path, info


def _extract_audio(video_path: Path) -> Path:
    wav = video_path.with_suffix(".wav")
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
          "-i", str(video_path), "-ar", "16000", "-ac", "1", str(wav)])
    return wav


def _whisper_transcribe(wav: Path) -> str:
    # first pass – auto-language
    segs, info = _whisper.transcribe(str(wav), beam_size=5)
    txt = " ".join(s.text.strip() for s in segs)

    if info.language == "ru":
        return txt

    cyrillic_ratio = sum("а" <= ch.lower() <= "я" for ch in txt) / max(len(txt), 1)

    if cyrillic_ratio > .4:  # likely Russian, redo with hint
        segs, _ = _whisper.transcribe(str(wav), language="ru", beam_size=5)
        txt = " ".join(s.text.strip() for s in segs)

    return txt


def _ocr_video_frames(video_path: Path, every_ms: int = 800) -> str:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(fps * every_ms / 1000)

    collected: list[str] = []
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            txt = pytesseract.image_to_string(gray, lang="eng+rus")
            if txt.strip():
                collected.append(txt)
        i += 1
    cap.release()
    return "\n".join(collected)


def _ocr_image(img_path: Path) -> str:
    img = cv2.imread(str(img_path))
    return pytesseract.image_to_string(img, lang="eng+rus")


def _pull_ig_metadata(url: str) -> str:
    """
    Fetch public caption / hashtags / owner from Instagram without login.
    Falls back to empty string if private.
    """
    try:
        shortcode = re.search(r"/([A-Za-z0-9_-]{11})/?$", url).group(1)
        post = instaloader.Post.from_shortcode(_instaloader.context, shortcode)
        meta = {
            "owner": post.owner_username,
            "caption": post.caption,
            "hashtags": list(post.caption_hashtags),
            "taken_at": post.date_utc.isoformat(),
            "likes": post.likes,
            "comments": post.comments,
        }
        return json.dumps(meta, ensure_ascii=False)
    except Exception:
        return ""


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

    # 2. Depending on media type
    merged_parts: list[str] = [_pull_ig_metadata(url)]
    if media_path.suffix == ".mp4":
        # (a) speech-to-text
        audio = _extract_audio(media_path)
        merged_parts.append(_whisper_transcribe(audio))
        # (b) frame OCR
        merged_parts.append(_ocr_video_frames(media_path))
    else:
        # jpeg / png (post)
        merged_parts.append(_ocr_image(media_path))

    merged_text = "\n".join(p for p in merged_parts if p)

    # 3. Chunk → embed → upsert
    chunks = _chunk_text(merged_text)
    doc_ids = _embed_and_store(chunks, url)

    return IngestResult(url=url, doc_ids=doc_ids, merged_text=merged_text)
