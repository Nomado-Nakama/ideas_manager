import os
import orjson
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List

import cv2
import loguru
import yt_dlp

import pytesseract
from dotenv import load_dotenv
from torch.cuda import is_available
from faster_whisper import WhisperModel
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nn_ideas_manager.core import _vectorstore

# -----------------------------------------------------------------------------
# CONSTANTS & ONE-TIME INITIALISATION
# -----------------------------------------------------------------------------
load_dotenv(r'E:\Projects\python\nakama-ideas-manager\configs\.env')

pytesseract.pytesseract.tesseract_cmd = (
        shutil.which("tesseract")
        or r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
)

_META_SUFFIX_JSON = ".meta.json"

_DL_DIR = Path("/ingestion/downloaded_content")
_DL_DIR.mkdir(parents=True, exist_ok=True)

_EMBED_DEVICE = "cuda" if is_available() else "cpu"
loguru.logger.info(f"DEVICE: {_EMBED_DEVICE}")

_COMPUTE_TYPE = "float16" if _EMBED_DEVICE == "cuda" else "default"
loguru.logger.info(f"_COMPUTE_TYPE: {_COMPUTE_TYPE}")

_whisper = WhisperModel(model_size_or_path="large-v3", device=_EMBED_DEVICE, compute_type=_COMPUTE_TYPE)


@dataclass(slots=True)
class IngestResult:
    url: str
    doc_ids: List[str]
    merged_text: str


def _dump_info(info: dict, out_dir: Path) -> None:
    vid_id = info["id"]
    (out_dir / f"{vid_id}{_META_SUFFIX_JSON}").write_bytes(
        orjson.dumps(info, option=orjson.OPT_NAIVE_UTC | orjson.OPT_INDENT_2, default=lambda o: None)
    )


def _get_escaped_word_timestamps(words: list[dict], detected_language):
    return [
        {"token": w['token'], "language": detected_language, "start": float(round(w['start'], 2)),
         "end": float(round(w['end'], 2))}
        for w in words
    ]


def _extract_content_type_and_content_id_from_url(link: str) -> tuple[str, str]:
    if "reel/" in link:
        return "reel", link.split("reel/")[1].split("/")[0]
    if "p/" in link:
        return "post", link.split("p/")[1].split("/")[0]
    raise ValueError(f"Can't parse content type and content id from: {link}")


def _load_cached_info(url: str, out_dir: Path) -> dict | None:
    _, vid_id = _extract_content_type_and_content_id_from_url(url)
    json_f = out_dir / f"{vid_id}{_META_SUFFIX_JSON}"
    if json_f.exists():
        return orjson.loads(json_f.read_bytes())
    return None


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _download_media(url: str, out_dir: Path = _DL_DIR) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = _load_cached_info(url, out_dir)
    if cached:
        for ext in [".mp4", ".jpg"]:
            final = out_dir / f"{cached['id']}{ext}"
            if final.exists():
                return final, cached

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "download_archive": str(out_dir / "downloaded.txt"),
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
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
    words = orjson.loads(words_path.read_bytes())
    text = "".join(w["token"] for w in words).strip()
    return text, words


def _whisper_transcribe(media_path: Path, wav: Path, chunk_sec: int | None = None) -> tuple[str, list[dict]]:
    if os.path.exists(media_path.with_suffix(".words.json")):
        return _load_cached_whisper_data(media_path.with_suffix(".words.json"))

    kw = {"chunk_length": max(chunk_sec, 1)} if chunk_sec else {}

    def _run(lang: str | None = None):
        seg_iter, info = _whisper.transcribe(str(wav), language=lang, beam_size=5, vad_filter=True,
                                             vad_parameters=dict(min_silence_duration_ms=300), no_speech_threshold=0.0,
                                             compression_ratio_threshold=float("inf"), word_timestamps=True, **kw)
        segments = list(seg_iter)
        text = " ".join(s.text.strip() for s in segments)
        words = [{"token": w.word, "start": round(w.start, 2), "end": round(w.end, 2)} for s in segments for w in
                 getattr(s, "words", []) or []]
        return text, info.language, words

    text, detected_language, words = _run()
    if detected_language != "ru" and sum("а" <= ch.lower() <= "я" for ch in text) / max(len(text), 1) > 0.40:
        text, detected_language, words = _run("ru")

    escaped = _get_escaped_word_timestamps(words, detected_language)
    media_path.with_suffix(".words.json").write_bytes(orjson.dumps(escaped, option=orjson.OPT_INDENT_2))
    return text, escaped


def _get_spoken_words(words, t_sec, word_margin: float = 0.4):
    return "".join(w["token"] for w in words if (w["start"] - word_margin) <= t_sec <= (w["end"] + word_margin))


def _ocr_video_frames(video_path: Path, every_ms: int = 800, words: list[dict] | None = None) -> str:
    detected_language = words[0]['language'] + ("g" if words[0]['language'] == "en" else "s")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return ""

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(fps * every_ms / 1000)

    collected = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ocr_txt = pytesseract.image_to_string(gray, lang=detected_language).strip()
            spoken = _get_spoken_words(words, t_sec)
            if ocr_txt:
                collected.append(f"[{t_sec:05.2f}s] Text on video frame: \"{ocr_txt}\"")
            if spoken:
                collected.append(f"[{t_sec:05.2f}s] Spoken audio in that moment: {spoken}")
        frame_idx += 1

    cap.release()
    return "\n".join(collected)


def _ocr_image(img_path: Path) -> str:
    return pytesseract.image_to_string(cv2.imread(str(img_path)), lang="eng+rus")


def _chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small", chunk_size=8000, chunk_overlap=200
    )
    return splitter.split_text(text)


def _embed_and_store(chunks: Iterable[str], url: str) -> List[str]:
    valid_docs = []
    for ch in chunks:
        valid_docs.append(Document(page_content=ch, metadata={"source_url": url}))

    return _vectorstore.add_documents(valid_docs) if valid_docs else []


def _extract_useful_metadata(meta_data: dict):
    useful_meta_data_fields = [
        'title',
        'fulltitle',
        'description',
        'duration_string',
        'upload_date',
        'uploader',
        'channel',
        'like_count',
        'comment_count',
        'webpage_url',
        'resolution',
    ]

    return {data_field: meta_data[data_field] for data_field in useful_meta_data_fields}


async def ingestion_workflow(url: str) -> IngestResult:
    media_path, meta_data = _download_media(url, _DL_DIR)
    only_useful_metadata_for_llm = _extract_useful_metadata(meta_data)
    merged_parts = [orjson.dumps(only_useful_metadata_for_llm, option=orjson.OPT_INDENT_2).decode()]
    if media_path.suffix == ".mp4":
        audio = _extract_audio(media_path)
        text, words = _whisper_transcribe(media_path, audio)
        merged_parts.append(text)
        merged_parts.append(_ocr_video_frames(media_path, words=words))
    else:
        merged_parts.append(_ocr_image(media_path))

    merged_text = "\n".join(p for p in merged_parts if p)
    chunks = _chunk_text(merged_text)
    doc_ids = _embed_and_store(chunks, url)
    return IngestResult(url=url, doc_ids=doc_ids, merged_text=merged_text)
