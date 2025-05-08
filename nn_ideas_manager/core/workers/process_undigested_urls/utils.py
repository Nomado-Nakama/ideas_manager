import os
import time
import uuid
import orjson
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Final

import cv2
import loguru
import yt_dlp

import ollama
from dotenv import load_dotenv
from torch.cuda import is_available
from faster_whisper import WhisperModel
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nn_ideas_manager.core.rag import vector_store, doc_store, summarizer

# -----------------------------------------------------------------------------
# CONSTANTS & ONE-TIME INITIALISATION
# -----------------------------------------------------------------------------
load_dotenv(r'\Projects\python\nakama-ideas-manager\configs\.env')

_IG_COOKIES_PATH = Path("cookies.txt").resolve()

_META_SUFFIX_JSON = ".meta.json"

_DL_DIR = Path("/ingestion/downloaded_content")
_DL_DIR.mkdir(parents=True, exist_ok=True)

_GEMMA_CACHE_DIR = Path("/ingestion/gemma_cache")
_GEMMA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_SUMMARY_CACHE_DIR = Path("/ingestion/summary_cache")
_SUMMARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
#  Throttle every HTTP request made by yt-dlp to stay below IG/CDN rate-limits
# ---------------------------------------------------------------------------

_YT_SLEEP_MIN = float(os.getenv("YT_SLEEP_MIN", "1"))  # seconds
_YT_SLEEP_MAX = float(os.getenv("YT_SLEEP_MAX", "5"))
if _YT_SLEEP_MAX < _YT_SLEEP_MIN:  # guard against bad env values
    _YT_SLEEP_MAX = _YT_SLEEP_MIN

loguru.logger.info(
    "yt-dlp will sleep for a random %.1f-%.1f s between HTTP requests",
    _YT_SLEEP_MIN, _YT_SLEEP_MAX,
)

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


def _get_escaped_word_timestamps(words: list[dict]):
    return [
        {"token": w['token'], "start": float(round(w['start'], 2)),
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
        "sleep_requests": _YT_SLEEP_MIN,
        "max_sleep_requests": _YT_SLEEP_MAX,
        "sleep_interval": _YT_SLEEP_MIN,
        "max_sleep_interval": _YT_SLEEP_MAX,
        "cookiefile": str(_IG_COOKIES_PATH)
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


def _whisper_transcribe(media_path: Path, wav: Path, chunk_sec: int | None = None, content_language: str = 'en') -> \
tuple[str, list[dict]]:
    if os.path.exists(media_path.with_suffix(".words.json")):
        return _load_cached_whisper_data(media_path.with_suffix(".words.json"))

    kw = {"chunk_length": max(chunk_sec, 1)} if chunk_sec else {}

    seg_iter, info = _whisper.transcribe(str(wav), language=content_language, beam_size=5, vad_filter=True,
                                         vad_parameters=dict(min_silence_duration_ms=300), no_speech_threshold=0.0,
                                         compression_ratio_threshold=float("inf"), word_timestamps=True, **kw)
    segments = list(seg_iter)
    text = " ".join(s.text.strip() for s in segments)
    words = [{"token": w.word, "start": round(w.start, 2), "end": round(w.end, 2)} for s in segments for w in
             getattr(s, "words", []) or []]

    escaped = _get_escaped_word_timestamps(words)
    media_path.with_suffix(".words.json").write_bytes(orjson.dumps(escaped, option=orjson.OPT_INDENT_2))

    return text, escaped


def _get_spoken_words(words, t_sec, word_margin: float = 0.2):
    return "".join(w["token"] for w in words if (w["start"] - word_margin) <= t_sec <= (w["end"] + word_margin))


def _convert_language_code(lang_code):
    convertion_dict = {
        "en": "eng",
        "ru": "rus"
    }

    if lang_code in convertion_dict.keys():
        return convertion_dict[lang_code]
    else:
        return lang_code


def _ocr_video_frames(
        video_path: Path,
        every_ms: int = 1200,
        words: list[dict] | None = None,
        content_language: str = "en",
        *,
        gemma_model: str = "gemma3:12b",
) -> str:
    """
    Scan *video_path* every *every_ms* ms and extract:
      • On‑screen text (Tesseract *and* Gemma‑3 Vision)
      • Synchronously spoken words within ±0.2 s

    Returns a newline‑separated log suitable for downstream LLM summarisation.
    """
    # Translate ISO‑639‑1 → Tesseract codes (rus/eng, etc.)
    content_language = _convert_language_code(content_language)

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
            # Timestamp (in seconds) of current frame
            t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # ----- Gemma‑3 Vision OCR ------------------------------------------
            gemma_txt = ""
            frame_key = f"{video_path.stem}_{int(t_sec * 1000):08d}"  # e.g. vid123_0000800
            cache_f = _GEMMA_CACHE_DIR / f"{frame_key}.md"

            try:
                if cache_f.exists():  # 1️⃣  read cache
                    gemma_txt = cache_f.read_text(encoding="utf-8").strip()
                else:
                    # encode frame → PNG → bytes (no tmp files); measure runtime
                    start_t = time.perf_counter()
                    ok, png_bytes = cv2.imencode(".png", frame)
                    if ok:
                        gemma_txt = _gemma3_ocr(bytes(png_bytes),
                                                model_name=gemma_model).strip()
                        # 2️⃣  persist cache
                        cache_f.write_text(gemma_txt, encoding="utf-8")
                        elapsed = time.perf_counter() - start_t
                        loguru.logger.info(f"Gemma‑3 OCR {frame_key} took {elapsed:.2f}s")
            except Exception as exc:
                loguru.logger.warning(f"Gemma‑3 OCR failed at {t_sec:.2f}s: {exc}")

            # ----- Spoken words -------------------------------------------------
            spoken = _get_spoken_words(words, t_sec)

            # ----- Collect output ----------------------------------------------
            if gemma_txt:
                collected.append(
                    f"[{t_sec:05.2f}s] Gemma‑3 text: \"{gemma_txt}\""
                )
            if spoken:
                collected.append(
                    f"[{t_sec:05.2f}s] Spoken audio: {spoken}"
                )

        frame_idx += 1

    cap.release()
    return "\n".join(collected)


def _chunk_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small", chunk_size=8000, chunk_overlap=200
    )
    return splitter.split_text(text)


def _embed_and_store(chunks: Iterable[str], url: str) -> List[str]:
    valid_docs = []
    for ch in chunks:
        valid_docs.append(Document(page_content=ch, metadata={"source_url": url}))

    return vector_store.add_documents(valid_docs) if valid_docs else []


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


def _determine_content_language(content_description):
    if sum("а" <= ch.lower() <= "я" for ch in content_description) / max(len(content_description), 1) > 0.40:
        return "ru"
    else:
        return "en"


def _gemma3_ocr(image: Path | bytes,
                *,
                prompt: str | None = None,
                model_name: str = "gemma3:12b") -> str:
    """
    Perform high‑quality OCR on *image_path* using Gemma‑3 Vision via Ollama.

    Parameters
    ----------
    image_path : Path | bytes
        Path to a PNG/JPEG image.
    prompt : str | None
        Custom system prompt.  Defaults to a concise Markdown‑oriented prompt.
    model_name : str
        Ollama model tag.  Defaults to ``gemma3:12b``.

    Returns
    -------
    str
        Structured Markdown extracted from the image.
    """

    DEFAULT_PROMPT: Final[str] = (
        "ANALYZE THE TEXT IN THE PROVIDED IMAGE. SHORTLY DESCRIBE WHAT YOU SEE ON THE IMAGE. "
        "EXTRACT ALL READABLE CONTENT AND PRESENT IT IN A STRUCTURED MARKDOWN FORMAT THAT IS CLEAR, "
        "CONCISE, AND WELL‑ORGANIZED. DO NOT RETURN ANY TEXT NOT RELATED TO THE IMAGE SUCH AS HUMAN LIKE COMMENTS "
        "BEFORE MEANINGFUL OUTPUT. TRY TO BE AS SHORT AS POSSIBLE, BUT DO NOT LEAVE OUT ANY MEANINGFUL INFORMATION."
    )
    use_prompt = prompt or DEFAULT_PROMPT

    if isinstance(image, (bytes, bytearray)):  # ← new
        img_bytes = image
    else:
        img_bytes = Path(image).expanduser().resolve().read_bytes()

    response = ollama.chat(
        model=model_name,
        messages=[{
            "role": "user",
            "content": use_prompt,
            "images": [img_bytes],
        }],
    )

    # ollama‑py returns an object with .message for streaming, but dict for sync
    if hasattr(response, "message"):
        return response.message.content
    if isinstance(response, dict) and "message" in response:
        return response["message"]["content"]
    raise RuntimeError("Unexpected response format from Ollama")


async def _summarize_document(media_path: Path, document_text: str):
    summary_file_cache = f"{media_path.stem}_summary"
    cache_f = _SUMMARY_CACHE_DIR / f"{summary_file_cache}.md"

    if not cache_f.exists():
        loguru.logger.info(f"{cache_f} summary cache file not exists, requesting summary from OpenAI...")
        try:
            summary = (await summarizer.ainvoke(
                "Summarize the following Instagram content in one short paragraph, "
                "preserving names, proper nouns and key facts:\n\n" + document_text
            )).content
            loguru.logger.info(f"Got summary from OpenAI {summary}...")
            cache_f.write_text(summary, encoding="utf-8")
            loguru.logger.info(f"Summary cached to {cache_f}...")
            return summary
        except Exception as e:
            loguru.logger.info(f"Failed to summarize {media_path}...")
            loguru.logger.exception(e)

    else:
        return cache_f.read_text(encoding="utf-8").strip()


async def ingestion_workflow(url: str): # -> IngestResult:
    media_path, meta_data = _download_media(url, _DL_DIR)
    only_useful_metadata_for_llm = _extract_useful_metadata(meta_data)
    content_language = _determine_content_language(only_useful_metadata_for_llm['description'])
    only_useful_metadata_for_llm['content_language'] = content_language
    merged_parts = [orjson.dumps(only_useful_metadata_for_llm, option=orjson.OPT_INDENT_2).decode()]

    if media_path.suffix == ".mp4":
        audio = _extract_audio(media_path)
        loguru.logger.info(f"Audio extracted...")
        text, words = _whisper_transcribe(media_path, audio, content_language=content_language)
        loguru.logger.info(f"Whisper transcribed text...")
        merged_parts.append(text)
        merged_parts.append(_ocr_video_frames(media_path, words=words, content_language=content_language))
        loguru.logger.info(f"Gemma ocred frames...")
    else:
        merged_parts.append(_gemma3_ocr(media_path))

    merged_text = "\n".join(p for p in merged_parts if p)

    loguru.logger.info(f"Got doc of {len(merged_text)} characters...")

    # summary = await _summarize_document(media_path, merged_text)
    # loguru.logger.info(f"Got doc summary {summary}...")
    #
    # doc_id = str(uuid.uuid4())
    # summary_doc = Document(page_content=summary,
    #                        metadata={"doc_id": doc_id, "source_url": url})
    # vector_store.add_document(summary_doc)
    # loguru.logger.info(f"Added summary doc to vector store...")
    #
    # await doc_store.put(doc_id, merged_text, url)
    # loguru.logger.info(f"Added full doc to doc store...")
    #
    # # chunks = _chunk_text(merged_text)
    # # doc_ids = _embed_and_store(chunks, url)
    # loguru.logger.info(f"Ingested {url} with doc_id: {doc_id}...")
    # loguru.logger.info(f"Full merged text: {merged_text}")
    # return IngestResult(url=url, doc_ids=[doc_id], merged_text=merged_text)
