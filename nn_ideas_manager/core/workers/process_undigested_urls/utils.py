import gc
import json
import os
import uuid
import time
import torch
import datetime
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Final, Any

import cv2
import loguru
import ollama
import orjson
import yt_dlp
from outgram import Instagram
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from outgram.models import BaseMedia, BasePost
from torch.cuda import is_available

from nn_ideas_manager.core.rag import vector_store, summarizer, doc_store

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

_DOC_CACHE_DIR = Path("/ingestion/doc_cache")
_DOC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

ig = Instagram()

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

_WHISPER_MODEL = None


@dataclass(slots=True)
class IngestResult:
    url: str
    doc_ids: List[str]
    merged_text: str


def _get_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        loguru.logger.info("Loading Whisper…")
        _WHISPER_MODEL = WhisperModel(
            model_size_or_path="large-v3",
            device="cuda" if is_available() else "cpu",
            compute_type="float16" if is_available() else "default",
        )
    return _WHISPER_MODEL


def _unload_whisper():
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        loguru.logger.info("Unloading Whisper and clearing GPU cache…")
        del _WHISPER_MODEL
        _WHISPER_MODEL = None
        torch.cuda.empty_cache()
        gc.collect()


def _dump_content_meta_data(content_id: str, meta_data: dict, out_dir: Path) -> None:
    (out_dir / f"{content_id}{_META_SUFFIX_JSON}").write_bytes(
        orjson.dumps(meta_data, option=orjson.OPT_NAIVE_UTC | orjson.OPT_INDENT_2, default=lambda o: None)
    )


def _get_escaped_word_timestamps(words: list[dict]):
    return [
        {"token": w['token'], "start": float(round(w['start'], 2)),
         "end": float(round(w['end'], 2))}
        for w in words
    ]


def _extract_content_type_and_content_id_from_url(url: str) -> tuple[str, str]:
    if "reel/" in url and "inst" in url:
        return "ig_reel", url.split("reel/")[1].split("/")[0]
    if "p/" in url and "inst" in url:
        return "ig_post", url.split("p/")[1].split("/")[0]
    raise ValueError(f"Can't parse content type and content id from: {url}")


def _load_cached_meta_data(content_id: str, out_dir: Path) -> dict | None:
    json_f = out_dir / f"{content_id}{_META_SUFFIX_JSON}"
    if json_f.exists():
        return json.loads(json_f.read_bytes())
    return None


def _remove_content_from_cache(out_dir, content_id):
    try:
        os.remove(out_dir / f"{content_id}.mp4")
        os.remove(out_dir / f"{content_id}{_META_SUFFIX_JSON}")
        archive_file = out_dir / "downloaded.txt"
        if archive_file.exists():
            content_identifier = f"instagram {content_id}"
            lines = archive_file.read_text(encoding="utf-8").splitlines()
            filtered = [line for line in lines if content_identifier not in line]
            archive_file.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    except Exception as e:
        pass


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def _download_video_media(url: str, out_dir: Path = _DL_DIR) -> tuple[Path, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _, content_id = _extract_content_type_and_content_id_from_url(url)
    cached = _load_cached_meta_data(content_id, out_dir)
    if cached:
        if isinstance(cached, dict):
            final = out_dir / f"{cached['id']}.mp4"
            if final.exists():
                return final, cached
        else:
            _remove_content_from_cache(out_dir, content_id)

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
        # "cookiefile": str(_IG_COOKIES_PATH)
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None:
            _remove_content_from_cache(out_dir, content_id)
            info = ydl.extract_info(url, download=True)
        content_id = info["id"]
        _dump_content_meta_data(content_id, content_id, out_dir)
        final_path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
        if not final_path.exists():
            raise FileNotFoundError(f"yt-dlp reported success but {final_path} is missing")
        return final_path, info


def _download_ig_post(content_id: str, out_dir: Path = _DL_DIR) -> tuple[List[Path], dict]:
    meta_data_path = out_dir / f"{content_id}{_META_SUFFIX_JSON}"
    post_images_paths: List[Path] = []

    if meta_data_path.exists():
        loguru.logger.info(f"Skipping post {content_id}; already downloaded.")

        for p in out_dir.iterdir():
            if p.is_file() and p.stem.startswith(f"{content_id}_"):
                post_images_paths.append(p)
        post_images_paths.sort()

        meta_data_dict = _load_cached_meta_data(content_id, out_dir)

        return post_images_paths, meta_data_dict

    post: BasePost = ig.post(content_id)

    loguru.logger.info(f"Saving post meta data to {meta_data_path}...")
    meta_data_dict = _to_dict_recursively(post)
    _dump_content_meta_data(content_id, meta_data_dict, out_dir)

    for i, media in enumerate(ig.download(post, parallel=4), 1):
        media: BaseMedia
        ext = media.content_type.split("/")[-1].lower()
        filename = out_dir / f"{content_id}_{i:02d}.{ext}"
        loguru.logger.info(f"Saving post media data to {filename}...")
        media.save(filename=filename)
        post_images_paths.append(filename)

    return post_images_paths, meta_data_dict


async def _ingest_instagram_reels(url: str):
    media_path, meta_data = _download_video_media(url, _DL_DIR)
    # ─── Whole‑doc cache check ──────────────────────────────────────
    doc_cache_f = _DOC_CACHE_DIR / f"{media_path.stem}.txt"
    if doc_cache_f.exists():
        merged_text = doc_cache_f.read_text(encoding="utf-8")
        loguru.logger.info(
            f"[CACHE‑HIT] Loaded cached doc for {media_path.stem} "
            f"({len(merged_text)} chars)…"
        )
        return IngestResult(url, [], merged_text)
    # ────────────────────────────────────────────────────────────────

    only_useful_metadata_for_llm = _extract_reels_useful_metadata(meta_data)
    content_language = _determine_content_language(only_useful_metadata_for_llm['description'])
    only_useful_metadata_for_llm['content_language'] = content_language
    merged_parts = [orjson.dumps(only_useful_metadata_for_llm, option=orjson.OPT_INDENT_2).decode()]

    audio = _extract_audio(media_path)
    loguru.logger.info(f"Audio extracted...")
    text, words = _whisper_transcribe(media_path, audio, content_language=content_language)
    loguru.logger.info(f"Whisper transcribed text...")
    merged_parts.append(text)

    merged_parts.append(_ocr_video_frames(media_path, words=words))
    loguru.logger.info(f"Gemma ocred frames...")

    merged_text = "\n".join(p for p in merged_parts if p)

    loguru.logger.info(f"Got doc of {len(merged_text)} characters...")

    # ─── Save to cache for future runs ──────────────────────────────
    doc_cache_f.write_text(merged_text, encoding="utf-8")
    loguru.logger.info(f"Document cached to {doc_cache_f}…")
    # ────────────────────────────────────────────────────────────────

    summary = await _summarize_document(media_path, merged_text)
    loguru.logger.info(f"Got doc summary {summary}...")

    doc_id = str(uuid.uuid4())
    # summary_doc = Document(page_content=summary,
    #                        metadata={"doc_id": doc_id, "source_url": url})
    # vector_store.add_documents([summary_doc])
    # loguru.logger.info(f"Added summary doc to vector store...")

    # await doc_store.put(doc_id, merged_text, url)
    # loguru.logger.info(f"Added full doc to doc store...")

    # chunks = _chunk_text(merged_text)
    # doc_ids = _embed_and_store(chunks, url)
    loguru.logger.info(f"Ingested {url} with doc_id: {doc_id}...")
    # loguru.logger.info(f"Full merged text: {merged_text}")
    return IngestResult(url=url, doc_ids=[doc_id], merged_text=merged_text)


async def _ingest_instagram_post(url: str) -> IngestResult:
    # 1️⃣  download / reuse ---------------------------------------------------
    _, content_id = _extract_content_type_and_content_id_from_url(url)
    media_paths, meta_data = _download_ig_post(content_id, _DL_DIR)

    # 2️⃣  whole‑doc cache ----------------------------------------------------
    doc_cache_f = _DOC_CACHE_DIR / f"{content_id}.txt"
    if doc_cache_f.exists():
        merged_text = doc_cache_f.read_text(encoding="utf-8")
        loguru.logger.info(f"[CACHE‑HIT] Loaded cached doc for {content_id} "
                           f"({len(merged_text)} chars)…")
        return IngestResult(url, [], merged_text)

    loguru.logger.info(f"No cached doc for: {url}")

    # 3️⃣  metadata block -----------------------------------------------------
    only_meta = _extract_ig_post_useful_metadata(meta_data)
    content_language = _determine_content_language(
        only_meta.get("text", "") or only_meta.get("accessibility_caption", "")
    )
    only_meta["content_language"] = content_language
    merged_parts: List[str] = [orjson.dumps(
        only_meta, option=orjson.OPT_INDENT_2).decode()]

    # 4️⃣  OCR every image (cache immediately) -------------------------------
    for p in media_paths:
        # skip non‑images (some carousels contain video)
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            loguru.logger.warning(f"Skipping non‑image {p.name} in post {content_id}")
            continue

        cache_f = _GEMMA_CACHE_DIR / f"{p.stem}.md"
        if cache_f.exists():                              # ⭐ already cached
            loguru.logger.info(f"[CACHE-HIT] for: {p.stem}")
            txt = cache_f.read_text(encoding="utf-8").strip()

        else:                                             # ⭐ fresh OCR
            try:
                loguru.logger.info(f"Running Gemma‑3 OCR model for: {p.stem}")
                start_t = time.perf_counter()
                txt = _gemma3_ocr(p,
                                  model_name="gemma3:12b",
                                  use_gpu=True)
                elapsed = time.perf_counter() - start_t
                loguru.logger.info(f"Gemma‑3 OCR {p.stem} took {elapsed:.2f}s")
                cache_f.write_text(txt, encoding="utf-8") # ↳ cache ASAP
            except Exception as exc:
                loguru.logger.error(f"Gemma‑3 OCR failed for {p.name}: {exc}")
                continue                                  # keep ingesting

        merged_parts.append(f"[{p.name}] Gemma‑3 text: \"{txt.strip()}\"")

    # 5️⃣  merge & cache doc --------------------------------------------------
    merged_text = "\n".join(filter(None, merged_parts))
    loguru.logger.info(f"Got doc of {len(merged_text)} characters…")
    doc_cache_f.write_text(merged_text, encoding="utf-8")
    loguru.logger.info(f"Document cached to {doc_cache_f}…")

    # 6️⃣  summary & vector‑store --------------------------------------------
    # summary = await _summarize_document(Path(content_id), merged_text)
    doc_id = str(uuid.uuid4())
    # vector_store.add_documents([Document(page_content=summary,
    #                                    metadata={"doc_id": doc_id,
    #                                              "source_url": url})])
    # await doc_store.put(doc_id, merged_text, url)
    # loguru.logger.info(f"Ingested post {url} with doc_id {doc_id}")

    # 7️⃣  return -------------------------------------------------------------
    return IngestResult(url=url, doc_ids=[doc_id], merged_text=merged_text)


async def ingestion_workflow(url: str) -> IngestResult:
    loguru.logger.info(f"Begin ingestion workflow for {url}")
    url_content_type, content_id = _extract_content_type_and_content_id_from_url(url)

    if url_content_type == "ig_reel":
        return await _ingest_instagram_reels(url)

    if url_content_type == "ig_post":
        return await _ingest_instagram_post(url)

    else:
        raise Exception("Unsupported content type")


def _extract_audio(video_path: Path) -> Path:
    wav = video_path.with_suffix(".wav")
    _run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
          "-i", str(video_path), "-ar", "16000", "-ac", "1", str(wav)])
    return wav


def _to_dict_recursively(obj: Any) -> Any:
    """
    Convert objects to JSON‐serializable structures,
    turning datetime/date/time into ISO strings.
    """
    if isinstance(obj, (datetime.datetime, datetime.date, datetime.time)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {k: _to_dict_recursively(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_dict_recursively(v) for v in obj)

    if hasattr(obj, "__dict__"):
        return {k: _to_dict_recursively(v) for k, v in vars(obj).items()}

    return obj


def _load_cached_whisper_data(words_path: Path) -> tuple[str, list[dict]]:
    words = orjson.loads(words_path.read_bytes())
    text = "".join(w["token"] for w in words).strip()
    loguru.logger.info(f"[CACHE HIT] GOT: {words_path}. Whisper is not loaded into VRAM...")
    return text, words


def _whisper_transcribe(media_path: Path, wav: Path, chunk_sec: int | None = None, content_language: str = 'en') -> \
        tuple[str, list[dict]]:
    if os.path.exists(media_path.with_suffix(".words.json")):
        return _load_cached_whisper_data(media_path.with_suffix(".words.json"))

    kw = {"chunk_length": max(chunk_sec, 1)} if chunk_sec else {}

    _whisper = _get_whisper()
    loguru.logger.info("Whisper is loaded into VRAM...")
    seg_iter, info = _whisper.transcribe(str(wav), language=content_language, beam_size=5, vad_filter=True,
                                         vad_parameters=dict(min_silence_duration_ms=300), no_speech_threshold=0.0,
                                         compression_ratio_threshold=float("inf"), word_timestamps=True, **kw)
    segments = list(seg_iter)
    text = " ".join(s.text.strip() for s in segments)
    words = [{"token": w.word, "start": round(w.start, 2), "end": round(w.end, 2)} for s in segments for w in
             getattr(s, "words", []) or []]

    escaped = _get_escaped_word_timestamps(words)
    media_path.with_suffix(".words.json").write_bytes(orjson.dumps(escaped, option=orjson.OPT_INDENT_2))

    _unload_whisper()
    loguru.logger.info(f"Whisper unloaded from VRAM...")
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
        every_ms: int = 2400,
        words: list[dict] | None = None,
        *,
        gemma_model: str = "gemma3:12b",
) -> str:
    """
    Scan *video_path* every *every_ms* ms and extract:
      • On‑screen text (Tesseract *and* Gemma‑3 Vision)
      • Synchronously spoken words within ±0.2 s

    Returns a newline‑separated log suitable for downstream LLM summarisation.
    """
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
            loguru.logger.info(f"Gemma ocring {video_path} current cursor is on {t_sec}...")
            try:
                if cache_f.exists():  # 1️⃣  read cache
                    loguru.logger.info(f"Gemma hit cache on {video_path} with current cursor is on {t_sec}...")
                    gemma_txt = cache_f.read_text(encoding="utf-8").strip()
                else:
                    # encode frame → PNG → bytes (no tmp files); measure runtime
                    start_t = time.perf_counter()
                    ok, png_bytes = cv2.imencode(".png", frame)
                    if ok:
                        gemma_txt = _gemma3_ocr(
                            image=bytes(png_bytes),
                            model_name=gemma_model,
                            use_gpu=True
                        ).strip()
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


def _extract_reels_useful_metadata(meta_data: dict):
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


def _extract_ig_post_useful_metadata(meta_data: dict):
    useful_meta_data_fields = [
        'code',
        'type',
        'published_at',
        'text',
        'accessibility_caption',
        'author'
    ]

    return {data_field: meta_data[data_field] for data_field in useful_meta_data_fields}


def _determine_content_language(content_description):
    if not content_description:
        return "en"
    if sum("а" <= ch.lower() <= "я" for ch in content_description) / max(len(content_description), 1) > 0.40:
        return "ru"
    else:
        return "en"


def _gemma3_ocr(
        image: Path | bytes | List[Path] | List[bytes] | tuple,
        *,
        prompt: str | None = None,
        model_name: str = "gemma3:12b",
        use_gpu: bool = True,
) -> str | list[str]:
    """
    Run Gemma‑3 Vision OCR.

    • If *image* is a single Path/bytes → returns one string.
    • If *image* is a list/tuple       → returns list[str] (same order).
    """

    default_prompt: Final[str] = (
        "ANALYZE THE TEXT IN THE PROVIDED IMAGE. SHORTLY DESCRIBE WHAT YOU SEE ON THE IMAGE. "
        "EXTRACT ALL READABLE CONTENT AND PRESENT IT IN A STRUCTURED MARKDOWN FORMAT THAT IS CLEAR, "
        "CONCISE, AND WELL‑ORGANIZED. DO NOT RETURN ANY TEXT NOT RELATED TO THE IMAGE SUCH AS HUMAN LIKE COMMENTS "
        "BEFORE MEANINGFUL OUTPUT. TRY TO BE AS SHORT AS POSSIBLE, BUT DO NOT LEAVE OUT ANY MEANINGFUL INFORMATION."
        "DO NOT REPEAT PROMPT INSTRUCTIONS IN YOUR OUTPUT."
    )

    # default_prompt: Final[str] = (
    #     """
    #     ### ROLE
    #     You are an OCR + layout interpreter.
    #     Return **only** the Markdown payload described below – no chit‑chat, no extra commentary.
    #
    #     ### TASK
    #     1. **Transcribe** every legible piece of text in the image exactly as it appears (respect capitalisation & line‑breaks).
    #     2. **Group** consecutive lines that belong together (e.g. paragraph, bullet) on the **same bullet**.
    #     3. **Describe** the visual context in ≤ 2 short sentences (fonts, colours, layout, icons, photos, etc.).
    #
    #     ### OUTPUT FORMAT (Markdown)
    #
    #     # RawText
    #     ...
    #
    #     # VisualContext
    #     ...
    #
    #     """
    # )
    use_prompt = prompt or default_prompt

    if isinstance(image, (list, tuple)):
        return [
            _gemma3_ocr(img,
                        prompt=default_prompt,
                        model_name=model_name,
                        use_gpu=use_gpu)
            for img in image
        ]

    if isinstance(image, (bytes, bytearray)):
        img_bytes = image
    else:
        img_bytes = Path(image).expanduser().resolve().read_bytes()

    if use_gpu:
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": use_prompt,
                "images": [img_bytes],
            }],
            options={'num_gpu_layers': 100, 'keep_alive': 180, 'temperature': 0}
        )
    else:
        response = ollama.chat(
            model=model_name,
            messages=[{
                "role": "user",
                "content": use_prompt,
                "images": [img_bytes],
            }],
            options={'keep_alive': 600, 'temperature': 0}
        )

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
