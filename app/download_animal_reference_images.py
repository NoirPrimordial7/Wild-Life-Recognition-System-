from __future__ import annotations

from utils.runtime_bootstrap import maybe_relaunch_with_project_runtime

maybe_relaunch_with_project_runtime()

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import Request, urlopen

from utils.class_names import load_class_names
from utils.reference_image_service import ReferenceImageService, normalize_class_name

WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
WIKIPEDIA_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "WildlifeDetectionSystem/1.1 (desktop dashboard reference image downloader)"
MIN_PREFERRED_WIDTH = 800
SUPPORTED_OUTPUT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUSPICIOUS_IMAGE_KEYWORDS = {
    "logo",
    "icon",
    "map",
    "range",
    "distribution",
    "locator",
    "coat_of_arms",
    "flag",
    "diagram",
    "skeleton",
    "skull",
}


@dataclass(slots=True)
class DownloadRecord:
    class_name: str
    local_file_path: str | None
    source_url: str | None
    source_title: str | None
    download_status: str
    error: str | None = None


@dataclass(slots=True)
class SummaryCandidate:
    title: str
    image_url: str
    width: int
    height: int
    score: int


def build_request(url: str) -> Request:
    return Request(url, headers={"User-Agent": USER_AGENT})


def fetch_json(url: str) -> dict:
    with urlopen(build_request(url), timeout=25) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, output_path: Path) -> None:
    with urlopen(build_request(url), timeout=45) as response:
        output_path.write_bytes(response.read())


def infer_extension(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in SUPPORTED_OUTPUT_EXTENSIONS:
        return suffix
    return ".jpg"


def candidate_titles(class_name: str) -> list[str]:
    cleaned = class_name.strip()
    compact = cleaned.replace("_", " ")
    title_case = compact.title()
    return list(
        dict.fromkeys(
            [
                cleaned,
                compact,
                compact.replace(" ", "_"),
                title_case,
                title_case.replace(" ", "_"),
            ]
        )
    )


def search_wikipedia_titles(query: str, limit: int = 6) -> list[str]:
    params = urlencode(
        {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": str(limit),
            "utf8": "1",
        }
    )
    payload = fetch_json(f"{WIKIPEDIA_SEARCH_URL}?{params}")
    return [item.get("title", "") for item in payload.get("query", {}).get("search", []) if item.get("title")]


def fetch_wikipedia_summary(title: str) -> dict:
    return fetch_json(WIKIPEDIA_SUMMARY_URL.format(title=quote(title)))


def extract_image_block(summary: dict) -> tuple[str, int, int] | None:
    image_block = summary.get("originalimage") or summary.get("thumbnail")
    if not isinstance(image_block, dict):
        return None

    image_url = image_block.get("source")
    if not image_url:
        return None

    width = int(image_block.get("width") or 0)
    height = int(image_block.get("height") or 0)
    return image_url, width, height


def looks_like_non_reference_image(summary: dict, image_url: str) -> bool:
    haystack = " ".join(
        [
            summary.get("title", ""),
            summary.get("description", ""),
            image_url,
        ]
    ).lower()
    return any(keyword in haystack for keyword in SUSPICIOUS_IMAGE_KEYWORDS)


def score_summary_candidate(class_name: str, summary: dict, image_url: str, width: int) -> int:
    normalized_class = normalize_class_name(class_name)
    normalized_title = normalize_class_name(summary.get("title", ""))

    score = min(width, 2400) // 10
    if normalized_title == normalized_class:
        score += 350
    elif normalized_class in normalized_title or normalized_title in normalized_class:
        score += 140

    if summary.get("originalimage"):
        score += 70
    if width >= MIN_PREFERRED_WIDTH:
        score += 120

    description = str(summary.get("description", "")).lower()
    if any(term in description for term in ("species", "mammal", "bird", "fish", "reptile", "animal")):
        score += 20

    if looks_like_non_reference_image(summary, image_url):
        score -= 600

    if summary.get("type") == "disambiguation":
        score -= 1000

    return score


def find_best_candidate(class_name: str) -> tuple[SummaryCandidate | None, str | None]:
    titles = candidate_titles(class_name)
    last_error: str | None = None

    try:
        titles.extend(search_wikipedia_titles(class_name))
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        last_error = str(exc)

    best_candidate: SummaryCandidate | None = None
    seen_titles: set[str] = set()

    for title in titles:
        if not title or title in seen_titles:
            continue
        seen_titles.add(title)

        try:
            summary = fetch_wikipedia_summary(title)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = str(exc)
            continue

        image_block = extract_image_block(summary)
        if image_block is None:
            continue

        image_url, width, height = image_block
        candidate = SummaryCandidate(
            title=summary.get("title", title),
            image_url=image_url,
            width=width,
            height=height,
            score=score_summary_candidate(class_name, summary, image_url, width),
        )

        if best_candidate is None or candidate.score > best_candidate.score:
            best_candidate = candidate

        if candidate.score >= 360 and candidate.width >= MIN_PREFERRED_WIDTH:
            return candidate, last_error

    return best_candidate, last_error


def resolve_target_classes(all_class_names: list[str], only_terms: list[str] | None, limit: int | None) -> list[str]:
    if not only_terms:
        selected = list(all_class_names)
    else:
        by_normalized = {normalize_class_name(name): name for name in all_class_names}
        selected = []
        for term in only_terms:
            selected.append(by_normalized.get(normalize_class_name(term), term.strip()))

    if limit is not None:
        selected = selected[:limit]
    return selected


def relative_to_project_root(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def download_reference_image_for_class(
    class_name: str,
    *,
    service: ReferenceImageService,
    project_root: Path,
    force: bool,
) -> DownloadRecord:
    existing_image = service.get_reference_image_path(class_name)
    if existing_image is not None and not force:
        return DownloadRecord(
            class_name=class_name,
            local_file_path=relative_to_project_root(existing_image, project_root),
            source_url=None,
            source_title=None,
            download_status="skipped_existing",
        )

    try:
        candidate, resolution_error = find_best_candidate(class_name)
    except Exception as exc:
        return DownloadRecord(
            class_name=class_name,
            local_file_path=None,
            source_url=None,
            source_title=None,
            download_status="failed",
            error=str(exc),
        )

    if candidate is None:
        return DownloadRecord(
            class_name=class_name,
            local_file_path=None,
            source_url=None,
            source_title=None,
            download_status="failed",
            error=resolution_error or "No suitable Wikimedia/Wikipedia image found.",
        )

    target_extension = existing_image.suffix.lower() if existing_image is not None else infer_extension(candidate.image_url)
    output_path = existing_image if existing_image is not None else service.build_output_path(class_name, target_extension)

    try:
        download_file(candidate.image_url, output_path)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return DownloadRecord(
            class_name=class_name,
            local_file_path=None,
            source_url=candidate.image_url,
            source_title=candidate.title,
            download_status="failed",
            error=str(exc),
        )

    return DownloadRecord(
        class_name=class_name,
        local_file_path=relative_to_project_root(output_path, project_root),
        source_url=candidate.image_url,
        source_title=candidate.title,
        download_status="downloaded",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download best-effort public animal reference images from Wikimedia/Wikipedia."
    )
    parser.add_argument("--force", action="store_true", help="Redownload images even if a local reference image already exists.")
    parser.add_argument("--limit", type=int, help="Only process the first N class names.")
    parser.add_argument(
        "--only",
        action="append",
        help="Only process the given class name. Repeat the flag to download multiple specific animals.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    service = ReferenceImageService()
    class_names = load_class_names()
    project_root = service.reference_dir.parent.parent
    target_classes = resolve_target_classes(class_names, args.only, args.limit)
    report_path = service.get_download_report_path()

    print(f"Reference image directory: {service.reference_dir}")
    print(f"Classes selected: {len(target_classes)}")
    print(f"Force redownload: {args.force}")
    print()

    results: list[DownloadRecord] = []
    for class_name in target_classes:
        record = download_reference_image_for_class(
            class_name,
            service=service,
            project_root=project_root,
            force=args.force,
        )
        results.append(record)

        if record.download_status == "downloaded":
            print(f"OK    {class_name}: saved -> {record.local_file_path}")
        elif record.download_status == "skipped_existing":
            print(f"SKIP  {class_name}: already exists -> {record.local_file_path}")
        else:
            print(f"FAIL  {class_name}: {record.error}")

    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Wikimedia/Wikipedia public images where available",
        "force": bool(args.force),
        "limit": args.limit,
        "only": args.only or [],
        "results": [asdict(record) for record in results],
        "counts": {
            "requested": len(target_classes),
            "downloaded": sum(1 for record in results if record.download_status == "downloaded"),
            "skipped_existing": sum(1 for record in results if record.download_status == "skipped_existing"),
            "failed": sum(1 for record in results if record.download_status == "failed"),
        },
    }
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Download report saved to: {report_path}")
    print("Finished best-effort reference image download.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
