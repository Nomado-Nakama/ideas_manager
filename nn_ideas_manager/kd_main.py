from pathlib import Path

from knowledge_devourer.core.config import Config
from knowledge_devourer.core.utils import load_links
from knowledge_devourer.core.processor import process_posts, process_reels


def main():
    links = load_links(path="reels_and_posts_links.txt", limit=10)

    config = Config(
        Path(__file__).parent.joinpath("storage")
    )

    process_posts(
        links=links,
        config=config
    )  # handle /p/ links

    process_reels(
        links=links,
        config=config
    )  # handle /reel/ links


if __name__ == "__main__":
    main()
