from pathlib import Path

from src.data.cards.pipelines import daily_pipeline
from src.logger import setup_logging


def main() -> None:
    log_file = setup_logging(log_dir=Path("logs"))
    if log_file:
        print(f"Logging to {log_file}")
    config_path = "configs/data_sources.yaml"

    daily_pipeline(config_path)


if __name__ == "__main__":
    main()
