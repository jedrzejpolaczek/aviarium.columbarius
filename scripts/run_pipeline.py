from src.data.cards.pipelines import daily_pipeline
from src.logger import setup_logging


def main() -> None:
    setup_logging()
    config_path = "configs/data_sources.yaml"

    daily_pipeline(config_path)


if __name__ == "__main__":
    main()
