from pathlib import Path

import lasio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "force2020_las"


def main() -> None:
    """Inspect the first FORCE 2020 LAS file."""

    las_files = sorted(DATA_DIR.rglob("*.las"))

    print(f"LAS files found: {len(las_files)}")

    if not las_files:
        raise FileNotFoundError(f"No LAS files found under: {DATA_DIR}")

    first_file = las_files[0]
    print(f"\nFirst file: {first_file}")
    
    las = lasio.read(first_file)

    print("\nWell information:")
    for item in las.well:
        print(f"{item.mnemonic}: {item.value}")

    print("\nAvailable curves:")
    for curve in las.curves:
        print(
            f"{curve.mnemonic:30} "
            f"unit={curve.unit:12} "
            f"description={curve.descr}"
        )

    dataframe = las.df()

    print("\nData shape:")
    print(dataframe.shape)

    print("\nFirst five rows:")
    print(dataframe.head())

    print("\nColumn names:")
    print(dataframe.columns.tolist())


if __name__ == "__main__":
    main()