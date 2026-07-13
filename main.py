from pathlib import Path

def get_folder_size(folder: str) -> int:
    folder = Path(folder)
    return sum(
        f.stat().st_size
        for f in folder.rglob("*")
        if f.is_file()
    )

def main() -> None:
    size_bytes = get_folder_size(".")
    print(f"Project: {size_bytes / (1024**3):.2f} GB")
    size_bytes = get_folder_size("src/datasets")
    print(f"Datasets: {size_bytes / (1024**3):.2f} GB")
    size_bytes = get_folder_size("src/models")
    print(f"Models: {size_bytes / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
