from pathlib import Path

def create_folder(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Folder created: {path}")

def get_folder_size(folder: str, announce: bool = True) -> int:
    folder = Path(folder)
    total_size = sum(
        f.stat().st_size
        for f in folder.rglob("*")
        if f.is_file()
    )
    
    if announce:
        print(f"Size of folder '{folder}': {total_size / (1024**2):.2f} MB")

    return total_size
