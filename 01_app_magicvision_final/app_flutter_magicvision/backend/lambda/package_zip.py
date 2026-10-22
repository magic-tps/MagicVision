"""Build a deterministic Linux Lambda ZIP inside the matching runtime image."""
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

if __name__ == '__main__':
    with ZipFile('/output/api.zip', 'w', compression=ZIP_DEFLATED) as archive:
        for path in sorted(Path('/asset').rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc' or path.name == 'package_zip.py':
                continue
            item = ZipInfo(path.relative_to('/asset').as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            item.external_attr = 0o100644 << 16
            item.compress_type = ZIP_DEFLATED
            archive.writestr(item, path.read_bytes())
    print('Linux Lambda package ready: /output/api.zip')
