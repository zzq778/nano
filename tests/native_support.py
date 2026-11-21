"""Skip ABI-incompatible bundled libraries on development hosts, not on Jetson."""
import platform
from pathlib import Path


def native_library_matches_host(path):
    path = Path(path)
    if not path.is_file():
        return False
    # The project carries ARM64 ELF binaries. Inspect the binary instead of
    # treating an x86 loader failure as a failure of the image-processing tests.
    host_machine = {'x86_64': 62, 'amd64': 62, 'aarch64': 183, 'arm64': 183}.get(platform.machine().lower())
    with path.open('rb') as stream:
        header = stream.read(20)
    if len(header) < 20 or header[:4] != b'\x7fELF' or host_machine is None:
        return True  # Let the real loader report unusual/unsupported environments.
    byteorder = 'little' if header[5] == 1 else 'big'
    return int.from_bytes(header[18:20], byteorder) == host_machine
