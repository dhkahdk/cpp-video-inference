"""Create a synthetic image for the model-free preprocessing check."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/smoke.ppm"))
    args = parser.parse_args()

    width, height = 971, 543
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as target:
        target.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        for y in range(height):
            row = bytearray()
            for x in range(width):
                row.extend(((x * 3) % 256, (y * 5) % 256, (x + y) % 256))
            target.write(row)
    print(f"Created {args.output} ({width}x{height})")


if __name__ == "__main__":
    main()
