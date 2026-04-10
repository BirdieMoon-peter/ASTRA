from __future__ import annotations

import json

from astra.data.build_experiment_package import build_experiment_package


def main() -> None:
    manifest = build_experiment_package()
    print("[OK] Experiment package build complete.")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
