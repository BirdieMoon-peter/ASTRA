from __future__ import annotations

from astra.paper.export_results import export_paper_artifacts


def main() -> None:
    export_paper_artifacts()
    print("[OK] Paper export completed.")


if __name__ == "__main__":
    main()
