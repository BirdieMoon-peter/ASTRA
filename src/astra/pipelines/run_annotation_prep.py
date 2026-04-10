from __future__ import annotations

from astra.labeling.sample_for_annotation import write_dev_aligned_workset, write_gold_workset


def main(mode: str = "gold") -> None:
    if mode == "dev_aligned":
        result = write_dev_aligned_workset()
        print("[OK] Dev-aligned annotation workset generated.")
        print(result)
        return

    result = write_gold_workset()
    print("[OK] Gold annotation workset generated.")
    print(result)


if __name__ == "__main__":
    main()
