from __future__ import annotations

import argparse
import os
from pathlib import Path

from run_rigen import MODEL_NAME, ROOT, apply_transformers5_gemma_shims, run_refiner


def main():
    parser = argparse.ArgumentParser(description=f'Refine a completed {MODEL_NAME} rollout.')
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    source = args.run_dir.resolve()
    output = args.output_dir.resolve() if args.output_dir else source / 'refined'
    if output.exists():
        raise FileExistsError(output)
    for path in (source / 'closed_loop_manifest.jsonl', source / 'model_rollout/refiner_cache.pt'):
        if not path.is_file():
            raise FileNotFoundError(path)
    os.chdir(ROOT)
    apply_transformers5_gemma_shims()
    # Kept as a compatibility entry point; normal `run_rigen.py` execution
    # now calls the same helper inline after the native AR rollout.
    run_refiner(source, output)


if __name__ == '__main__':
    main()
