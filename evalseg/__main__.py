"""Entry point for evalseg."""

from . import cli


if __name__ == "__main__":  # pragma: no cover
    root_data = 'datasets'
    out_root = 'out'
    cli.mutli_run_all_datasets(root_data, out_root)
