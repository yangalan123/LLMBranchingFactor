import sys


def main():
    artifact_args = {
        "--artifact_pattern",
        "--artifact_patterns_file",
    }
    if any(arg in artifact_args or arg.startswith("--artifact_pattern") for arg in sys.argv[1:]):
        from build_model_token_prefix_random_string_datasets import main as token_prefix_main
        token_prefix_main()
    else:
        from build_prefix_source_dataset import main as legacy_main
        legacy_main()


if __name__ == "__main__":
    main()
