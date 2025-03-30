from huggingface_hub import create_repo, HfApi
import argparse
api = HfApi()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument(
        "folder_path",
        type=str,
        help="Path to the folder to upload",
    )
    args.add_argument(
        "--repo_id",
        type=str,
        help="ID of the repository to upload to",
    )
    args.add_argument(
        "--repo_type",
        type=str,
        help="Type of the repository to upload to",
    )
    args = args.parse_args()
    print(args)
    api.upload_large_folder(
        folder_path=args.folder_path,
        repo_id=args.repo_id,
        repo_type='dataset',
    )