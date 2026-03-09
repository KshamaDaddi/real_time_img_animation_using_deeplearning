import os
import sys


def prefetch_base_model():
    """Download/cache the base model repo using Hugging Face Hub directly."""
    from huggingface_hub import snapshot_download

    base_model = os.environ.get("SD_BASE_MODEL", "runwayml/stable-diffusion-v1-5")
    print(f"[prefetch] Snapshotting base model: {base_model}")
    try:
        local_dir = snapshot_download(repo_id=base_model)
        print(f"[prefetch] Base model cached at {local_dir}")
    except Exception as e:
        print(f"[prefetch] Base model download failed: {e}")


def prefetch_lora(style_key: str, repo_env: str, weight_env: str):
    """Optionally download a LoRA weight file from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download

    repo = os.environ.get(repo_env)
    weight_name = os.environ.get(weight_env)
    if not repo:
        print(f"[prefetch] {style_key}: no repo env {repo_env} set; skipping")
        return
    print(f"[prefetch] {style_key}: downloading {repo}/{weight_name or '(default)'}")
    try:
        # If weight_name is None, the repo must provide a default file via pipeline APIs later
        if weight_name:
            local_path = hf_hub_download(repo_id=repo, filename=weight_name)
            print(f"[prefetch] {style_key}: cached at {local_path}")
        else:
            # Trigger repo presence; actual weight will be loaded by pipeline via weight_name
            hf_hub_download(repo_id=repo, filename="README.md")
            print(f"[prefetch] {style_key}: repo cached; weight will be loaded at runtime")
    except Exception as e:
        print(f"[prefetch] {style_key}: LoRA download failed: {e}")


def main():
    prefetch_base_model()

    # Optional style LoRAs – set envs before running if you want them cached
    styles = [
        ("90s_anime", "SD_LORA_90S_REPO", "SD_LORA_90S_WEIGHT"),
        ("cyberpunk_anime", "SD_LORA_CYBERPUNK_REPO", "SD_LORA_CYBERPUNK_WEIGHT"),
        ("ghibli_anime", "SD_LORA_GHIBLI_REPO", "SD_LORA_GHIBLI_WEIGHT"),
        ("aesthetic_anime", "SD_LORA_AESTHETIC_REPO", "SD_LORA_AESTHETIC_WEIGHT"),
        ("japanese_anime", "SD_LORA_JAPANESE_REPO", "SD_LORA_JAPANESE_WEIGHT"),
        ("anime_illustration", "SD_LORA_ILLUSTRATION_REPO", "SD_LORA_ILLUSTRATION_WEIGHT"),
        ("fantasy_anime", "SD_LORA_FANTASY_REPO", "SD_LORA_FANTASY_WEIGHT"),
    ]
    for key, repo_env, weight_env in styles:
        prefetch_lora(key, repo_env, weight_env)

    print("[prefetch] Done.")


if __name__ == "__main__":
    sys.exit(main())