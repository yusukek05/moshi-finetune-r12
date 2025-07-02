import os

with open("/home/acg17145sv/experiments/0162_dialogue_model/moshi-finetune/missing_text_dialogue_names.txt") as f:
    ids = [line.strip() for line in f if line.strip()]

base_path = "/home/acg17145sv/experiments/0162_dialogue_model/moshi-finetune/data/J-CHAT/tokenized_audio/podcast_valid"

for id_ in ids:
    file_path = f"{base_path}/{id_}.npz"
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"Not found: {file_path}")
