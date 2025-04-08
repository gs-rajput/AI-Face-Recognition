import os
import shutil

encodings_dir = "encodings"
dataset_dir = "dataset"

def list_profiles():
    return [f.replace(".pkl", "") for f in os.listdir(encodings_dir) if f.endswith(".pkl")]

def delete_profile(name):
    try:
        os.remove(os.path.join(encodings_dir, f"{name}.pkl"))
        shutil.rmtree(os.path.join(dataset_dir, name))
        print(f"[INFO] Deleted profile: {name}")
    except Exception as e:
        print(f"[ERROR] {e}")

print("1. List all profiles")
print("2. Delete a profile")
choice = input("Enter your choice: ").strip()

if choice == "1":
    profiles = list_profiles()
    print("Saved profiles:")
    for p in profiles:
        print("-", p)

elif choice == "2":
    name = input("Enter the name to delete: ").strip()
    delete_profile(name)

