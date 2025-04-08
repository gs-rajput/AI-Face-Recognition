import os

ENCODINGS_DIR = "dl_encodings"

def list_profiles():
    profiles = [f.replace(".pkl", "") for f in os.listdir(ENCODINGS_DIR) if f.endswith(".pkl")]
    if not profiles:
        print("No profiles found.")
    else:
        print("Saved profiles:")
        for i, profile in enumerate(profiles, start=1):
            print(f"{i}. {profile}")

def delete_profile():
    list_profiles()
    name = input("Enter the name of the profile to delete: ").strip()
    file_path = os.path.join(ENCODINGS_DIR, f"{name}.pkl")
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"[INFO] Profile '{name}' deleted.")
    else:
        print("[ERROR] Profile not found.")

def main():
    while True:
        print("\n--- Manage DL Profiles ---")
        print("1. List all profiles")
        print("2. Delete a profile")
        print("3. Exit")
        choice = input("Enter your choice: ").strip()

        if choice == "1":
            list_profiles()
        elif choice == "2":
            delete_profile()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    os.makedirs(ENCODINGS_DIR, exist_ok=True)
    main()

