import sys

def sync(src_path, dest_path):
    with open(src_path, 'r') as f:
        src_content = f.read()

    try:
        logic = src_content.split("# <logic-sync>")[1].split("# </logic-sync>")[0]
    except IndexError:
        print(f"Error: Could not find tags in {src_path}")
        sys.exit(1)

    with open(dest_path, 'r') as f:
        dest_content = f.read()

    parts = dest_content.split("# <logic-sync>")
    prefix = parts[0]
    suffix = parts[1].split("# </logic-sync>")[1]

    new_content = f"{prefix}# <logic-sync>{logic}# </logic-sync>{suffix}"

    with open(dest_path, 'w') as f:
        f.write(new_content)
    print(f"✅ Synced logic to {dest_path}")

if __name__ == "__main__":
    sync(sys.argv[1], sys.argv[2])