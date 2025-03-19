import os
import shutil

def clear_folder(folder):
    """Clear the contents of a folder."""
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def clear_folder_except(folder, exclude_items):
    """Clear folder contents except specified items."""
    if os.path.exists(folder):
        for item in os.listdir(folder):
            if item not in exclude_items:
                item_path = os.path.join(folder, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
    else:
        os.makedirs(folder)

def save_file(file_path, content):
    """Save content to a specified file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def generate_unique_filename(directory, filename):
    """Generate a unique filename in a specified directory."""
    base_name, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base_name}_{counter}{ext}"
        counter += 1
    return new_filename
