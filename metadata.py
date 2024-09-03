import os
from PIL import Image
from PIL.ExifTags import TAGS
import openpyxl
from openpyxl.utils.exceptions import IllegalCharacterError
import re

def sanitize_value(value):
    """Convert metadata values to a format compatible with Excel."""
    if isinstance(value, bytes):
        return value.decode(errors='ignore')  # Convert bytes to string
    elif isinstance(value, tuple):
        return str(value)  # Convert tuple to string
    elif isinstance(value, dict):
        return str(value)  # Convert dict to string
    else:
        return str(value)  # Convert other types to string

def remove_illegal_characters(value):
    """Remove characters that are illegal in Excel."""
    # Define a regex pattern to match illegal characters
    illegal_characters = re.compile(r'[\000-\010]|[\013-\014]|[\016-\037]')
    return illegal_characters.sub('', value)

def extract_metadata(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    metadata = {}

    if exif_data is not None:
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            metadata[tag_name] = remove_illegal_characters(sanitize_value(value))

    return metadata

def create_excel(metadata_list, output_file='image_metadata.xlsx'):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Image Metadata"

    headers = ["Filename"] + list(metadata_list[0][1].keys()) if metadata_list else []
    sheet.append(headers)

    for filename, metadata in metadata_list:
        try:
            row = [filename] + [metadata.get(header, "N/A") for header in headers[1:]]
            sheet.append(row)
        except IllegalCharacterError as e:
            print(f"Error writing {filename}: {e}")

    workbook.save(output_file)
    print(f"Metadata saved to {output_file}")

def main():
    metadata_list = []
    for filename in os.listdir('.'):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            metadata = extract_metadata(filename)
            metadata_list.append((filename, metadata))

    if metadata_list:
        create_excel(metadata_list)
    else:
        print("No images found in the current directory.")

if __name__ == "__main__":
    main()
