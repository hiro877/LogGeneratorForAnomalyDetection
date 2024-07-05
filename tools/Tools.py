
# def load_frequencies_file(file_path):
#     frequencies = []
#     print("load_frequencies_file(file_path): file_path={}".format(file_path))
#     with open(file_path, 'r', encoding='utf-8') as file:
#         for line in file:
#             parts = line.split()
#             if parts:
#                 last_value = parts[-1].replace("[%]", "")  # Remove the percentage sign and brackets
#                 frequencies.append(float(last_value))
#     return frequencies


import os
from tqdm import tqdm

def load_frequencies_file(file_path):
    # Check the file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    print("file_extension: {}".format(file_extension))
    if file_extension != ".freq":
        print("Processing non-.freq file:", file_path)
        frequencies = load_frequencies_file_(file_path)

        # Generate new file path with .freq extension
        new_file_path = os.path.splitext(file_path)[0] + ".freq"

        # Save the frequencies to the new file
        save_frequencies_file(new_file_path, frequencies)
    else:
        # Load frequencies from the file
        print("Processing .freq file:", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        frequencies = []
        for line in tqdm(lines):
            parts = line.rstrip('\n')
            frequencies.append(float(parts))

        # print(frequencies)
    return frequencies


def load_frequencies_file_(file_path):
    frequencies = []
    print("load_frequencies_file(file_path): file_path={}".format(file_path))
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file):
            parts = line.split()
            if parts:
                last_value = parts[-1].replace("[%]", "")  # Remove the percentage sign and brackets
                frequencies.append(float(last_value))
    return frequencies


def save_frequencies_file(file_path, frequencies):
    with open(file_path, 'w', encoding='utf-8') as file:
        for frequency in frequencies:
            file.write(f"{frequency}\n")





# Example usage
# file_path = "example.freq"
# process_file(file_path)
