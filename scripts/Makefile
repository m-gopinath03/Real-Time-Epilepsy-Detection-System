PYTHON = python
INPUT_FOLDER_VIDEOS = "C:/data/videos/seizure C:/data/videos/normal"
OUTPUT_FOLDER_VIDEOS = "C:/data/videos/seizure C:/data/videos/normal"
INPUT_FOLDER_DATA = "C:/data/videos/seizure C:/data/videos/normal"
OUTPUT_FOLDER_DATA = "C:/data/CSV/seizure C:/data/CSV/normal"
INPUT_CSV_FOLDER = "C:/data/CSV/seizure C:/data/CSV/normal"
OUTPUT_CSV_FOLDER = "C:/data/CSV/processed"
OUTPUT_PICKLE_FOLDER = "C:/data/pickle"

.PHONY: all preprocess extract balance label

all: preprocess

preprocess: extract balance label
    @echo "Data preprocessing pipeline completed."

extract:
    @echo "Performing data extraction..."
    $(PYTHON) data_preprocessing.py --input_folder_videos $(INPUT_FOLDER_VIDEOS) --output_folder_videos $(OUTPUT_FOLDER_VIDEOS) --input_folder_data $(INPUT_FOLDER_DATA) --output_folder_data $(OUTPUT_FOLDER_DATA)

balance:
    @echo "Performing data balancing..."
    $(PYTHON) data_preprocessing.py --input_csv_folder $(INPUT_CSV_FOLDER) --output_csv_folder $(OUTPUT_CSV_FOLDER)

label:
    @echo "Performing data labeling..."
    $(PYTHON) data_preprocessing.py --output_pickle_folder $(OUTPUT_PICKLE_FOLDER)
