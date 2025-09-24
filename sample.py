# Function to load subject documents based on paper type
def load_documents(subject, paper_type):
    subject_folder_map = {
        "Agriculture Science and Technology": "agriculturescienceandtechnology",
        "Social Science": "socialscience",
        "English": "english"
    }
    subject_path = os.path.join(BASE_RESOURCE_PATH, subject_folder_map.get(subject, subject.lower().replace(" ", "")))
    selected_folders = []

    if subject == "Social Science":
        if paper_type == "Paper 1":
            selected_folders.append("paper1")
        elif paper_type == "Paper 2":
            selected_folders.append("paper2")
    selected_folders.append("textbook")

    docs = []
    for folder in selected_folders:
        full_path = os.path.join(subject_path, folder)
        if os.path.exists(full_path):
            loader = DirectoryLoader(full_path)
            docs.extend(loader.load())
    return docs
