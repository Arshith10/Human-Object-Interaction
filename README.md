# Human-Object Interaction Detection

This project focuses on detecting and classifying interactions between humans and objects using deep learning techniques. 

## Project Structure
- `app.py`: Main application entry point (likely a web interface or GUI).
- `model.py`: Defines the deep learning model architecture.
- `inference.py`: Script to run the model on new data and generate predictions.
- `config.py`: Configuration parameters and settings for the project.
- `interaction_prior_database_qwen.pt`: Pre-trained weights/database for the interaction priors.
- `*list.json`: JSON files (`hoi_list.json`, `object_list.json`, `verb_list.json`) containing the vocabulary of objects, verbs, and interactions the model can recognize.

## Note on Large Files
The trained model weights file (`best.pt`) is required to run the full inference but is not included in this repository due to GitHub's file size limits. Please ensure you place `best.pt` in the root directory before running the application.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Arshith10/Human-Object-Interaction.git
   cd Human-Object-Interaction
   ```

2. Run the application:
   ```bash
   python app.py
   ```

*(Note: Update this section with any specific installation requirements like `pip install -r requirements.txt` if you add one later!)*
