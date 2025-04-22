# Amico - Your Life Long Personal AI Assistant

## Description

Amico is an intelligent personal AI assistant prototype designed to overcome the limitations of standard stateless assistants. It features persistent memory, allowing it to remember user preferences and recall past conversations locally on the user's device using SQLite. This stored context is leveraged alongside a fine-tuned OpenAI GPT-4 model to provide more natural, contextually relevant, and personalized interactions compared to assistants that treat every interaction independently.

Amico integrates several technologies, including natural language processing, speech recognition, text-to-speech, local database management, and external API communication (for real-time data like weather, news, and stocks). It aims to explore a privacy-conscious, local-first approach while acknowledging the reliance on cloud services for core AI functionalities.

## Features

*   **Persistent Memory:** Stores conversation history and embeddings locally in a SQLite database.
*   **Contextual Awareness:** Retrieves relevant past interactions using an algorithm considering recency, semantic similarity (via text embeddings), and importance (message length heuristic).
*   **Natural Conversation:** Utilizes a fine-tuned OpenAI GPT-4 model via API for understanding and generating human-like responses.
*   **Voice Interaction:**
    *   Speech-to-Text (STT): Uses the `SpeechRecognition` library (configured for Google's online service).
    *   Text-to-Speech (TTS): Uses the `pyttsx3` library for offline voice output.
*   **External Information:** Integrates with external APIs to fetch and present:
    *   Weather forecasts (via WeatherAPI.com)
    *   Top news headlines (via GNews)
    *   Stock market data (via Financial Modeling Prep)
*   **API Caching & Retries:** Implements basic caching and retry mechanisms for external API calls to improve reliability and reduce redundancy.
*   **User Profile Awareness:** Stores and utilizes basic user profile information (e.g., name, location) if available.
*   **Performance Logging:** Logs key performance metrics (latency, memory usage) to a CSV file for analysis.
*   **Modular Architecture:** Built with a class-based structure for better organization and maintainability.

## Technologies Used

*   **Core Language:** Python 3.x
*   **AI & NLP:**
    *   OpenAI API (GPT-4 Fine-Tuned Model, `text-embedding-ada-002` for embeddings)
*   **Database:** SQLite (via Python's built-in `sqlite3` module)
*   **Speech:**
    *   `pyttsx3` (Offline TTS)
    *   `SpeechRecognition` (STT - requires internet for Google service)
*   **External APIs:** `requests` library
*   **Configuration:** `python-dotenv`
*   **System Monitoring:** `psutil`

## Setup Instructions

1. Clone the Repository

```bash
https://github.com/Cristian070297/Amico-the-personal-AI-assistant.git
cd amico-personal-ai-assistant

2. Install Dependencies
Ensure you have Python 3 installed. Then, install the required packages using pip:
pip install -r requirements.txt

3. Environment Variables
Amico requires API keys and configuration stored in an environment file.
Create a file named .env in the root directory of the project (amico_project/).
Add the following variables to the .env file, replacing your_..._key_here with your actual secret keys/IDs:
OPENAI_API_KEY=your_openai_api_key_here
fine_tuned_model_id=your_openai_fine_tuned_model_id_here
GNEWS-API=your_gnews_api_key_here
FMP-FINANCE-API=your_fmp_api_key_here
weather-api_key=your_weatherapi_com_key_here

How to Run
Navigate to the project's root directory in your terminal and run the main script:
python run_amico.py

Amico will initialize, load the user profile (if any), greet you, and start listening for voice commands.

Project Context
This project was developed as part of the Final Year Project for
the BEng (Hons) Computer Science program at Anglia Ruskin University (ARU).

Known Limitations / Future Work
This prototype demonstrates core concepts but has areas for improvement:
Text-Embedding Performance: Storing embeddings as text in SQLite causes a performance bottleneck during retrieval,
especially with long histories. Future work could explore binary storage, quantization, or lightweight local vector databases.
Local Data Security: The SQLite database is currently not encrypted at rest, posing a
 security risk if the user's device is compromised. Implementing encryption is a crucial next step.
Online Dependencies: Core functions like STT (Google), embeddings, and LLM generation (OpenAI) require an active internet connection.
Exploring robust offline STT solutions is a key area for future development.
TTS Naturalness: The offline pyttsx3 library is functional but less natural-sounding than commercial cloud-based TTS systems.
User Memory Management: Currently lacks a user interface for viewing, managing, or deleting stored conversation history.
STT Robustness: Speech recognition accuracy degrades significantly in noisy environments.
Future directions focus on addressing these limitations, particularly enhancing memory efficiency,
 improving speech interaction (offline options), strengthening local data security (encryption), and providing user control over stored data.
