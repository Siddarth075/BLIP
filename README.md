# BLIP Project

A Flask-based web application featuring speech processing and image upload capabilities.

## 🛠️ Project Structure

<img width="399" height="510" alt="Screenshot 2026-07-07 193244" src="https://github.com/user-attachments/assets/7c418659-c5a4-4f86-872e-73f6d3fb18bb" />


Based on the repository layout, here is the organizational structure:

*   `app.py` - Main Flask application routing and logic.
*   `templates/` - HTML files (`index.html`, `history.html`) rendering the frontend views.
*   `static/` - Static assets including custom stylesheets (`css/style.css`) and JavaScript files (`js/speech.js`).
*   `uploads/` - Directory for storing uploaded media assets.
*   `history.db` - Local database tracking application history.
*   `requirements.txt` - Python package dependencies.

## 🚀 Features

*   **Speech Integration:** Frontend audio/speech capabilities powered by `speech.js`.
*   **Image Management:** Secure file upload handling and tracking.
*   **Persistent History:** Local data tracking managed via `history.db`.

## 🏁 Getting Started

### Prerequisites

*   Python 3.x
*   `pip` (Python package installer)

### Installation


1. Clone this repository to your local machine:
   ```bash
   git clone <your-repository-url>
   cd BLIP_PROJECT
