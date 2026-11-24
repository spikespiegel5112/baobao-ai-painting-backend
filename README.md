# baobao-ai-painting-backend

## About the Project

This project is an AI painting backend service built with Django. It provides image generation, task queueing, and related data management APIs for painting applications. The service receives client requests over HTTP and processes them through generation controllers for Stable Diffusion models and Flux.

### Features

- **AI image generation**: Provides text-to-image APIs for SDXL, Stable Diffusion 3, and Flux, with support for queueing tasks, checking generation status, and canceling tasks.
- **Image processing**: Supports image uploads, saving Base64-encoded images, adding generated works to the database, and SDXL-based inpainting.
- **Painting data management**: Provides management APIs for works, models, prompts, prompt categories, dictionaries, and logs.
- **Users and authentication**: Provides user information management and authentication APIs.
- **Additional utilities**: Includes queue status checks and text translation APIs.

### Technology Stack

- Python and Django
- SQLAlchemy for data models and database connectivity
- Hugging Face Diffusers, Transformers, PyTorch, and other model inference components

### Project Structure

```text
.
├── manage.py                    # Django command-line entry point
├── stableDiffusion/
│   ├── settings.py               # Django project settings
│   ├── urls*.py                  # API routes
│   ├── controllers/              # Controllers for generation, users, images, and data management
│   └── models/                   # Data models
├── requirements.txt              # Python dependencies
├── requirements_macos.txt        # Dependencies for macOS
└── requirements_linux.txt        # Dependencies for Linux
```

## Running in PyCharm

1. Open the project root directory in PyCharm.
2. Go to **Run > Edit Configurations…**, click **+**, and select **Python**.
3. Configure the run settings:
   - **Name**: `Django runserver`
   - **Script path**: `manage.py` in the project root directory
   - **Parameters**: `runserver`
   - **Working directory**: the project root directory
   - **Python interpreter**: the interpreter in the project virtual environment, for example `venv/bin/python` (`venv\Scripts\python.exe` on Windows)
4. Click **Apply** and **OK**, then run this configuration.

Alternatively, run this command in the PyCharm Terminal:

```bash
venv/bin/python manage.py runserver
```

On Windows:

```powershell
venv\Scripts\python.exe manage.py runserver
```
