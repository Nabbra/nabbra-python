# Meet Nabbra AI !

Provides APIs and WebSocket support for audiogram processing, including reading audiograms and applying gain adjustments to audio files. It integrates AI-powered models to extract audiogram features and manipulate audio frequency ranges.

## Features

- AI-powered Audiogram Reader:
    - Detects and processes audiogram images using YOLO models.
    - Extracts data for left and right ears.

- Audio Amplification:
    - Adjusts frequency ranges with customizable gain.
    - Supports FFT-based audio processing.

## Installation

### Prerequisites

- Python 3.12+
- GPU support (for YOLO and EasyOCR)

1. Clone the repository:

```bash
git clone https://github.com/Nabbra/nabbra-api
cd nabbra-api
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run application!

```bash
python ./app/nabbra.py run
```

## Deployment

We use [Modal](https://modal.com) as a server-less ai cloud serve, head to their documentation to learn more.

For local serving:

```bash
modal serve ./deployment/modal_app.py
```

For production deployments:

```bash
modal deploy ./deployment/modal_app.py
```