"""
Application configuration, constants, sample datasets, and environment variables.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

# Create required directories
for d in [ARTIFACTS_DIR, DATA_DIR, SAMPLE_IMAGES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Pretrained model identifiers
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# VAE settings
VAE_WEIGHTS_PATH = ARTIFACTS_DIR / "vae_fashion_mnist.pth"
VAE_LATENT_DIM = 2
FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Default Sample Text Sentences spanning diverse semantic domains
DEFAULT_TEXT_SAMPLES = [
    # Technology & AI
    "Artificial intelligence and deep neural networks are transforming technology.",
    "Machine learning models learn patterns and representations from data.",
    "Generative adversarial networks synthesize highly realistic synthetic imagery.",
    "Large language models understand context through multi-head self-attention.",
    "Quantum computing leverages superposition and entanglement for computation.",
    # Nature & Astronomy
    "The solar system consists of eight planets orbiting the Sun.",
    "Deep ocean trenches harbor unique bioluminescent marine ecosystems.",
    "Dense tropical rainforests produce oxygen and sustain high biodiversity.",
    "Glaciers in alpine valleys are retreating rapidly due to global warming.",
    "The James Webb Space Telescope observes the earliest galaxies in the universe.",
    # Art & Literature
    "Impressionist painters captured fleeting atmospheric light and vibrant color.",
    "A classical symphony weaves complex harmonic counterpoint and orchestral melody.",
    "Renaissance sculptures celebrated anatomical precision and humanism.",
    "Poetry evokes profound emotional resonance through meter and metaphor.",
    "Modernist literature experiments with stream-of-consciousness narratives.",
    # Sports & Fitness
    "The marathon runner maintained a steady pace across thirty kilometers.",
    "Soccer requires strategic spatial awareness, tactical passing, and endurance.",
    "Olympic gymnasts demonstrate extraordinary balance, strength, and grace.",
    "Swimming engages full-body cardiovascular endurance and breath control.",
    "Basketball teams execute pick-and-roll plays to create open shooting lanes.",
    # Culinary & Food
    "Freshly baked sourdough bread features a crispy crust and airy crumb.",
    "Traditional Italian pizza is baked in a wood-fired brick oven at high heat.",
    "A delicate green tea possesses subtle herbal aromas and antioxidants.",
    "Culinary chefs balance sweet, salty, sour, bitter, and umami flavors.",
    "Dark roasted coffee beans release aromatic oils during grinding.",
]

# Occupation test list for Responsible AI bias evaluation (Unit 5)
OCCUPATIONS_BIAS_DATA = [
    {"occupation": "Software Engineer", "category": "STEM/Tech"},
    {"occupation": "Registered Nurse", "category": "Healthcare/Care"},
    {"occupation": "Aerospace Pilot", "category": "Aviation/Transport"},
    {"occupation": "Primary School Teacher", "category": "Education/Care"},
    {"occupation": "Research Scientist", "category": "STEM/Research"},
    {"occupation": "Fashion Designer", "category": "Creative/Design"},
    {"occupation": "Corporate Executive / CEO", "category": "Leadership/Business"},
    {"occupation": "Early Childhood Caregiver", "category": "Caregiving"},
    {"occupation": "Mechanical Engineer", "category": "Engineering/STEM"},
    {"occupation": "Dental Hygienist", "category": "Healthcare/Support"},
]
