import cloudinary
import cloudinary.uploader
import cloudinary.api

# Import the CloudinaryImage and CloudinaryVideo methods for the simplified syntax used in this guide
from cloudinary import CloudinaryImage
from cloudinary import CloudinaryVideo
from cloudinary_uploader import CloudinaryUploader
from dotenv import load_dotenv
import os       

# Load environment variables from .env file
load_dotenv()
