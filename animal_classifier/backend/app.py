import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

app = Flask(__name__)
CORS(app)

# Load the pre-trained ResNet50 model
# This model can identify 1000 different categories, including many animals
model = ResNet50(weights='imagenet')

# A list of keywords to help identify if a category is an animal
ANIMAL_KEYWORDS = [
    'dog', 'cat', 'bird', 'fish', 'snake', 'lizard', 'spider', 'insect', 
    'mammal', 'primate', 'rodent', 'ungulate', 'carnivore', 'bear', 
    'elephant', 'whale', 'dolphin', 'shark', 'frog', 'toad', 'turtle',
    'crocodile', 'alligator', 'dinosaur', 'butterfly', 'moth', 'bee',
    'ant', 'beetle', 'crab', 'lobster', 'snail', 'slug', 'shell',
    'monkey', 'ape', 'gorilla', 'chimpanzee', 'orangutan', 'baboon',
    'macaque', 'lemur', 'gibbon', 'marmoset', 'tamarin', 'squirrel',
    'beaver', 'porcupine', 'hamster', 'guinea_pig', 'mouse', 'rat',
    'rabbit', 'hare', 'deer', 'elk', 'moose', 'antelope', 'gazelle',
    'giraffe', 'camel', 'llama', 'alpaca', 'pig', 'boar', 'hippo',
    'rhino', 'horse', 'zebra', 'donkey', 'mule', 'cow', 'ox', 'bull',
    'sheep', 'goat', 'bison', 'buffalo', 'lion', 'tiger', 'leopard',
    'cheetah', 'jaguar', 'panther', 'cougar', 'lynx', 'wolf', 'fox',
    'coyote', 'jackal', 'hyena', 'badger', 'otter', 'skunk', 'raccoon',
    'panda', 'koala', 'kangaroo', 'wallaby', 'wombat', 'possum',
    'platypus', 'echidna', 'armadillo', 'sloth', 'anteater', 'bat',
    'seal', 'walrus', 'penguin', 'ostrich', 'emu', 'cassowary', 'kiwi',
    'eagle', 'hawk', 'falcon', 'owl', 'parrot', 'macaw', 'cockatoo',
    'toucan', 'hummingbird', 'woodpecker', 'kingfisher', 'swan', 'goose',
    'duck', 'chicken', 'turkey', 'peacock', 'pheasant', 'pigeon', 'dove',
    'crane', 'heron', 'stork', 'flamingo', 'pelican', 'gull', 'tern',
    'puffin', 'albatross', 'petrel', 'penguin', 'rottweiler', 'terrier', 
    'retriever', 'spaniel', 'shepherd', 'collie', 'hound', 'mastiff', 
    'setter', 'pointer', 'bulldog', 'poodle', 'tabby', 'siamese', 'persian', 
    'sphynx', 'maine coon', 'bengal', 'beagle', 'boxer', 'chihuahua', 
    'dachshund', 'dalmatian', 'great dane', 'husky', 'malamute', 'pug', 
    'shih tzu', 'corgi', 'pomeranian', 'maltese', 'yorkshire', 'labrador',
    'golden retriever', 'german shepherd', 'doberman', 'schnauzer',
    'cat', 'dog', 'kitten', 'puppy', 'feline', 'canine'
]

def is_animal(label):
    """Check if the label contains any animal-related keywords."""
    label_lower = label.lower().replace('_', ' ')
    return any(keyword in label_lower for keyword in ANIMAL_KEYWORDS)

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Open and prepare the image
        img = Image.open(file.stream)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to 224x224 as required by ResNet50
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = preprocess_input(img_array)
        
        # Get predictions
        preds = model.predict(img_preprocessed)
        decoded = decode_predictions(preds, top=5)[0]
        
        # Format the results
        results = []
        for _, label, score in decoded:
            results.append({
                "name": label.replace('_', ' ').title(),
                "confidence": float(score) * 100
            })
        
        # The top result is the first one in the decoded list
        top_result = results[0]
        
        return jsonify({
            "predictions": results,
            "top_result": top_result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
