from flask import Flask, request, jsonify
import replicate
import os
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv()

# Récupérer la clé API Replicate à partir de la variable d'environnement
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Configurer la bibliothèque Replicate pour utiliser la clé API
replicate.Client(api_token=REPLICATE_API_TOKEN)

app = Flask(__name__)

# Route pour l'analyse de l'image et la réponse à la question
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json  # Récupérer les données au format JSON

    # Vérification des champs dans la requête
    if 'image_url' not in data or 'question' not in data:
        return jsonify({"error": "Missing image_url or question"}), 400

    # Récupération des données de l'utilisateur
    image_url = data['image_url']
    question = data['question']

    # Définir l'entrée pour l'API Replicate
    input = {
        "image": image_url,
        "question": question
    }

    # Exécuter le modèle sur Replicate
    try:
        output = replicate.run(
            "andreasjansson/blip-2:f677695e5e89f8b236e52ecd1d3f01beb44c34606419bcc19345e046d8f786f9",
            input=input
        )
        return jsonify({"answer": output}), 200  # Retourner la réponse au format JSON
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Démarrer le serveur Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
