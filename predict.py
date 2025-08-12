import sys
import catboost
import numpy as np
from feature_extraction import extract_features

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 predict.py <path_to_wav_file>")
        sys.exit(1)

    wav_file = sys.argv[1]

    # Load model
    model = catboost.CatBoostClassifier()
    model.load_model("music_genre_classifier.cbm")

    # Genre mapping
    mapping = {
        0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
        5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
    }

    # Extract features and reshape for prediction
    features = extract_features(wav_file).reshape(1, -1)

    # Predict
    prediction = model.predict(features)[0]
    print(f"Predicted genre: {mapping[int(prediction)]}")

if __name__ == "__main__":
    main()