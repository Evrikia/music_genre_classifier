# Music Genre Classification with CatBoost

This project uses a **CatBoost** model to classify the genre of a song based on audio features extracted from a `.wav` file.

The model can predict the following genres:

| Label | Genre     |
| ----- | --------- |
| 0     | blues     |
| 1     | classical |
| 2     | country   |
| 3     | disco     |
| 4     | hiphop    |
| 5     | jazz      |
| 6     | metal     |
| 7     | pop       |
| 8     | reggae    |
| 9     | rock      |

---

## Project Structure

```
.
├── music_genre_model.cbm    # Trained CatBoost model
├── feature_extraction.py    # Script to extract features from a .wav file
├── predict.py               # Script to load model & predict genre from command line
├── requirements.txt         # Required dependencies
└── README.md
```

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Evrikia/music-genre-classifier.git
cd music-genre-classifier
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## How to Predict Genre from a WAV File

Run the prediction script with your `.wav` file as an argument:

```bash
python3 predict.py path/to/your_song.wav
```

---

## Feature Extraction Details

The model uses these audio features (in this order):

1. MFCC mean (20 values)
2. MFCC standard deviation (20 values)
3. Chroma STFT mean (12 values)
4. Spectral centroid mean (1 value)
5. Spectral bandwidth mean (1 value)
6. Spectral rolloff mean (1 value)
7. Spectral contrast mean (7 values)
8. Tonnetz mean (6 values)
9. Zero Crossing Rate mean (1 value)
10. RMS energy mean (1 value)
11. Tempo (1 value)

Total feature vector length: **71**



## License

This project is licensed under the MIT License.
