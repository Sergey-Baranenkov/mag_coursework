import audio_metadata
import numpy as np
from flask import Flask, send_file, request
from flask_json import FlaskJSON, json_response
import uuid
from alhorithms.predict_bpm import predict_bpm
from alhorithms.predict_tonalty import krumhansl_schmuckler
from alhorithms.apply_deezer_spleeter import apply_deezer_spleeter
from models.vggish.extract_vggish_features import extract_vggish_features
from models.decade_classification.predict_decade import predict_decade
from models.genre_classification.predict_genre import predict_genre
from models.instrument_classification.predict_instruments import predict_instruments
from flask_cors import CORS

app = Flask(__name__)
json = FlaskJSON(app)
CORS(app)

@app.route("/bpm", methods=['GET'])
def get_bpm():
    filename = request.args.get('filename')
    bpm = predict_bpm(f'audio/{filename}')
    return json_response(data={'bpm': int(bpm)})


@app.route("/tonality", methods=['GET'])
def get_tonality():
    filename = request.args.get('filename')
    pitch, key, probability = krumhansl_schmuckler(f'audio/{filename}')
    return json_response(data={'tonality': f'{pitch}-{key}'})


@app.route("/deezer", methods=['GET'])
def deezer_split():
    filename = request.args.get('filename')
    apply_deezer_spleeter(f'audio/{filename}')
    path = f'audio_dir/tmp/{filename.split("/")[-1].split(".")[0]}'
    return json_response(data={
        'bass_filename': f'{path}/bass.wav',
        'drums_filename': f'{path}/drums.wav',
        'other_filename': f'{path}/other.wav',
        'piano_filename': f'{path}/piano.wav',
        'vocals_filename': f'{path}/vocals.wav',
    })


@app.route("/audio_dir/<path:path>", methods=['GET'])
def get_static(path):
    return send_file(f'audio/{path}')


@app.route("/decade", methods=['GET'])
def get_decade():
    filename = request.args.get('filename')
    features = extract_vggish_features(f'audio/{filename}')
    features = np.mean(features, axis=0, keepdims=False)
    decade = predict_decade(features)
    return json_response(data={'decade': decade})


@app.route("/genre", methods=['GET'])
def get_genre():
    filename = request.args.get('filename')
    features = extract_vggish_features(f'audio/{filename}')
    features = np.mean(features, axis=0, keepdims=False)
    genre = predict_genre(features)
    return json_response(data={'genre': genre})


@app.route("/instruments", methods=['GET'])
def get_instruments():
    filename = request.args.get('filename')
    features = extract_vggish_features(f'audio/{filename}')
    features = np.mean(features, axis=0, keepdims=False)
    instruments = predict_instruments(features)
    return json_response(data={'instruments': instruments})


@app.route("/upload_file", methods=['POST'])
def upload_file():
    file = request.files['file']
    if not file:
        return 'no file provided', 400

    ext = file.filename.split('.')[-1]
    if ext != 'mp3' and ext != 'wav':
        return 'incorrect file extension', 400

    new_filename = uuid.uuid4()
    file.save(f'audio/uploaded/{new_filename}.{ext}')

    return json_response(data={'file_path': f'uploaded/{new_filename}.{ext}'})


@app.route("/get_metadata", methods=['GET'])
def get_metadata():
    filename = request.args.get('filename')
    metadata = audio_metadata.load(f'audio/{filename}')
    duration = metadata['streaminfo']['duration']
    artist = metadata['tags'].get('artist')
    artist = artist[0] if artist is not None else 'unknown'
    title = metadata['tags'].get('title')
    title = title[0] if title else 'unknown'

    return json_response(data={'duration': int(duration), 'artist': artist, 'title': title})


if __name__ == "__main__":
    app.run(host='localhost', port=8888)
