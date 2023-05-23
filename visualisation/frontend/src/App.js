import './App.css';
import TextComponent from "./components/TextComponent/TextComponent";
import WaveformComponent from "./components/WaveformComponent/WaveformComponent";
import DragNDropComponent from "./components/DragNDropComponent/DragNDropComponent";
import ListComponent from "./components/ListComponent/ListComponent";
import {useEffect, useState} from "react";
import Requester from "./Requester";

function App() {
  const [file, setFile] = useState(null);
  const [metadata, setMetadata] = useState({
      bpm: 'unknown',
      tonality: 'unknown',
      metadata: {
          artist: 'unknown',
          duration: 'unknown',
          title: 'unknown',
      },
      decade: 'unknown',
      genre: 'unknown',
      instruments: [],
      deezer: null,
  })
  const requester = new Requester()
  useEffect(() => {
      if (file !== null) {
          console.log(file);
          requester.getAll(file)
              .then(result => {
                  setMetadata(result);
              });
      }
  }, [file])

  return (
    <div className="App">
        <DragNDropComponent handleChange={(file) => setFile(file)}/>
        <div className={'text-container'}>
            <TextComponent type={'Title'} text={metadata.metadata.title}/>
            <TextComponent type={'Author'} text={metadata.metadata.artist}/>

            <TextComponent type={'BPM'} text={metadata.bpm}/>
            <TextComponent type={'Tonality'} text={metadata.tonality}/>

            <TextComponent type={'Genre'} text={metadata.genre}/>
            <TextComponent type={'Decade'} text={metadata.decade}/>
        </div>

        <ListComponent elements={metadata.instruments.map(el => ({text: el}))}></ListComponent>

        { file && <WaveformComponent title={'Original track'} id={'original-track'} file={file}/> }

        {metadata.deezer &&
            <div className={'deezer-container'}>
                <WaveformComponent title={'Bass'} id={'bass-track'} file={metadata.deezer.bass_filename}/>
                <WaveformComponent title={'Drums'} id={'drums-track'} file={metadata.deezer.drums_filename}/>
                <WaveformComponent title={'Other'} id={'other-track'} file={metadata.deezer.other_filename}/>
                <WaveformComponent title={'Piano'} id={'piano-track'} file={metadata.deezer.piano_filename}/>
                <WaveformComponent title={'Vocal'} id={'vocal-track'} file={metadata.deezer.vocals_filename}/>
            </div>
        }
    </div>
  );
}

export default App;
