import './WaveformComponent.css';
import WaveSurfer from 'wavesurfer.js'
import {useEffect, useRef, useState} from "react";



function WaveformComponent({title, id, file}) {
    const waveformRef = useRef(null);
    const [isPlaying, setIsPlaying] = useState(false);

    useEffect(() => {
        setIsPlaying(false)
        waveformRef.current = WaveSurfer.create({
            container: `#${id}`,
            waveColor: '#EFEFEF',
            progressColor: '#2D5BFF',
            barWidth: 3,
            cursorWidth: 3,
            cursorColor: 'transparent',
            hideScrollbar: true,
        });
        if (typeof file === 'string') {
            waveformRef.current.load(file);
        } else {
            waveformRef.current.loadBlob(file)
        }


        return () => {
            waveformRef.current.destroy();
        }
    }, [file])

    const handlePlay = async () => {
        setIsPlaying(!isPlaying)
        await waveformRef.current.playPause()
    }

    return (
        <div className={'waveform-container'}>
            <p className={'waveform-label'}>{title}</p>
            <div className={'waveform-container-row'}>
                <div className={`playpause ${!isPlaying ? 'play' : ''}`} onClick={handlePlay}/>
                <div className={'waveform'} id={id}/>
            </div>
        </div>

    );
}

export default WaveformComponent;
