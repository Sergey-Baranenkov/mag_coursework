import './DragNDropComponent.css';
import {useState} from "react";
import {FileUploader} from "react-drag-drop-files";
const jsmediatags = window.jsmediatags

// Читаются таги только у мп3
const fileTypes = ["mp3", "wav"];
function DragNDropComponent({handleChange}) {
    return (
        <FileUploader
            classes={'dragndrop'}
            handleChange={handleChange}
            name="file"
            types={fileTypes}
        />
    );
}

export default DragNDropComponent;
