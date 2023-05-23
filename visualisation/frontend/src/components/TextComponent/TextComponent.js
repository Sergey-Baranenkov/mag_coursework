import './TextComponent.css';

function TextComponent({ type, text }) {
    return (
        <div className='text-component'>
            <p className={'text-component_header'}>{type}</p>
            <hr></hr>
            <p className={'text-component_header'}>{text}</p>
        </div>
    );
}

export default TextComponent;
