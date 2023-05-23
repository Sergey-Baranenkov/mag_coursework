import './ListComponent.css';

function ListComponent({elements}) {
    return (
        <div className={'list-component'}>
            <p className={'list-component-label'}>Musical instrument list</p>
            <ul className='list-component-ul'>
                {elements.map(el => <li key={el.text}>{el.text}</li>)}
            </ul>
        </div>

    );
}

export default ListComponent;
