import pandas as pd


def get_top_genre_id(genres: pd.DataFrame, genre_id: int) -> int:
    top_genre_id = genres[genres['genre_id'] == genre_id]['top_level']
    if top_genre_id is None:
        raise Exception(f'Жанр с id {genre_id} не найден')
    return int(top_genre_id)