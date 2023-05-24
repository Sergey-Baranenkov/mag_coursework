# Курсовая работа магистратуры Интеллектуальный анализ данных

Реализует определение метаданных аудиозаписи, таких как:

1. Название и автор (берутся из метаданных песни)
2. Жанр (нейросеть)
3. Музыкальные инструменты (нейросеть)
4. Год (декада) выпуска (нейросеть)
5. Разбиение дорожки на 5 составляющих (deezer spleeter)
6. bpm (librosa)
7. Тональность (алгоритм Krumhansl & Kessler’s)

Репозиторий содержит backend на flask и frontend на React.js, а также jupyter notebook`s с EDA, feature extracting и model training для нейросетей.

Нейросети были обучены на датасете [fma](https://github.com/mdeff/fma), причем часть с распознаванием музыкальных инструментов была обучена на [подвыборке](https://github.com/cosmir/openmic-2018)

Демонстрация работы алгоритма:

https://github.com/Sergey-Baranenkov/mag_coursework/assets/50075840/f450b0be-c8f4-4b00-b8ad-637b1884edfb

