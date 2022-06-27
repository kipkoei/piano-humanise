from collections import defaultdict
from turtle import st
import pandas as pd
import os
import shutil
from pathlib import Path

# This script renames the individual midi files of the MAESTRO dataset to use the more common '.mid' extension and prepends a letter indicating the period to 
# each file name. Furthermore, it places the MIDI files in a folder corresponding to their train/validation/test split.

df = pd.read_csv('data/maestro-v3.0.0/maestro-v3.0.0.csv')

composers = {
    'Alban Berg': 'm',
    'Alexander Scriabin': 'm',
    'Antonio Soler': 'b',
    'Carl Maria von Weber': 'r',
    'Charles Gounod / Franz Liszt': 'l',
    'Claude Debussy': 'm',
    'César Franck': 'l',
    'Domenico Scarlatti': 'b',
    'Edvard Grieg': 'r',
    'Felix Mendelssohn': 'r',
    'Felix Mendelssohn / Sergei Rachmaninoff': 'l',
    'Franz Liszt': 'l',
    'Franz Liszt / Camille Saint-Saëns': 'l',
    'Franz Liszt / Vladimir Horowitz': 'l',
    'Franz Schubert': 'r',
    'Franz Schubert / Franz Liszt': 'r',
    'Franz Schubert / Leopold Godowsky': 'r',
    'Fritz Kreisler / Sergei Rachmaninoff': 'l',
    'Frédéric Chopin': 'r',
    'George Enescu': 'm',
    'George Frideric Handel': 'b',
    'Georges Bizet / Ferruccio Busoni': 'r',
    'Georges Bizet / Moritz Moszkowski': 'r',
    'Georges Bizet / Vladimir Horowitz': 'r',
    'Giuseppe Verdi / Franz Liszt': 'r',
    'Henry Purcell': 'b',
    'Isaac Albéniz': 'm',
    'Isaac Albéniz / Leopold Godowsky': 'm',
    'Jean-Philippe Rameau': 'b',
    'Johann Christian Fischer / Wolfgang Amadeus Mozart': 'c',
    'Johann Pachelbel': 'b',
    'Johann Sebastian Bach': 'b',
    'Johann Sebastian Bach / Egon Petri': 'b',
    'Johann Sebastian Bach / Ferruccio Busoni': 'b',
    'Johann Sebastian Bach / Franz Liszt': 'b',
    'Johann Sebastian Bach / Myra Hess': 'b',
    'Johann Strauss / Alfred Grünfeld': 'b',
    'Johannes Brahms': 'r',
    'Joseph Haydn': 'c',
    'Leoš Janáček': 'm',
    'Ludwig van Beethoven': 'c',
    'Mikhail Glinka / Mily Balakirev': 'm',
    'Mily Balakirev': 'm',
    'Modest Mussorgsky': 'm',
    'Muzio Clementi': 'c',
    'Niccolò Paganini / Franz Liszt': 'l',
    'Nikolai Medtner': 'm',
    'Nikolai Rimsky-Korsakov / Sergei Rachmaninoff': 'm',
    'Orlando Gibbons': 'b',
    'Percy Grainger': 'm',
    'Pyotr Ilyich Tchaikovsky': 'l',
    'Pyotr Ilyich Tchaikovsky / Mikhail Pletnev': 'l',
    'Pyotr Ilyich Tchaikovsky / Sergei Rachmaninoff': 'l',
    'Richard Wagner / Franz Liszt': 'l',
    'Robert Schumann': 'r',
    'Robert Schumann / Franz Liszt': 'r',
    'Sergei Rachmaninoff': 'l',
    'Sergei Rachmaninoff / György Cziffra': 'l',
    'Sergei Rachmaninoff / Vyacheslav Gryaznov': 'l',
    'Wolfgang Amadeus Mozart': 'c'
}

counts = defaultdict(int)


for split in df['split'].unique():
    path = f'data/{split}'
    if os.path.exists(path):
        shutil.rmtree(path)

    os.mkdir(path)

for name, split, composer in zip(df['midi_filename'], df['split'], df['canonical_composer']):
    name = str(name)
    style = composers[composer]
    counts[style] += 1
    src = 'data/maestro-v3.0.0/' + name
    # Change the file extension to .mid
    newname = name.split('/')[-1].rstrip('i')
    dst = f'data/{split}'

    shutil.copy(src, dst)
    p = Path(dst, name.split('/')[-1])
    p.rename(p.with_name(style + '_' + p.name).with_suffix('.mid'))

print(counts)