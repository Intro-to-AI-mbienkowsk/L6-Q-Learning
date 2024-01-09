# WSI 2023 LAB6 - Uczenie ze wzmocnieniem
### Maksym Bieńkowski

## Zawartość archiwum
## `/src`
* `Agent.py` - klasa abstrakcyjna reprezentująca agenta zadania RL i klasa FrozenLakeAgent
implementująca ten interfejs dla zadania FrozenLake
* `constants.py` - stałe używane w obliczeniach, domyślne argumenty dla parametrów

## Uruchamialne skrypty
### `main.py`
przyjmuje następujące parametry: 
* `-lr [float]` - learning rate agenta
* `-e [int]` - liczba epizodów treningowych
* `-d [float]` - współczynnik dyskonta
* `-ed [float]` - po jakiej części epizodów agent ma zminimalizować epsilon, skupiając się na eksploatacji
* `-s {4/8}` - rozmiar planszy
* `-rs (a, b, c)` - system nagród, a za przejście do prezentu; b za wpadnięcie w dziurę; c w każdej innej sytuacji
* `-slip {0/1}` - włączony (1) lub wyłączony (0) poślizg

Wszystkie argumenty mają domyślne wartości, więc możemy uruchomić skrypt poprzez
```shell
python3 -m main
```
lub, określając argumenty:
```shell
python3 -m main -lr 0.01 -e 1000 -d 0.87 -ed 0.4 -s 4 -slip 0
```
## Krótki opis rozwiązania
Implementacja algorytmu Q-Learning i uruchamialny skrypt z parametrami do dostosowania. 
Analiza wpływu wartości parametrów na wyniki w sprawozdaniu.