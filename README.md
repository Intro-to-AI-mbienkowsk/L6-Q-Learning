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
* `-on_win [float]` - nagroda za dotarcie do prezentu
* `-on_lose [float]` - nagroda za wpadnięcie w dziurę
* `-on_else [float]` - nagroda za każdy inny nieterminujący ruch
* `-slip {0/1}` - włączony (1) lub wyłączony (0) poślizg
* `-visualize` - wyświetla wizualizację tablicy q-values po zakończeniu treningu
* `-display` - wyświetla środowisko i uruchamia 3 epizody, aby obejrzeć działanie modelu
* `-goal_sum` - wyświetla wykres zmiany sumy funkcji celu w czasie

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