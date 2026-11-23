'''
Archivo para iniciar las aplicaciones de entrenamiento

Usar

    python main.py analisa
    python main.py train simple
    python main.py train complex
    python main.py test simple
    python main.py test complex

'''

import argparse
import subprocess
import sys

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "Dataset"

ANALISE_DIR = BASE_DIR / "dataset_analises"

TRAIN_AND_TEST_SIMPLE_DIR = BASE_DIR / "Test_&_Train" / "Simple"

TRAIN_AND_TEST_COMPLEX_DIR = BASE_DIR / "Test_&_Train" / "Complex"

MODES = ["simple", "complex"]


def run_script(script):
    '''Ejecuta un script de Python y le pasa la ruta del dataset como argumento.'''
    if not script.exists():
        sys.exit(f"Error: no se encuentra el archivo {script}")

    print(f"Ejecutando {script.relative_to(BASE_DIR)} ...")
    subprocess.run([sys.executable, str(script), str(DATA_DIR)], check=True)


def get_mode_dir(mode):
    if mode == "simple":
        return TRAIN_AND_TEST_SIMPLE_DIR
    return TRAIN_AND_TEST_COMPLEX_DIR


def analise():
    run_script(ANALISE_DIR / "analise.py")


def train(mode):
    run_script(get_mode_dir(mode) / "train.py")


def test(mode):
    run_script(get_mode_dir(mode) / "test.py")


def pars_build():
    parser = argparse.ArgumentParser(description="Analizar el dataset, entrenar o testear un modelo.")
    parser.add_argument("action", choices=["analisa", "train", "test"])
    parser.add_argument("mode", choices=MODES, nargs="?", default="simple")
    return parser.parse_args()


def main():
    args = pars_build()

    action = args.action
    mode = args.mode

    if action == "analisa":
        analise()
    elif action == "train":
        train(mode)
    elif action == "test":
        test(mode)


if __name__ == "__main__":
    main()
