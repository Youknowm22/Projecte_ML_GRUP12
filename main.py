'''
Archivo para iniciar las aplicaciones de análisis, entrenamiento y test.

Uso:

    python main.py analisa
    python main.py train simple
    python main.py train complex
    python main.py test simple
    python main.py test complex

Funciones:
    run_script(script)
        Ejecuta un script de Python en un subproceso y le pasa la ruta del dataset.
    get_mode_dir(mode)
        Devuelve la carpeta de train/test correspondiente al modo (simple o complex).
    analise()
        Lanza el script de análisis del dataset.
    train(mode)
        Lanza el script de entrenamiento del modo indicado.
    test(mode)
        Lanza el script de test del modo indicado.
    pars_build()
        Define y lee los argumentos de la línea de comandos (acción y modo).
    main()
        Lee los argumentos y llama a la función correspondiente.
'''

import argparse
import subprocess
import sys

from pathlib import Path

# Carpeta donde está este archivo; todas las rutas se construyen a partir de ella
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "Dataset"

ANALISE_DIR = BASE_DIR / "Dataset_analise"

TRAIN_AND_TEST_SIMPLE_DIR = BASE_DIR / "Test_&_Train" / "Simple"

TRAIN_AND_TEST_COMPLEX_DIR = BASE_DIR / "Test_&_Train" / "Complex"

MODES = ["simple", "complex"]


def run_script(script):
    '''
    Ejecuta un script de Python en un subproceso y le pasa la ruta del dataset
    (DATA_DIR) como primer argumento.

    Parámetros:
        script (Path): ruta del script a ejecutar.

    Si el archivo no existe, termina el programa con un mensaje de error.
    Si el script falla, se lanza subprocess.CalledProcessError (check=True).
    '''
    if not script.exists():
        sys.exit(f"Error: no se encuentra el archivo {script}")

    print(f"Ejecutando {script.relative_to(BASE_DIR)} ...")
    
    subprocess.run([sys.executable, str(script), str(DATA_DIR)], check=True)


def get_mode_dir(mode):
    '''
    Devuelve la carpeta de train/test que corresponde al modo.

    Parámetros:
        mode (str): "simple" o "complex".

    Devuelve:
        Path: TRAIN_AND_TEST_SIMPLE_DIR si el modo es "simple",
              TRAIN_AND_TEST_COMPLEX_DIR en cualquier otro caso.
    '''
    if mode == "simple":
        return TRAIN_AND_TEST_SIMPLE_DIR
    return TRAIN_AND_TEST_COMPLEX_DIR


def analise():
    '''Ejecuta el script de análisis del dataset (Dataset_analise/dataset_analises_main.py).'''
    run_script(ANALISE_DIR / "dataset_analises_main.py")


def train(mode):
    '''
    Ejecuta el script de entrenamiento (train.py) del modo indicado.

    Parámetros:
        mode (str): "simple" o "complex".
    '''
    run_script(get_mode_dir(mode) / "train.py")


def test(mode):
    '''
    Ejecuta el script de test (test.py) del modo indicado.

    Parámetros:
        mode (str): "simple" o "complex".
    '''
    run_script(get_mode_dir(mode) / "test.py")


def pars_build():
    '''
    Define y lee los argumentos de la línea de comandos.

    Argumentos:
        action: "analisa", "train" o "test" (obligatorio).
        mode:   "simple" o "complex" (opcional, por defecto "simple").

    Devuelve:
        argparse.Namespace: objeto con los atributos action y mode.
    '''
    parser = argparse.ArgumentParser(description="Analizar el dataset, entrenar o testear un modelo.")
    parser.add_argument("action", choices=["analisa", "train", "test"])
    parser.add_argument("mode", choices=MODES, nargs="?", default="simple")
    return parser.parse_args()


def main():
    '''Lee los argumentos de la línea de comandos y lanza la acción pedida.'''
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