# Domain Adaptation Pipeline

Este projeto implementa um pipeline completo para adaptação de domínio _from target to source_ em tarefas de classificação.

## Objetivo

O objetivo deste projeto é transferir conhecimento de um domínio fonte (source) para um domínio alvo (target) usando diferentes técnicas de adaptação de domínio, melhorando assim a performance de classificação no domínio alvo.

## Pré-requisitos

### Conda (recomendado)

- Ter `conda` instalado (Miniconda, Anaconda ou Mambaforge).

### Criar o ambiente Conda a partir do environment.yml

```bash
# Na raiz do repositório
conda env create -f environment.yml
conda activate tensorflow
```

- Para atualizar um ambiente já existente com base no `environment.yml`:

```bash
conda env update -f environment.yml -n tensorflow --prune
```

- (Opcional) Recriar de um lockfile mais determinístico, se disponível:

```bash
conda env create -f environment.lock.yml
```

- Verificação rápida do TensorFlow após ativar o ambiente:

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

## Como Usar

### Execução Básica

```bash
python main.py
```

### Execução com Parâmetros Customizados

```bash
python main.py \
    --source /path/to/source/dataset \
    --target /path/to/target/dataset \
    --model /path/to/classifier.keras \
    --epochs 100 \
    --batch-size 64 \
    --clusters 3
```

### Parâmetros Disponíveis

- `-s, --source`: Caminho para o dataset fonte (padrão: `../datasets/features/chest-xray-processed/source/`)
- `-t, --target`: Caminho para o dataset alvo (padrão: `../datasets/features/chest-xray-processed/target/`)
- `-m, --model`: Caminho para o modelo classificador (padrão: `../medical-classifier/results/chest-xray-processed/feature_classifier.keras`)
- `-e, --epochs`: Número de épocas de treinamento (padrão: 100)
- `-b, --batch-size`: Tamanho do batch (padrão: 64)
- `-k, --clusters`: Número de clusters K-means (padrão: 1)
- `--debug`: Modo debug (padrão: False)
- `--seed`: Seed para reprodutibilidade (padrão: 42)
