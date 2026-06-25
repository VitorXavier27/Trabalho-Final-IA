# Trabalho-Final-IA

# Classificador de Espécies de Cogumelos com PyTorch

Este documento resume a implementação de um modelo de visão computacional para classificar espécies de cogumelos. O código foi estruturado para rodar no Google Colab, aproveitando a aceleração por GPU (CUDA).

## 1. Aquisição e Preparo dos Dados
Para agilizar o setup, utilizei a biblioteca `kagglehub` para puxar o dataset `dariobaumberger/combined-kaggle-mushrooms-dataset` direto para o ambiente do Colab, sem precisar de downloads manuais.

Para viabilizar testes rápidos e não sobrecarregar o processamento:
* Selecionei um subset correspondente a **2% do dataset original**.
* Desse recorte, fiz um split de **70% para treinamento** e **30% para testes**.
* As imagens passaram por um pipeline de transformação via `torchvision.transforms`: redimensionamento para 128x128 pixels, conversão para tensores e normalização (média e desvio padrão em 0.5).

## 2. Arquitetura do Modelo
A espinha dorsal da classificação foi o **EfficientNet-B0**, importado através do `torchvision.models` com os pesos pré-treinados padrão (`weights="DEFAULT"`). 

Para adaptar o modelo ao meu problema, fiz um fine-tuning focado na última camada: peguei as *features* de entrada da camada final original e substituí por uma nova camada linear (`nn.Linear`) com o número exato de espécies presentes no dataset do Kaggle.

## 3. Parametrização e Treinamento
O loop de treinamento foi direto ao ponto, configurado com:
* **Épocas:** 5
* **Batch Size:** 32
* **Otimizador:** Adam (Learning Rate de `1e-3`)
* **Função de Perda:** CrossEntropyLoss

Tanto o modelo quanto os dados (features e labels) foram alocados na GPU. Após o processamento das 5 épocas e o cálculo da precisão, os pesos da rede foram salvos localmente em um arquivo `.pth` (`efficientnet_mushrooms_v2.pth`), evitando a necessidade de retreino constante.

## 4. Inferência (Testando com imagem local)
Para colocar o modelo à prova:
1. Fiz o upload manual de uma imagem-alvo (`cogumelo_teste.jpg`) pelo painel de arquivos do Google Colab.
2. Carreguei o modelo salvo e o coloquei em modo de avaliação (`eval()`).
3. A imagem foi lida com a biblioteca PIL, passou pelas mesmas transformações de redimensionamento e normalização do treino, e foi processada pelo modelo dentro de um bloco `torch.no_grad()`.
4. Utilizei a função `torch.topk` sobre as saídas em `softmax` para extrair e printar as 3 classes com as maiores probabilidades de acerto, exibindo também a imagem original revertida da normalização usando Matplotlib.
