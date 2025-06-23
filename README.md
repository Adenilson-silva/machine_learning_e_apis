# Projeto Básico de MLOps (Machine Learning e APIs) 
## API de Priorização com Modelos de Machine Learning 
<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1WdUIXOim7YvCHhwN4QUqaXVE-d2GUw-u" width="500">
</div>

## Sobre este Projeto 
Este projeto é uma API desenvolvida em _Flask_ que expõe quatro modelos de _Machine Learning_ para classificação de priorização com base em atributos fornecidos via requisição HTTP POST.

Os modelo de _Machine Learning_ aqui empregados foram desenvolvidos em outro projeto já disponibilizado <a href="https://github.com/Adenilson-silva/sicor" target="_blank">aqui</a>.

## Modelos Disponíveis
A API permite acesso a quatro modelos:

- _CategoricalNBModel_ (Naive Bayes para dados categóricos)

- _DecisionTreeClassifier_ (Árvore de decisão)

- _RandomForestClassifier_ (Floresta aleatória)

- _XGBClassifier_ (XGBoost Classifier)

Cada modelo retorna a priorização prevista com base em cinco campos obrigatórios.

A documentação da API pode ser visualizada <a href="https://documenter.getpostman.com/view/17572991/2sB2qaiMfH" target="_blank">aqui</a>.
