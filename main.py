
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
import pandas as pd
import pickle

colunas_categoricalNB_model = ['finalidade','atividade','modalidade','produto', 'categoria_empresa']

categoricalNB_model = pickle.load(open('modelos\modelo_1_CategoricalNB\CategoricalNB_model.pickle','rb'))
encoder_categoricalNB_model = pickle.load(open('modelos\modelo_1_CategoricalNB\encoder.pickle', 'rb'))

decision_tree_classifier_model = pickle.load(open('modelos\modelo_2_DecisionTreeClassifier\DecisionTreeClassifier.pickle','rb'))
colunas_decision_tree_classifier = pickle.load(open('modelos\modelo_2_DecisionTreeClassifier\colunas.pickle', 'rb'))

random_forest_classifier_model = pickle.load(open('modelos\modelo_3_RandomForestClassifier\RandomForestClassifier.pickle','rb'))
colunas_random_forest_classifier = pickle.load(open('modelos\modelo_3_RandomForestClassifier\colunas.pickle', 'rb'))

XGB_classifier_model = pickle.load(open('modelos\modelo_4_XGBClassifier\XGBClassifier.pickle','rb'))
colunas_XGB_classifier = pickle.load(open('modelos\modelo_4_XGBClassifier\colunas.pickle', 'rb'))


app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'user'
app.config['BASIC_AUTH_PASSWORD'] = 'password'

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Modelos de Machine Learning"

@app.route('/CategoricalNBModel/', methods=['POST'])
@basic_auth.required
def CategoricalNBModel():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas_categoricalNB_model]
    dados_codificados = encoder_categoricalNB_model.transform([dados_input])
    priorizacao = categoricalNB_model.predict(dados_codificados)
    return jsonify(priorizacao = priorizacao[0])

@app.route('/DecisionTreeClassifier/', methods=['POST'])
@basic_auth.required
def DecisionTreeClassifier():
    dados = request.get_json()
    df = pd.DataFrame([dados])
    df_dummies = pd.get_dummies(df)
    colunas_faltantes = [col for col in colunas_decision_tree_classifier if col not in df_dummies]
    df_novas_colunas = pd.DataFrame(0, index=df_dummies.index, columns=colunas_faltantes)
    df_dummies = pd.concat([df_dummies, df_novas_colunas], axis=1)
    df_dummies = df_dummies[colunas_decision_tree_classifier]
    priorizacao = decision_tree_classifier_model.predict(df_dummies)
    valor = bool(priorizacao[0])
    resultado = 'Normal' if valor else 'Alto'
    return jsonify(priorizacao=resultado)

@app.route('/RandomForestClassifier/', methods=['POST'])
@basic_auth.required
def RandomForestClassifier():
    dados = request.get_json()
    df = pd.DataFrame([dados])
    df_dummies = pd.get_dummies(df)
    colunas_faltantes = [col for col in colunas_random_forest_classifier if col not in df_dummies]
    df_novas_colunas = pd.DataFrame(0, index=df_dummies.index, columns=colunas_faltantes)
    df_dummies = pd.concat([df_dummies, df_novas_colunas], axis=1)
    df_dummies = df_dummies[colunas_random_forest_classifier]
    priorizacao = random_forest_classifier_model.predict(df_dummies)
    valor = bool(priorizacao[0])
    resultado = 'Normal' if valor else 'Alto'
    return jsonify(priorizacao=resultado)

@app.route('/XGBClassifier/', methods=['POST'])
@basic_auth.required
def XGBClassifier():
    dados = request.get_json()
    df = pd.DataFrame([dados])
    df_dummies = pd.get_dummies(df)
    colunas_faltantes = [col for col in colunas_random_forest_classifier if col not in df_dummies]
    df_novas_colunas = pd.DataFrame(0, index=df_dummies.index, columns=colunas_faltantes)
    df_dummies = pd.concat([df_dummies, df_novas_colunas], axis=1)
    df_dummies = df_dummies[colunas_random_forest_classifier]
    priorizacao = random_forest_classifier_model.predict(df_dummies)
    valor = bool(priorizacao[0])
    resultado = 'Normal' if valor else 'Alto'
    return jsonify(priorizacao=resultado)


app.run(debug=True)