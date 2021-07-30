# Faz os imports utilizados
import os
from flask import Flask, request, render_template
import numpy as np

# importa o deserializador de objetos
import joblib

# Carrega a classe de predição do diretório local
# Carregando o modelo em disco para a memória da nossa aplicação.
modelo = joblib.load('modelo/dm2_30042021_3930_01.sav')

app = Flask(__name__)      # Iniciando a aplicação.

@app.route('/')
def index():
    return render_template('index.html')              # renderizando o um template html

def obtem_dados_form(): # obtém todos os dados do paciente fornecidos no formulário de entrada
    dados=dict()
    
    dados["sexo"] = int(request.form['sexo'])
    dados["idade"] = int(request.form['idade'])
    #  Glicose, soro refrigerado (mmol / L);
    dados["glicose_mmolL"] = float(request.form['glicose_mmolL'])
    # Triglicerídeos (mmol / L)
    dados["triglicerides_mmolL"] = float(request.form['triglicerides_mmolL'])
    # Hemoglobina glicada (Hb1ac)
    dados["hemoglobina_glicada"] = float(request.form['hemoglobina_glicada'])
    #  Insulina (pmol / L);
    dados["insulina_pmolL"] = float(request.form['insulina_pmolL'])
    #  Peso (kg);
    dados["peso"] = float(request.form['peso'])
    #  Altura em pé (cm);
    dados["altura"] = float(request.form['altura'])
    #  Circunferência da cintura (cm);
    dados["circunferencia_cintura"] = float(request.form['circunferencia_cintura'])

    return dados

def monta_dicionario_resposta(dados, classe, res_proba, cor):

    resposta = dict()

    resposta["idade"] = str(dados["idade"])
    resposta["sexo"] = "Mulher" if dados["sexo"] == 1 else "Homem"
    resposta["glicose_mmolL"] = str(dados["glicose_mmolL"])
    resposta["triglicerides_mmolL"] = str(dados["triglicerides_mmolL"])
    resposta["hemoglobina_glicada"] = str(dados["hemoglobina_glicada"])
    resposta["insulina_pmolL"] = str(dados["insulina_pmolL"])
    resposta["peso"] = str(dados["peso"])
    resposta["altura"] = str(dados["altura"])
    resposta["circunferencia_cintura"] = str(dados["circunferencia_cintura"])
    
    resposta["cor"] = cor
    resposta["classe"] = classe
    resposta["res_proba"] = res_proba

    return resposta





@app.route('/verificar', methods=['POST'])          # recebe a requisição, coleta os dados a partir dos requests, esses dados compõe as variáveis em uma amostra de teste e faz a predição.
def verificar():

    # obtém os valores dos atributos obtidos do formulário HTML index.html
    dados = obtem_dados_form()
    print("dados:"+str(dados.values));
    # Cria array numpy para teste
    teste = np.array([list(dados.values())])
    print("teste: "+ str(teste))
    # Na função acima temos todos os atributos que foram usados para treinar o modelo.
    print(":::::: Mostra alguns dados de teste ::::::")

    print("idade: {}".format(dados["idade"]))
    print("sexo: {}".format(dados["sexo"]))
    print("triglicerideos:  "+str(dados["triglicerides_mmolL"]))

    print("hemoglobina_glicada: "+str(dados["hemoglobina_glicada"]))
    print("insulina_pmolL: "+str(dados["insulina_pmolL"]))
    print("glicose_mmolL: "+str(dados["glicose_mmolL"]))

    print("peso: "+str(dados["peso"]))
    print("altura: "+str(dados["altura"]))
    print("circunferencia_cintura: "+str(dados["circunferencia_cintura"]))


    print("\n")
    # Fazendo a predição:
    classe = modelo.predict(teste)[0]
    proba = modelo.predict_proba(teste) # obtém a probabilidade da predição
    res_proba = "%5.2f"%(proba[0][1]*100) # posição 0,0 proba não portador, 0,1 proba portador


    print("Classe Predita: {}".format(str(classe)))
    # Mostrar na aplicação qual foi o retorno do modelo
    if classe == 0:
        classe = 'NEGATIVO'
    else:
        classe = 'POSITIVO'
        
    if (proba[0][1]*100) < 20:
        cor = '#E7EEF8'
    else:
        if (proba[0][1]*100) < 40:
            cor = '#CFDBEA'
        else:
            if (proba[0][1]*100) < 60:
                cor = '#B3C0D0'
            else:
                if (proba[0][1]*100) < 80:
                    cor = '#6885A3'
                else:
                    cor = '#054F77'

   
    resposta = monta_dicionario_resposta(dados, classe, res_proba, cor)

    # renderiza o HTML resultado.html passando os valores idade, forma, Margem
    # densidade, resultado da predição e a probabilidade do Nódulo ser Maligno
    return render_template('resultado.html', resultado=resposta)

# Executa a aplição na porta 80 (localhost)
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5998))
    app.run(host='127.0.0.1', port=port)

 