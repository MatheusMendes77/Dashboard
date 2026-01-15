import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculadora de Torre de Resfriamento", layout="wide")

# Configurar para usar vírgula como separador decimal
import locale
try:
    locale.setlocale(locale.LC_NUMERIC, 'pt_BR.UTF-8')
except:
    locale.setlocale(locale.LC_NUMERIC, 'Portuguese_Brazil.1252')

def formatar_numero(valor, casas_decimais=3):
    """Formata número com vírgula como separador decimal"""
    try:
        # Usar locale para formatação
        return locale.format_string(f"%.{casas_decimais}f", valor, grouping=False)
    except:
        # Fallback se locale falhar
        return f"{valor:.{casas_decimais}f}".replace('.', ',')

# CSS para melhorar a aparência
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 5px 0;
        border-left: 5px solid #4CAF50;
    }
    .result-title {
        color: #1f77b4;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📊 Calculadora de Torre de Resfriamento")
st.markdown("---")

# Sidebar para parâmetros de entrada
with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada")
    
    st.subheader("Dados Básicos")
    VZ_rec = st.number_input("Vazão de Recirculação (m³/h)", min_value=0.0, value=1000.0, step=50.0, format="%.2f")
    Vol_estatico = st.number_input("Volume Estático (m³)", min_value=0.0, value=50.0, step=5.0, format="%.2f")
    T_retorno = st.number_input("Temperatura de Retorno (°C)", min_value=0.0, value=40.0, step=1.0, format="%.1f")
    T_bacia = st.number_input("Temperatura de Bacia (°C)", min_value=0.0, value=30.0, step=1.0, format="%.1f")
    perc_arraste = st.number_input("% Arraste", min_value=0.0, max_value=100.0, value=0.1, step=0.01, format="%.4f")
    perc_utilizacao = st.number_input("% Utilização", min_value=0.0, max_value=100.0, value=100.0, step=5.0, format="%.1f")
    
    st.markdown("---")
    st.subheader("Ciclos de Concentração")
    
    # Dicionário de parâmetros
    parametros = {
        "Sílica": {"torre": 150.0, "reposicao": 50.0, "unidade": "ppm"},
        "Cloreto": {"torre": 200.0, "reposicao": 50.0, "unidade": "ppm"},
        "Dureza Total": {"torre": 300.0, "reposicao": 100.0, "unidade": "ppm CaCO₃"},
        "Alcalinidade Total": {"torre": 250.0, "reposicao": 80.0, "unidade": "ppm CaCO₃"},
        "Ferro Total": {"torre": 1.5, "reposicao": 0.3, "unidade": "ppm"}
    }
    
    # Criar colunas para cada parâmetro
    ciclos_calculados = {}
    
    for param, dados in parametros.items():
        col1, col2 = st.columns(2)
        with col1:
            torre_val = st.number_input(
                f"{param} na Torre", 
                min_value=0.0, 
                value=dados["torre"],
                step=10.0 if "ppm" in dados["unidade"] else 0.1,
                key=f"torre_{param}",
                format="%.2f"
            )
        with col2:
            repos_val = st.number_input(
                f"{param} na Reposição", 
                min_value=0.01, 
                value=dados["reposicao"],
                step=5.0 if "ppm" in dados["unidade"] else 0.1,
                key=f"repos_{param}",
                format="%.2f"
            )
        
        if repos_val > 0:
            ciclo = torre_val / repos_val
            ciclos_calculados[param] = ciclo
    
    # Selecionar qual ciclo usar
    st.markdown("---")
    st.subheader("Selecionar Ciclo para Cálculos")
    
    if ciclos_calculados:
        # Criar opções formatadas
        opcoes = [f"{param}: {ciclo:.2f} vezes" for param, ciclo in ciclos_calculados.items()]
        opcoes.insert(0, "Usar valor manual")
        
        ciclo_selecionado = st.selectbox("Escolha o ciclo para os cálculos:", opcoes)
        
        if ciclo_selecionado == "Usar valor manual":
            ciclos = st.number_input("Ciclos de Concentração (manual)", 
                                     min_value=1.0, value=3.0, step=0.5, format="%.2f")
        else:
            # Extrair o parâmetro selecionado
            param_selecionado = ciclo_selecionado.split(":")[0]
            ciclos = ciclos_calculados[param_selecionado]
            st.success(f"Usando ciclo de **{param_selecionado}**: **{ciclos:.2f} vezes**")
    else:
        st.warning("Insira valores de parâmetros para calcular ciclos")
        ciclos = st.number_input("Ciclos de Concentração", 
                                 min_value=1.0, value=3.0, step=0.5, format="%.2f")
    
    st.markdown("---")
    
    # Botão de calcular
    calcular = st.button("🚀 CALCULAR", type="primary", use_container_width=True)

# Área principal para resultados
if calcular:
    st.header("📈 Resultados dos Cálculos")
    
    # Converter porcentagens para decimal
    perc_utilizacao_decimal = perc_utilizacao / 100.0
    
    # 1. Delta Temperatura
    delta_T = T_retorno - T_bacia
    
    # 2. Evaporação
    evaporacao = VZ_rec * delta_T * (0.85 / 556) * perc_utilizacao_decimal
    
    # 3. Perda Líquida
    if ciclos > 1:
        perda_liquida = evaporacao / (ciclos - 1)
    else:
        perda_liquida = 0.0
        st.error("⚠️ Ciclos de concentração devem ser maiores que 1!")
    
    # 4. HTI (Índice de Tempo de Retenção)
    if perda_liquida > 0:
        HTI = 0.693 * (Vol_estatico / perda_liquida)
    else:
        HTI = 0.0
    
    # 5. Perda por Arraste
    perda_arraste = (perc_arraste / 100.0) * VZ_rec * perc_utilizacao_decimal
    
    # 6. Purga do Sistema
    purgas = perda_liquida - perda_arraste
    if purgas < 0:
        purgas = 0.0
        st.warning("Perda por arraste maior que perda líquida - purga ajustada para zero")
    
    # 7. Reposição
    reposicao = evaporacao + perda_liquida
    
    # Exibir resultados em colunas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="result-title">Resultados Principais</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card">'
                    f'<strong>Delta Temperatura:</strong><br>'
                    f'{formatar_numero(delta_T, 2)} °C'
                    f'</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card">'
                    f'<strong>Evaporação:</strong><br>'
                    f'{formatar_numero(evaporacao, 3)} m³/h'
                    f'</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card">'
                    f'<strong>HTI (Tempo de Retenção):</strong><br>'
                    f'{formatar_numero(HTI, 2)} horas'
                    f'</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card">'
                    f'<strong>Reposição:</strong><br>'
                    f'{formatar_numero(reposicao, 3)} m³/h'
                    f'</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="result-title">Perdas e Purga</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card">'
                    f'<strong>Perda Líquida:</strong><br>'
                    f'{formatar_numero(perda_liquida, 3)} m³/h'
                    f'</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card">'
                    f'<strong>Perda por Arraste:</strong><br>'
                    f'{formatar_numero(perda_arraste, 3)} m³/h'
                    f'</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card">'
                    f'<strong>Purga do Sistema:</strong><br>'
                    f'{formatar_numero(purgas, 3)} m³/h'
                    f'</div>', unsafe_allow_html=True)
        
        st.markdown(f'<div class="metric-card">'
                    f'<strong>Ciclos de Concentração:</strong><br>'
                    f'{formatar_numero(ciclos, 2)} vezes'
                    f'</div>', unsafe_allow_html=True)
    
    # Tabela resumo
    st.markdown("---")
    st.subheader("📋 Resumo dos Resultados")
    
    resumo_df = pd.DataFrame({
        "Parâmetro": [
            "Delta Temperatura",
            "Evaporação",
            "Perda Líquida",
            "HTI",
            "Perda por Arraste",
            "Purga do Sistema",
            "Reposição",
            "Ciclos de Concentração"
        ],
        "Valor": [
            f"{formatar_numero(delta_T, 2)} °C",
            f"{formatar_numero(evaporacao, 3)} m³/h",
            f"{formatar_numero(perda_liquida, 3)} m³/h",
            f"{formatar_numero(HTI, 2)} h",
            f"{formatar_numero(perda_arraste, 3)} m³/h",
            f"{formatar_numero(purgas, 3)} m³/h",
            f"{formatar_numero(reposicao, 3)} m³/h",
            f"{formatar_numero(ciclos, 2)} vezes"
        ],
        "Descrição": [
            "Diferença entre temperatura de retorno e bacia",
            "Vazão evaporada na torre",
            "Água perdida total",
            "Índice de Tempo de Retenção",
            "Água perdida por arraste",
            "Água descartada para controle",
            "Vazão de água reposta",
            "Ciclos de concentração selecionados"
        ]
    })
    
    st.table(resumo_df)
    
    # Verificações de consistência
    st.markdown("---")
    st.subheader("✅ Verificação de Consistência")
    
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        st.markdown(f"**Evaporação + Perda Líquida = Reposição**")
        st.markdown(f"{formatar_numero(evaporacao, 3)} + {formatar_numero(perda_liquida, 3)} = {formatar_numero(reposicao, 3)} m³/h")
        
    with col_v2:
        st.markdown(f"**Perda Líquida = Purga + Arraste**")
        st.markdown(f"{formatar_numero(perda_liquida, 3)} = {formatar_numero(purgas, 3)} + {formatar_numero(perda_arraste, 3)} m³/h")
    
    # Botão para novo cálculo
    st.markdown("---")
    if st.button("🔄 Novo Cálculo"):
        st.rerun()

else:
    # Tela inicial quando ainda não calculou
    st.markdown("""
    ## Bem-vindo à Calculadora de Torre de Resfriamento
    
    ### Instruções:
    1. Preencha todos os parâmetros na **barra lateral** ←
    2. Insira valores para os **5 parâmetros** (Torre e Reposição)
    3. Selecione qual ciclo de concentração usar
    4. Clique no botão **🚀 CALCULAR** para ver os resultados
    
    ### Parâmetros disponíveis para cálculo de ciclos:
    - Sílica
    - Cloreto
    - Dureza Total
    - Alcalinidade Total
    - Ferro Total
    
    ---
    
    *Os resultados serão exibidos aqui após o cálculo.*
    """)
    
    # Placeholder vazio
    st.empty()

# Rodapé
st.markdown("---")
st.markdown("⚡ *Calculadora desenvolvida para otimização de torres de resfriamento*")
