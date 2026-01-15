import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculadora de Torre de Resfriamento", layout="wide")

def formatar_numero(valor, casas_decimais=3):
    """Formata número com vírgula como separador decimal"""
    try:
        if valor is None:
            return "0,0"
        
        # Verifica se é NaN
        if pd.isna(valor):
            return "0,0"
            
        # Formata com o número correto de casas decimais
        format_string = f"{{:.{casas_decimais}f}}"
        numero_formatado = format_string.format(float(valor))
        return numero_formatado.replace('.', ',')
    except Exception as e:
        return f"{valor}"

# CSS para melhorar a aparência
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card h4 {
        margin: 0 0 10px 0;
        color: #1f77b4;
        font-size: 18px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .metric-unit {
        font-size: 16px;
        color: #666;
    }
    .result-title {
        color: #1f77b4;
        font-size: 22px;
        font-weight: bold;
        margin: 20px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid #eee;
    }
    .sidebar-header {
        color: #4CAF50;
        font-weight: bold;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🗼 Calculadora de Torre de Resfriamento")
st.markdown("---")

# Inicializar estado da sessão
if 'calcular' not in st.session_state:
    st.session_state.calcular = False

# Sidebar para parâmetros de entrada
with st.sidebar:
    st.header("💧 Parâmetros de Entrada")
    
    st.markdown('<div class="sidebar-header">Dados Básicos</div>', unsafe_allow_html=True)
    VZ_rec = st.number_input("Vazão de Recirculação (m³/h)", min_value=0.0, value=1000.0, step=50.0, format="%.2f")
    Vol_estatico = st.number_input("Volume Estático (m³)", min_value=0.0, value=50.0, step=5.0, format="%.2f")
    T_retorno = st.number_input("Temperatura de Retorno (°C)", min_value=0.0, value=40.0, step=1.0, format="%.1f")
    T_bacia = st.number_input("Temperatura de Bacia (°C)", min_value=0.0, value=30.0, step=1.0, format="%.1f")
    perc_arraste = st.number_input("% Arraste", min_value=0.0, max_value=100.0, value=0.1, step=0.01, format="%.4f")
    perc_utilizacao = st.number_input("% Utilização", min_value=0.0, max_value=100.0, value=100.0, step=5.0, format="%.1f")
    
    st.markdown("---")
    st.markdown('<div class="sidebar-header">Ciclos de Concentração</div>', unsafe_allow_html=True)
    
    # Dicionário de parâmetros com valores padrão
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
                f"{param} Torre", 
                min_value=0.0, 
                value=dados["torre"],
                step=10.0 if "ppm" in dados["unidade"] else 0.1,
                key=f"torre_{param}",
                format="%.2f",
                help=f"{param} na torre ({dados['unidade']})"
            )
        with col2:
            repos_val = st.number_input(
                f"{param} Reposição", 
                min_value=0.01, 
                value=dados["reposicao"],
                step=5.0 if "ppm" in dados["unidade"] else 0.1,
                key=f"repos_{param}",
                format="%.2f",
                help=f"{param} na reposição ({dados['unidade']})"
            )
        
        if repos_val > 0:
            ciclo = torre_val / repos_val
            ciclos_calculados[param] = ciclo
    
    # Selecionar qual ciclo usar
    st.markdown("---")
    st.markdown('<div class="sidebar-header">Selecionar Ciclo para Cálculos</div>', unsafe_allow_html=True)
    
    if ciclos_calculados:
        # Mostrar os ciclos calculados
        for param, ciclo in ciclos_calculados.items():
            st.text(f"{param}: {formatar_numero(ciclo, 2)} vezes")
        
        # Criar opções para seleção
        opcoes = list(ciclos_calculados.keys())
        opcoes.insert(0, "Usar valor manual")
        
        ciclo_selecionado = st.selectbox("Escolha o ciclo para os cálculos:", opcoes)
        
        if ciclo_selecionado == "Usar valor manual":
            ciclos = st.number_input("Ciclos de Concentração (manual)", 
                                     min_value=1.0, value=3.0, step=0.5, format="%.2f")
        else:
            ciclos = ciclos_calculados[ciclo_selecionado]
            st.success(f"**Usando ciclo de {ciclo_selecionado}:** {formatar_numero(ciclos, 2)} vezes")
    else:
        st.warning("Insira valores de parâmetros para calcular ciclos")
        ciclos = st.number_input("Ciclos de Concentração", 
                                 min_value=1.0, value=3.0, step=0.5, format="%.2f")
    
    st.markdown("---")
    
    # Botão de calcular
    if st.button("🚀 CALCULAR", type="primary", use_container_width=True):
        st.session_state.calcular = True
        st.rerun()

# Área principal para resultados
if st.session_state.calcular:
    st.header("📈 Resultados dos Cálculos")
    
    try:
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
                        f'<h4>Delta Temperatura</h4>'
                        f'<div class="metric-value">{formatar_numero(delta_T, 2)}</div>'
                        f'<div class="metric-unit">°C</div>'
                        f'</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="metric-card">'
                        f'<h4>Evaporação</h4>'
                        f'<div class="metric-value">{formatar_numero(evaporacao, 3)}</div>'
                        f'<div class="metric-unit">m³/h</div>'
                        f'</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="metric-card">'
                        f'<h4>HTI (Tempo de Retenção)</h4>'
                        f'<div class="metric-value">{formatar_numero(HTI, 2)}</div>'
                        f'<div class="metric-unit">horas</div>'
                        f'</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="metric-card">'
                        f'<h4>Reposição</h4>'
                        f'<div class="metric-value">{formatar_numero(reposicao, 3)}</div>'
                        f'<div class="metric-unit">m³/h</div>'
                        f'</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="result-title">Perdas e Purga</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="metric-card">'
                        f'<h4>Perda Líquida</h4>'
                        f'<div class="metric-value">{formatar_numero(perda_liquida, 3)}</div>'
                        f'<div class="metric-unit">m³/h</div>'
                        f'</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="metric-card">'
                        f'<h4>Perda por Arraste</h4>'
                        f'<div class="metric-value">{formatar_numero(perda_arraste, 3)}</div>'
                        f'<div class="metric-unit">m³/h</div>'
                        f'</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="metric-card">'
                        f'<h4>Purga do Sistema</h4>'
                        f'<div class="metric-value">{formatar_numero(purgas, 3)}</div>'
                        f'<div class="metric-unit">m³/h</div>'
                        f'</div>', unsafe_allow_html=True)
            
            st.markdown(f'<div class="metric-card">'
                        f'<h4>Ciclos de Concentração</h4>'
                        f'<div class="metric-value">{formatar_numero(ciclos, 2)}</div>'
                        f'<div class="metric-unit">vezes</div>'
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
        
        st.dataframe(resumo_df, use_container_width=True, hide_index=True)
        
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
        
    except Exception as e:
        st.error(f"Erro nos cálculos: {str(e)}")
    
    # Botões para novo cálculo
    st.markdown("---")
    col_b1, col_b2 = st.columns(2)
    
    with col_b1:
        if st.button("🔄 Novo Cálculo", use_container_width=True):
            st.session_state.calcular = False
            st.rerun()
    
    with col_b2:
        if st.button("📥 Exportar Resultados", use_container_width=True):
            # Criar DataFrame para exportação
            dados_exportacao = {
                "Parâmetro": [
                    "Vazão de Recirculação (m³/h)",
                    "Volume Estático (m³)",
                    "Temperatura de Retorno (°C)",
                    "Temperatura de Bacia (°C)",
                    "% Arraste",
                    "% Utilização",
                    "Ciclos de Concentração (vezes)",
                    "Delta Temperatura (°C)",
                    "Evaporação (m³/h)",
                    "Perda Líquida (m³/h)",
                    "HTI (h)",
                    "Perda por Arraste (m³/h)",
                    "Purga do Sistema (m³/h)",
                    "Reposição (m³/h)"
                ],
                "Valor": [
                    formatar_numero(VZ_rec, 2),
                    formatar_numero(Vol_estatico, 2),
                    formatar_numero(T_retorno, 1),
                    formatar_numero(T_bacia, 1),
                    formatar_numero(perc_arraste, 4),
                    formatar_numero(perc_utilizacao, 1),
                    formatar_numero(ciclos, 2),
                    formatar_numero(delta_T, 2),
                    formatar_numero(evaporacao, 3),
                    formatar_numero(perda_liquida, 3),
                    formatar_numero(HTI, 2),
                    formatar_numero(perda_arraste, 3),
                    formatar_numero(purgas, 3),
                    formatar_numero(reposicao, 3)
                ]
            }
            
            export_df = pd.DataFrame(dados_exportacao)
            
            # Converter para CSV
            csv = export_df.to_csv(index=False, sep=';', decimal=',')
            st.download_button(
                label="📄 Baixar CSV",
                data=csv,
                file_name="resultados_torre_resfriamento.csv",
                mime="text/csv"
            )

else:
    # Tela inicial quando ainda não calculou
    st.markdown("""
    ## 🏭 Bem-vindo à Calculadora de Torre de Resfriamento
    
    ### 📋 Instruções:
    1. **Preencha todos os parâmetros** na **barra lateral** ←
    2. Insira valores para os **5 parâmetros químicos** (Torre e Reposição)
    3. **Selecione qual ciclo** de concentração usar nos cálculos
    4. Clique no botão **🚀 CALCULAR** para ver os resultados
    
    ### 🔬 Parâmetros disponíveis para cálculo de ciclos:
    - **Sílica** (ppm)
    - **Cloreto** (ppm)
    - **Dureza Total** (ppm CaCO₃)
    - **Alcalinidade Total** (ppm CaCO₃)
    - **Ferro Total** (ppm)
    
    ---
    
    *Os resultados serão exibidos aqui após o cálculo.*
    """)
    
    # Exemplo de layout vazio
    with st.expander("ℹ️ Sobre os cálculos"):
        st.markdown("""
        Esta calculadora realiza os seguintes cálculos:
        
        1. **Delta Temperatura** - Diferença entre retorno e bacia
        2. **Evaporação** - Baseada na vazão, delta T e utilização
        3. **Perda Líquida** - Relacionada aos ciclos de concentração
        4. **HTI** - Índice de Tempo de Retenção
        5. **Perda por Arraste** - Baseada no percentual de arraste
        6. **Purga do Sistema** - Perda líquida menos arraste
        7. **Reposição** - Evaporação mais perda líquida
        """)

# Rodapé
st.markdown("---")
st.markdown("⚡ *Calculadora desenvolvida para otimização de torres de resfriamento* | 📧 Suporte técnico disponível")
