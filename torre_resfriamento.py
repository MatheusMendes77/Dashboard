import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculadora de Torre de Resfriamento", layout="wide")

def formatar_numero(valor, casas_decimais=3):
    """Formata número com vírgula como separador decimal e ponto como separador de milhar"""
    try:
        if valor is None or valor == 0:
            return "0,00"
        
        if pd.isna(valor):
            return "0,00"
            
        format_string = f"{{:.{casas_decimais}f}}"
        numero_formatado = format_string.format(float(valor))
        
        partes = numero_formatado.split('.')
        parte_inteira = partes[0]
        parte_decimal = partes[1] if len(partes) > 1 else ''
        
        parte_inteira_com_pontos = ""
        for i, char in enumerate(reversed(parte_inteira)):
            if i > 0 and i % 3 == 0:
                parte_inteira_com_pontos = '.' + parte_inteira_com_pontos
            parte_inteira_com_pontos = char + parte_inteira_com_pontos
        
        if parte_decimal:
            return f"{parte_inteira_com_pontos},{parte_decimal}"
        else:
            return f"{parte_inteira_com_pontos}"
            
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
    .sidebar-header {
        color: #4CAF50;
        font-weight: bold;
        margin-top: 20px;
        font-size: 16px;
    }
    .instruction-box {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border-left: 5px solid #4CAF50;
    }
    .param-box {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        min-height: 100px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .param-title {
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 5px;
        font-size: 16px;
    }
    .param-unit {
        color: #666;
        font-size: 14px;
        margin-top: 5px;
    }
    .center-container {
        display: flex;
        justify-content: center;
        margin: 20px 0;
    }
    .flow-container {
        width: 100%;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .flow-step {
        background-color: white;
        border-radius: 10px;
        padding: 25px;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        border-left: 6px solid;
    }
    .flow-title {
        font-weight: bold;
        margin-bottom: 20px;
        font-size: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid;
        text-align: center;
    }
    .flow-value {
        font-size: 36px;
        font-weight: bold;
        margin: 15px 0;
        line-height: 1.2;
        text-align: center;
    }
    .flow-unit {
        color: #555;
        font-size: 16px;
        margin-top: 8px;
        font-weight: 500;
        text-align: center;
    }
    .flow-arrow {
        text-align: center;
        font-size: 40px;
        color: #4CAF50;
        margin: 15px 0;
        opacity: 0.7;
    }
    .flow-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 25px;
        margin: 25px 0;
    }
    .flow-grid-item {
        background-color: #f8f9fa;
        padding: 25px;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #e0e0e0;
        transition: transform 0.3s ease;
    }
    .flow-grid-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    .flow-diagram {
        background-color: #f5f9ff;
        padding: 35px;
        border-radius: 20px;
        margin: 30px 0;
        border: 2px solid #d0e3ff;
    }
    
    /* Cores específicas para cada seção */
    .step-entrada {
        border-left-color: #FF6B6B;
    }
    .step-entrada .flow-title {
        color: #FF6B6B;
        border-bottom-color: #FF6B6B;
    }
    .step-entrada .flow-value {
        color: #FF6B6B;
    }
    
    .step-resfriamento {
        border-left-color: #4ECDC4;
    }
    .step-resfriamento .flow-title {
        color: #4ECDC4;
        border-bottom-color: #4ECDC4;
    }
    .step-resfriamento .flow-value {
        color: #4ECDC4;
    }
    
    .step-perdas {
        border-left-color: #FFD166;
    }
    .step-perdas .flow-title {
        color: #FFD166;
        border-bottom-color: #FFD166;
    }
    .step-perdas .flow-value {
        color: #FFD166;
    }
    
    .step-reposicao {
        border-left-color: #06D6A0;
    }
    .step-reposicao .flow-title {
        color: #06D6A0;
        border-bottom-color: #06D6A0;
    }
    .step-reposicao .flow-value {
        color: #06D6A0;
    }
    
    .flow-grid-item:nth-child(1) .flow-value {
        color: #FF6B6B;
    }
    .flow-grid-item:nth-child(2) .flow-value {
        color: #4ECDC4;
    }
    .flow-grid-item:nth-child(3) .flow-value {
        color: #06D6A0;
    }
    
    .button-row {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 40px;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Centralização dos conteúdos dentro das colunas */
    .flow-column-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    
    /* Estilos para remover quadros brancos */
    .st-emotion-cache-1y4p8pa {
        padding: 0;
    }
    .st-emotion-cache-1y4p8pa > div {
        padding: 0;
    }
    
    /* Estilo específico para PERDAS E CONTROLE - 3 itens lado a lado */
    .perdas-container {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 20px;
        margin: 20px 0;
        padding: 20px 0;
    }
    
    .perda-item {
        flex: 1;
        text-align: center;
        padding: 15px;
        border-radius: 8px;
        background-color: #f8f9fa;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .perda-valor {
        font-size: 32px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 8px;
    }
    
    .perda-titulo {
        font-size: 16px;
        font-weight: 600;
        color: #444;
        margin-bottom: 4px;
    }
    
    .perda-descricao {
        font-size: 12px;
        color: #666;
        margin-top: 4px;
        line-height: 1.3;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏭 Calculadora de Torre de Resfriamento")
st.markdown("---")

# Inicializar estado da sessão
if 'calcular' not in st.session_state:
    st.session_state.calcular = False

# Sidebar para parâmetros de entrada
with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada")
    
    st.markdown('<div class="sidebar-header">Dados Básicos</div>', unsafe_allow_html=True)
    VZ_rec = st.number_input("Vazão de Recirculação (m³/h)", min_value=0.0, value=None, step=50.0, format="%.2f", placeholder="Ex: 1.000,00")
    Vol_estatico = st.number_input("Volume Estático (m³)", min_value=0.0, value=None, step=5.0, format="%.2f", placeholder="Ex: 50,00")
    T_retorno = st.number_input("Temperatura de Retorno (°C)", min_value=0.0, value=None, step=1.0, format="%.1f", placeholder="Ex: 40,0")
    T_bacia = st.number_input("Temperatura de Bacia (°C)", min_value=0.0, value=None, step=1.0, format="%.1f", placeholder="Ex: 30,0")
    perc_arraste = st.number_input("% Arraste", min_value=0.0, max_value=100.0, value=None, step=0.01, format="%.4f", placeholder="Ex: 0,1000")
    perc_utilizacao = st.number_input("% Utilização", min_value=0.0, max_value=100.0, value=100.0, step=5.0, format="%.1f")
    
    st.markdown("---")
    st.markdown('<div class="sidebar-header">Ciclos de Concentração</div>', unsafe_allow_html=True)
    
    parametros = {
        "Sílica": {"torre": None, "reposicao": None, "unidade": "ppm"},
        "Cloreto": {"torre": None, "reposicao": None, "unidade": "ppm"},
        "Dureza Total": {"torre": None, "reposicao": None, "unidade": "ppm CaCO₃"},
        "Alcalinidade Total": {"torre": None, "reposicao": None, "unidade": "ppm CaCO₃"},
        "Ferro Total": {"torre": None, "reposicao": None, "unidade": "ppm"}
    }
    
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
                help=f"{param} na torre ({dados['unidade']})",
                placeholder="Ex: 150,00"
            )
        with col2:
            repos_val = st.number_input(
                f"{param} Reposição", 
                min_value=0.0,
                value=dados["reposicao"],
                step=5.0 if "ppm" in dados["unidade"] else 0.1,
                key=f"repos_{param}",
                format="%.2f",
                help=f"{param} na reposição ({dados['unidade']})",
                placeholder="Ex: 50,00"
            )
        
        if repos_val is not None and repos_val > 0 and torre_val is not None:
            ciclo = torre_val / repos_val
            ciclos_calculados[param] = ciclo
    
    # Selecionar qual ciclo usar
    st.markdown("---")
    st.markdown('<div class="sidebar-header">Selecionar Ciclo para Cálculos</div>', unsafe_allow_html=True)
    
    if ciclos_calculados:
        opcoes = list(ciclos_calculados.keys())
        opcoes.insert(0, "Usar valor manual")
        
        ciclo_selecionado = st.selectbox("Escolha o ciclo para os cálculos:", opcoes)
        
        if ciclo_selecionado == "Usar valor manual":
            ciclos = st.number_input("Ciclos de Concentração (manual)", 
                                     min_value=1.0, value=None, step=0.5, format="%.2f",
                                     placeholder="Ex: 3,00")
        else:
            ciclos = ciclos_calculados[ciclo_selecionado]
            st.success(f"**Usando ciclo de {ciclo_selecionado}:** {formatar_numero(ciclos, 2)} vezes")
    else:
        st.warning("Insira valores de parâmetros para calcular ciclos")
        ciclos = st.number_input("Ciclos de Concentração", 
                                 min_value=1.0, value=None, step=0.5, format="%.2f",
                                 placeholder="Ex: 3,00")
    
    st.markdown("---")
    
    # Botão de calcular
    if st.button("🚀 CALCULAR", type="primary", use_container_width=True):
        st.session_state.calcular = True
        st.rerun()

# Área principal para resultados
if st.session_state.calcular:
    st.markdown('<h2 style="text-align: center; color: #1f77b4; margin-bottom: 30px; font-size: 32px;">📊 FLUXO DA TORRE DE RESFRIAMENTO</h2>', unsafe_allow_html=True)
    
    try:
        # Tratar valores None
        VZ_rec = VZ_rec if VZ_rec is not None else 0.0
        Vol_estatico = Vol_estatico if Vol_estatico is not None else 0.0
        T_retorno = T_retorno if T_retorno is not None else 0.0
        T_bacia = T_bacia if T_bacia is not None else 0.0
        perc_arraste = perc_arraste if perc_arraste is not None else 0.0
        perc_utilizacao = perc_utilizacao if perc_utilizacao is not None else 100.0
        ciclos = ciclos if ciclos is not None else 1.0
        
        # Converter porcentagens para decimal
        perc_utilizacao_decimal = perc_utilizacao / 100.0
        
        # Cálculos
        delta_T = T_retorno - T_bacia
        evaporacao = VZ_rec * delta_T * (0.85 / 556) * perc_utilizacao_decimal
        
        if ciclos > 1:
            perda_liquida = evaporacao / (ciclos - 1)
        else:
            perda_liquida = 0.0
            if ciclos <= 1 and ciclos > 0:
                st.error("⚠️ Ciclos de concentração devem ser maiores que 1!")
        
        if perda_liquida > 0:
            HTI = 0.693 * (Vol_estatico / perda_liquida)
        else:
            HTI = 0.0
        
        perda_arraste = (perc_arraste / 100.0) * VZ_rec * perc_utilizacao_decimal
        purgas = perda_liquida - perda_arraste
        if purgas < 0:
            purgas = 0.0
            st.warning("Perda por arraste maior que perda líquida - purga ajustada para zero")
        
        reposicao = evaporacao + perda_liquida
        
        # Diagrama do Fluxo da Torre
        st.markdown('<div class="flow-diagram">', unsafe_allow_html=True)
        
        # Seção 1: Entrada de Água Quente
        st.markdown('<div class="flow-step step-entrada">', unsafe_allow_html=True)
        st.markdown('<div class="flow-title">🔥 ENTRADA - ÁGUA QUENTE DO PROCESSO</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="flow-column-content">', unsafe_allow_html=True)
            st.markdown(f'<div class="flow-value">🌡️ {formatar_numero(T_retorno, 1)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="flow-unit">Temperatura de Retorno</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="flow-column-content">', unsafe_allow_html=True)
            st.markdown(f'<div class="flow-value">💧 {formatar_numero(VZ_rec, 2)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="flow-unit">Vazão de Recirculação</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="flow-column-content">', unsafe_allow_html=True)
            st.markdown(f'<div class="flow-value">⚙️ {formatar_numero(perc_utilizacao, 1)}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="flow-unit">Utilização da Torre</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Seta para baixo
        st.markdown('<div class="flow-arrow">⬇️</div>', unsafe_allow_html=True)
        
        # Seção 2: Resfriamento na Torre
        st.markdown('<div class="flow-step step-resfriamento">', unsafe_allow_html=True)
        st.markdown('<div class="flow-title">🏭 RESFRIAMENTO NA TORRE</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="flow-column-content">', unsafe_allow_html=True)
            st.markdown(f'<div class="flow-value">📉 {formatar_numero(delta_T, 2)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="flow-unit">ΔT (Redução de Temperatura)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="flow-column-content">', unsafe_allow_html=True)
            st.markdown(f'<div class="flow-value">🌡️ {formatar_numero(T_bacia, 1)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="flow-unit">Temperatura da Bacia</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="flow-column-content">', unsafe_allow_html=True)
            st.markdown(f'<div class="flow-value">💨 {formatar_numero(evaporacao, 3)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="flow-unit">Evaporação</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Seta para baixo
        st.markdown('<div class="flow-arrow">⬇️</div>', unsafe_allow_html=True)
        
        # Seção 3: Perdas e Purga - AGORA COM OS 3 VALORES LADO A LADO
        st.markdown('<div class="flow-step step-perdas">', unsafe_allow_html=True)
        st.markdown('<div class="flow-title">💧 PERDAS E CONTROLE</div>', unsafe_allow_html=True)
        
        # Container com os 3 itens lado a lado
        st.markdown('<div class="perdas-container">', unsafe_allow_html=True)
        
        # Item 1: Perda Líquida Total
        st.markdown('''
        <div class="perda-item">
            <div class="perda-valor">''' + formatar_numero(perda_liquida, 3) + '''</div>
            <div class="perda-titulo">Perda Líquida Total</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Item 2: Perda por Arraste
        st.markdown('''
        <div class="perda-item">
            <div class="perda-valor">''' + formatar_numero(perda_arraste, 3) + '''</div>
            <div class="perda-titulo">Perda por Arraste</div>
            <div class="perda-descricao">(''' + formatar_numero(perc_arraste, 4) + '''% do recirculado)</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Item 3: Purga do Sistema
        st.markdown('''
        <div class="perda-item">
            <div class="perda-valor">''' + formatar_numero(purgas, 3) + '''</div>
            <div class="perda-titulo">Purga do Sistema</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Fecha perdas-container
        
        st.markdown('</div>', unsafe_allow_html=True)  # Fecha flow-step
        
        # Seta para baixo
        st.markdown('<div class="flow-arrow">⬇️</div>', unsafe_allow_html=True)
        
        # Seção 4: Reposição e Balanço
        st.markdown('<div class="flow-step step-reposicao">', unsafe_allow_html=True)
        st.markdown('<div class="flow-title">🔄 REPOSIÇÃO E BALANÇO HÍDRICO</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="flow-column-content">', unsafe_allow_html=True)
            st.markdown(f'<div class="flow-value">🚰 {formatar_numero(reposicao, 3)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="flow-unit">Reposição Total</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="flow-column-content">', unsafe_allow_html=True)
            st.markdown(f'<div class="flow-value">♻️ {formatar_numero(ciclos, 2)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="flow-unit">Ciclos de Concentração</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="flow-column-content">', unsafe_allow_html=True)
            st.markdown(f'<div class="flow-value">⏱️ {formatar_numero(HTI, 2)}</div>', unsafe_allow_html=True)
            st.markdown('<div class="flow-unit">HTI (Tempo Retenção)</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Informações adicionais
        st.markdown("---")
        col4, col5 = st.columns(2)
        with col4:
            st.markdown(f'<div style="color: #555; font-size: 16px; text-align: center;"><strong>🏊 Volume Estático:</strong> {formatar_numero(Vol_estatico, 2)} m³</div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div style="color: #555; font-size: 16px; text-align: center;"><strong>⚖️ Balanço:</strong><br>💨 Evaporação ({formatar_numero(evaporacao, 3)}) +<br>💧 Perda Líquida ({formatar_numero(perda_liquida, 3)}) =<br>🚰 Reposição ({formatar_numero(reposicao, 3)}) m³/h</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Fecha flow-step
        
        st.markdown('</div>', unsafe_allow_html=True)  # Fecha flow-diagram
        
        # Botões para ações
        st.markdown("---")
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            if st.button("🔄 Novo Cálculo", use_container_width=True):
                st.session_state.calcular = False
                st.rerun()
        
        with col_b2:
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
            csv = export_df.to_csv(index=False, sep=';', decimal=',')
            
            st.download_button(
                label="📥 Exportar Dados",
                data=csv,
                file_name="resultados_torre_resfriamento.csv",
                mime="text/csv",
                use_container_width=True
            )
        
    except Exception as e:
        st.error(f"Erro nos cálculos: {str(e)}")

else:
    # Tela inicial
    st.markdown("## 📋 Instruções")
    
    st.markdown('<div class="instruction-box">', unsafe_allow_html=True)
    st.markdown("""
    **Para usar a calculadora:**
    
    1. **Preencha todos os parâmetros** na barra lateral
    2. **Insira valores** para os 5 parâmetros químicos (Torre e Reposição)
    3. **Selecione qual ciclo** de concentração usar nos cálculos
    4. **Clique em 🚀 CALCULAR** para ver o fluxo da torre
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("### 🔬 Parâmetros Químicos Disponíveis")
    
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown('''
        <div class="param-box">
            <div class="param-title" style="text-align: center;">🔬 Sílica</div>
            <div class="param-unit" style="text-align: center;">ppm</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div class="param-box">
            <div class="param-title" style="text-align: center;">🧪 Cloreto</div>
            <div class="param-unit" style="text-align: center;">ppm</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
        <div class="param-box">
            <div class="param-title" style="text-align: center;">💎 Dureza Total</div>
            <div class="param-unit" style="text-align: center;">ppm CaCO₃</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown('''
        <div class="param-box">
            <div class="param-title" style="text-align: center;">⚗️ Alcalinidade Total</div>
            <div class="param-unit" style="text-align: center;">ppm CaCO₃</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col5:
        st.markdown('''
        <div class="param-box">
            <div class="param-title" style="text-align: center;">🧲 Ferro Total</div>
            <div class="param-unit" style="text-align: center;">ppm</div>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("⚡ **Clique no botão CALCULAR na barra lateral para visualizar o fluxo da torre**")

# Rodapé
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; padding: 20px; font-size: 14px;'>🏭 Calculadora de Torre de Resfriamento • Diagrama de Fluxo • Versão 2.0</div>", unsafe_allow_html=True)
