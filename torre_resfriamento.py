import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculadora de Torre de Resfriamento", layout="wide")

def formatar_numero(valor, casas_decimais=3):
    """Formata número com vírgula como separador decimal"""
    try:
        if valor is None:
            return "0,0"
        
        if pd.isna(valor):
            return "0,0"
            
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
        font-size: 30px;
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
        padding: 25px;
        border-radius: 12px;
        margin: 15px;
        border-left: 6px solid #4CAF50;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
        height: 100%;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .metric-title {
        margin: 0 0 15px 0;
        color: #1f77b4;
        font-size: 20px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
        margin: 10px 0;
        line-height: 1.2;
    }
    .metric-unit {
        font-size: 32px;
        color: #7f8c8d;
        margin-top: 5px;
    }
    .result-title {
        color: #1f77b4;
        font-size: 28px;
        font-weight: bold;
        margin: 20px 0 30px 0;
        padding-bottom: 15px;
        border-bottom: 3px solid #eee;
        text-align: center;
    }
    .sidebar-header {
        color: #4CAF50;
        font-weight: bold;
        margin-top: 20px;
        font-size: 16px;
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
</style>
""", unsafe_allow_html=True)

st.title("📊 Calculadora de Torre de Resfriamento")
st.markdown("---")

# Inicializar estado da sessão
if 'calcular' not in st.session_state:
    st.session_state.calcular = False

# Sidebar para parâmetros de entrada
with st.sidebar:
    st.header("⚙️ Parâmetros de Entrada")
    
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
    st.markdown('<div class="result-title">📈 RESULTADOS DOS CÁLCULOS</div>', unsafe_allow_html=True)
    
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
        
        # Layout com duas colunas
        col1, col2 = st.columns(2)
        
        with col1:
            # Card 1: Delta Temperatura
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Delta Temperatura</div>
                <div class="metric-value">{formatar_numero(delta_T, 2)}</div>
                <div class="metric-unit">°C</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Card 2: Evaporação
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Evaporação</div>
                <div class="metric-value">{formatar_numero(evaporacao, 3)}</div>
                <div class="metric-unit">m³/h</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Card 3: Perda Líquida
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Perda Líquida</div>
                <div class="metric-value">{formatar_numero(perda_liquida, 3)}</div>
                <div class="metric-unit">m³/h</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Card 4: HTI
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">HTI</div>
                <div class="metric-value">{formatar_numero(HTI, 2)}</div>
                <div class="metric-unit">horas</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            # Card 5: Perda por Arraste
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Perda por Arraste</div>
                <div class="metric-value">{formatar_numero(perda_arraste, 3)}</div>
                <div class="metric-unit">m³/h</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Card 6: Purga do Sistema
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Purga do Sistema</div>
                <div class="metric-value">{formatar_numero(purgas, 3)}</div>
                <div class="metric-unit">m³/h</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Card 7: Reposição
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Reposição</div>
                <div class="metric-value">{formatar_numero(reposicao, 3)}</div>
                <div class="metric-unit">m³/h</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Card 8: Ciclos de Concentração
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-title">Ciclos de Concentração</div>
                <div class="metric-value">{formatar_numero(ciclos, 2)}</div>
                <div class="metric-unit">vezes</div>
            </div>
            ''', unsafe_allow_html=True)
        
        # Botões para ações
        st.markdown('<div class="button-row">', unsafe_allow_html=True)
        
        col_b1, col_b2 = st.columns(2)
        
        with col_b1:
            if st.button("🔄 Novo Cálculo", use_container_width=True):
                st.session_state.calcular = False
                st.rerun()
        
        with col_b2:
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
            csv = export_df.to_csv(index=False, sep=';', decimal=',')
            
            st.download_button(
                label="📥 Exportar CSV",
                data=csv,
                file_name="resultados_torre_resfriamento.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Erro nos cálculos: {str(e)}")

else:
    # Tela inicial simples
    st.markdown("""
    <div style="text-align: center; padding: 60px 20px;">
        <h2 style="color: #1f77b4; margin-bottom: 40px;">🏭 Calculadora de Torre de Resfriamento</h2>
        
        <div style="max-width: 800px; margin: 0 auto;">
            <h3 style="color: #4CAF50; margin-bottom: 30px;">📋 Como Usar</h3>
            
            <div style="display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 40px;">
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); text-align: left;">
                    <h4 style="color: #4CAF50; margin-bottom: 10px;">1️⃣ Preencha os Parâmetros</h4>
                    <p>Insira todos os dados na <strong>barra lateral</strong></p>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); text-align: left;">
                    <h4 style="color: #4CAF50; margin-bottom: 10px;">2️⃣ Insira Valores Químicos</h4>
                    <p>Digite os valores para os 5 parâmetros (Torre e Reposição)</p>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); text-align: left;">
                    <h4 style="color: #4CAF50; margin-bottom: 10px;">3️⃣ Selecione o Ciclo</h4>
                    <p>Escolha qual ciclo de concentração usar nos cálculos</p>
                </div>
                
                <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); text-align: left;">
                    <h4 style="color: #4CAF50; margin-bottom: 10px;">4️⃣ Calcule</h4>
                    <p>Clique em <strong>🚀 CALCULAR</strong> para ver os resultados</p>
                </div>
            </div>
            
            <div style="background-color: #e8f5e9; padding: 25px; border-radius: 12px; margin: 30px 0;">
                <h4 style="color: #2e7d32; margin-bottom: 20px;">🔬 Parâmetros Químicos</h4>
                <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 15px; margin-bottom: 15px;">
                    <span style="background-color: white; padding: 10px 20px; border-radius: 25px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); font-weight: 500;">Sílica</span>
                    <span style="background-color: white; padding: 10px 20px; border-radius: 25px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); font-weight: 500;">Cloreto</span>
                    <span style="background-color: white; padding: 10px 20px; border-radius: 25px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); font-weight: 500;">Dureza Total</span>
                </div>
                <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 15px;">
                    <span style="background-color: white; padding: 10px 20px; border-radius: 25px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); font-weight: 500;">Alcalinidade Total</span>
                    <span style="background-color: white; padding: 10px 20px; border-radius: 25px; box-shadow: 0 3px 6px rgba(0,0,0,0.1); font-weight: 500;">Ferro Total</span>
                </div>
            </div>
            
            <p style="color: #666; font-style: italic; margin-top: 40px; font-size: 18px;">
                ⚡ <strong>Clique em CALCULAR na barra lateral para começar</strong>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Rodapé
st.markdown("---")
st.markdown("<div style='text-align: center; color: #666; padding: 20px;'>📊 <strong>Calculadora de Torre de Resfriamento</strong> • Otimização de Sistemas</div>", unsafe_allow_html=True)
