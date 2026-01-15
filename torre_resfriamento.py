import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calculadora de Torre de Resfriamento", layout="wide")

st.title("📊 Calculadora de Torre de Resfriamento")

st.markdown("### 🔧 Parâmetros de Entrada")

col1, col2, col3 = st.columns(3)

with col1:
    VZ_rec = st.number_input("Vazão de Recirculação (m³/h)", min_value=0.0, value=1000.0, step=50.0)
    T_retorno = st.number_input("Temperatura de Retorno (°C)", min_value=0.0, value=40.0, step=1.0)
    ciclos = st.number_input("Ciclos de Concentração (vezes)", min_value=1.0, value=3.0, step=0.5)

with col2:
    Vol_estatico = st.number_input("Volume Estático (m³)", min_value=0.0, value=50.0, step=5.0)
    T_bacia = st.number_input("Temperatura de Bacia (°C)", min_value=0.0, value=30.0, step=1.0)
    perc_arraste = st.number_input("% Arraste", min_value=0.0, max_value=100.0, value=0.1, step=0.01, format="%.4f")

with col3:
    perc_utilizacao = st.number_input("% Utilização", min_value=0.0, max_value=100.0, value=100.0, step=5.0, format="%.1f")
    perc_utilizacao_decimal = perc_utilizacao / 100.0

st.markdown("---")
st.markdown("### 🔄 Cálculo dos Ciclos de Concentração")

parametros_opcoes = {
    "Sílica (ppm)": "Sílica",
    "Cloreto (ppm)": "Cloreto",
    "Dureza Total (ppm como CaCO₃)": "Dureza Total",
    "Alcalinidade Total (ppm como CaCO₃)": "Alcalinidade Total",
    "Ferro Total (ppm)": "Ferro Total"
}

param_selecionado = st.selectbox("Selecione o parâmetro para cálculo dos ciclos:", list(parametros_opcoes.keys()))

col_a, col_b = st.columns(2)
with col_a:
    param_torre = st.number_input(f"{param_selecionado} na Torre (ppm)", min_value=0.0, value=150.0, step=10.0)
with col_b:
    param_reposicao = st.number_input(f"{param_selecionado} na Reposição (ppm)", min_value=0.0, value=50.0, step=5.0)

if param_reposicao > 0:
    ciclos_calculado = param_torre / param_reposicao
    st.info(f"**Ciclos de Concentração calculados:** {ciclos_calculado:.2f} vezes")
    usar_ciclos_calculado = st.checkbox("Usar ciclos calculados no lugar do valor inserido acima?", value=False)
    if usar_ciclos_calculado:
        ciclos = ciclos_calculado
else:
    st.warning("Valor na reposição deve ser maior que zero para cálculo dos ciclos.")

st.markdown("---")
st.markdown("## 📈 Resultados dos Cálculos")

# 1. Delta Temperatura
delta_T = T_retorno - T_bacia

# 2. Evaporação
evaporacao = VZ_rec * delta_T * (0.85 / 556) * perc_utilizacao_decimal

# 3. Perda Líquida
if ciclos > 1:
    perda_liquida = evaporacao / (ciclos - 1)
else:
    perda_liquida = 0.0
    st.error("Ciclos de concentração devem ser maiores que 1 para cálculo da perda líquida.")

# 4. HTI (Índice de Tempo de Retenção)
if perda_liquida > 0:
    HTI = 0.693 * (Vol_estatico / perda_liquida)
else:
    HTI = 0.0

# 5. Perda por Arraste
perda_arraste = (perc_arraste / 100.0) * VZ_rec * perc_utilizacao_decimal

# 6. Purga do Sistema
purgas = perda_liquida - perda_arraste

# 7. Reposição
reposicao = evaporacao + perda_liquida

# Tabela de resultados
resultados = pd.DataFrame({
    "Fórmula": [
        "Delta Temperatura (°C)",
        "Evaporação (m³/h)",
        "Perda Líquida (m³/h)",
        "HTI (h)",
        "Perda por Arraste (m³/h)",
        "Purga do Sistema (m³/h)",
        "Reposição (m³/h)"
    ],
    "Valor": [
        f"{delta_T:.2f}",
        f"{evaporacao:.3f}",
        f"{perda_liquida:.3f}",
        f"{HTI:.2f}",
        f"{perda_arraste:.3f}",
        f"{purgas:.3f}",
        f"{reposicao:.3f}"
    ],
    "Descrição": [
        "Diferença entre temperatura de retorno e bacia",
        "Vazão evaporada na torre",
        "Água perdida total (inclui purga e arraste)",
        "Índice de Tempo de Retenção",
        "Água perdida por arraste de gotículas",
        "Água descartada para controle de sólidos",
        "Vazão de água reposta na torre"
    ]
})

st.table(resultados)

st.markdown("---")
st.markdown("### 📋 Resumo Operacional")

col_res1, col_res2 = st.columns(2)

with col_res1:
    st.metric("Ciclos de Concentração", f"{ciclos:.2f} vezes")
    st.metric("Delta T", f"{delta_T:.2f} °C")
    st.metric("Evaporação", f"{evaporacao:.3f} m³/h")
    st.metric("Reposição", f"{reposicao:.3f} m³/h")

with col_res2:
    st.metric("HTI", f"{HTI:.2f} h")
    st.metric("Perda Líquida", f"{perda_liquida:.3f} m³/h")
    st.metric("Purga", f"{purgas:.3f} m³/h")
    st.metric("Perda por Arraste", f"{perda_arraste:.3f} m³/h")

# Cálculos de verificação
st.markdown("---")
st.markdown("### ✅ Verificação de Consistência")
st.markdown(f"**Evaporação + Perda Líquida = Reposição:** {evaporacao:.3f} + {perda_liquida:.3f} = {reposicao:.3f} m³/h")
st.markdown(f"**Perda Líquida = Purga + Arraste:** {perda_liquida:.3f} = {purgas:.3f} + {perda_arraste:.3f} m³/h")

# Instruções
st.markdown("---")
st.markdown("### 📝 Instruções:")
st.markdown("""
1. Preencha todos os parâmetros de entrada nas três colunas superiores
2. Selecione o parâmetro para cálculo dos ciclos de concentração
3. Insira os valores do parâmetro na torre e na reposição
4. Os resultados serão calculados automaticamente
5. Você pode optar por usar os ciclos calculados automaticamente
""")

# Botão para limpar/recarregar
if st.button("🔄 Limpar/Recalcular"):
    st.rerun()
