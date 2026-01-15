import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Configuração da página
st.set_page_config(
    page_title="Calculadora de Trocadores de Calor",
    page_icon="🔥",
    layout="wide"
)

# Título e descrição
st.title("🔥 Calculadora Completa de Trocadores de Calor")
st.markdown("""
Esta ferramenta calcula todos os parâmetros importantes para análise de trocadores de calor:
- Número de Reynolds
- Duty Térmico (Q)
- Coeficiente Global U
- Fator de Fouling
- Monitoramento por Queda de Pressão
""")

# Sidebar para seleção do tipo de cálculo
st.sidebar.header("🔧 Configurações")
calc_type = st.sidebar.selectbox(
    "Tipo de Cálculo",
    ["Análise Completa", "Reynolds & Duty", "Fouling & Monitoramento", "Vapor-Líquido"]
)

# Funções de cálculo
def calculate_reynolds(d, v, rho, mu):
    """Calcula número de Reynolds"""
    Re = (rho * v * d) / mu
    return Re

def calculate_duty(m, cp, delta_T):
    """Calcula duty térmico"""
    Q = m * cp * delta_T
    return Q

def calculate_lmtd(T1_in, T1_out, T2_in, T2_out, flow_type="counter"):
    """Calcula LMTD para correntes paralelas ou contracorrente"""
    if flow_type == "counter":
        delta_T1 = T1_in - T2_out
        delta_T2 = T1_out - T2_in
    else:  # parallel
        delta_T1 = T1_in - T2_in
        delta_T2 = T1_out - T2_out
    
    if delta_T1 <= 0 or delta_T2 <= 0:
        return 0
    elif abs(delta_T1 - delta_T2) < 1e-6:
        return delta_T1
    
    LMTD = (delta_T1 - delta_T2) / np.log(delta_T1 / delta_T2)
    return LMTD

def calculate_u_value(Q, A, LMTD):
    """Calcula coeficiente global U"""
    if A > 0 and LMTD > 0:
        U = Q / (A * LMTD)
    else:
        U = 0
    return U

def calculate_fouling(U_dirty, U_clean):
    """Calcula fator de fouling"""
    if U_dirty > 0 and U_clean > 0:
        R_f = (1/U_dirty) - (1/U_clean)
    else:
        R_f = 0
    return R_f

def calculate_flow_coefficient(F_clean, dP_clean, dP_current, F_current=None):
    """Calcula coeficiente de vazão e estimativa"""
    if dP_clean > 0:
        C_clean = F_clean / np.sqrt(dP_clean)
        F_estimated = C_clean * np.sqrt(dP_current)
        
        if F_current is not None and F_current > 0:
            fouling_percentage = (1 - (np.sqrt(dP_clean)/np.sqrt(dP_current))) * 100
            deviation = ((F_estimated - F_current) / F_current) * 100
        else:
            fouling_percentage = 0
            deviation = 0
            
        return C_clean, F_estimated, fouling_percentage, deviation
    else:
        return 0, 0, 0, 0

# Container principal
if calc_type == "Análise Completa":
    st.header("📈 Análise Completa do Trocador")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔵 Lado Quente (Fluido 1)")
        m1 = st.number_input("Vazão mássica (kg/s)", min_value=0.0, value=10.0, key="m1")
        cp1 = st.number_input("Calor específico (J/kg·K)", min_value=0.0, value=4180.0, key="cp1")
        T1_in = st.number_input("Temperatura entrada (°C)", value=80.0, key="T1_in")
        T1_out = st.number_input("Temperatura saída (°C)", value=50.0, key="T1_out")
        rho1 = st.number_input("Densidade (kg/m³)", min_value=0.0, value=998.0, key="rho1")
        mu1 = st.number_input("Viscosidade (Pa·s)", min_value=0.0, value=0.001, key="mu1", format="%.6f")
        
    with col2:
        st.subheader("🔴 Lado Frio (Fluido 2)")
        m2 = st.number_input("Vazão mássica (kg/s)", min_value=0.0, value=12.0, key="m2")
        cp2 = st.number_input("Calor específico (J/kg·K)", min_value=0.0, value=4180.0, key="cp2")
        T2_in = st.number_input("Temperatura entrada (°C)", value=20.0, key="T2_in")
        T2_out = st.number_input("Temperatura saída (°C)", value=45.0, key="T2_out")
        rho2 = st.number_input("Densidade (kg/m³)", min_value=0.0, value=998.0, key="rho2")
        mu2 = st.number_input("Viscosidade (Pa·s)", min_value=0.0, value=0.001, key="mu2", format="%.6f")
    
    st.subheader("📐 Geometria do Trocador")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        d_tube = st.number_input("Diâmetro interno do tubo (m)", min_value=0.0, value=0.05, key="d_tube")
        v1 = st.number_input("Velocidade lado quente (m/s)", min_value=0.0, value=1.5, key="v1")
        v2 = st.number_input("Velocidade lado frio (m/s)", min_value=0.0, value=1.2, key="v2")
    
    with col4:
        A_total = st.number_input("Área total de transferência (m²)", min_value=0.0, value=50.0, key="A_total")
        flow_type = st.selectbox("Tipo de escoamento", ["Contracorrente", "Paralelo"])
    
    with col5:
        U_clean_design = st.number_input("U limpo de projeto (W/m²·K)", min_value=0.0, value=800.0, key="U_clean")
    
    # Cálculos
    if st.button("🎯 Calcular Tudo", type="primary"):
        # Reynolds
        Re1 = calculate_reynolds(d_tube, v1, rho1, mu1)
        Re2 = calculate_reynolds(d_tube, v2, rho2, mu2)
        
        # Duties
        Q1 = calculate_duty(m1, cp1, T1_in - T1_out)
        Q2 = calculate_duty(m2, cp2, T2_out - T2_in)
        Q_avg = (Q1 + Q2) / 2
        
        # LMTD e U
        flow_type_code = "counter" if flow_type == "Contracorrente" else "parallel"
        LMTD = calculate_lmtd(T1_in, T1_out, T2_in, T2_out, flow_type_code)
        U_operational = calculate_u_value(Q_avg, A_total, LMTD)
        
        # Fouling
        R_f = calculate_fouling(U_operational, U_clean_design)
        
        # Resultados
        st.success("Cálculos completos!")
        
        # Display results in columns
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.metric("Duty Térmico (Q)", f"{Q_avg/1000:.1f} kW")
            st.metric("LMTD", f"{LMTD:.1f} °C")
            st.metric("U Operacional", f"{U_operational:.1f} W/m²·K")
            
        with col_r2:
            st.metric("Reynolds Lado Quente", f"{Re1:,.0f}")
            st.metric("Reynolds Lado Frio", f"{Re2:,.0f}")
            regime1 = "Turbulento" if Re1 > 4000 else "Laminar" if Re1 < 2300 else "Transição"
            regime2 = "Turbulento" if Re2 > 4000 else "Laminar" if Re2 < 2300 else "Transição"
            st.metric("Regime Lado Quente", regime1)
            st.metric("Regime Lado Frio", regime2)
            
        with col_r3:
            st.metric("Fator de Fouling", f"{R_f*1e4:.3f} ×10⁻⁴ m²·K/W")
            fouling_percent = ((1/U_operational - 1/U_clean_design) / (1/U_clean_design)) * 100
            st.metric("Aumento Resistência", f"{fouling_percent:.1f}%")
            efficiency = (U_operational / U_clean_design) * 100 if U_clean_design > 0 else 0
            st.metric("Eficiência vs Projeto", f"{efficiency:.1f}%")
        
        # Balanço térmico
        st.subheader("⚖️ Balanço Térmico")
        balance_error = abs(Q1 - Q2) / max(Q1, Q2) * 100
        col_b1, col_b2, col_b3 = st.columns(3)
        
        with col_b1:
            st.metric("Q Lado Quente", f"{Q1/1000:.1f} kW")
        with col_b2:
            st.metric("Q Lado Frio", f"{Q2/1000:.1f} kW")
        with col_b3:
            st.metric("Diferença", f"{balance_error:.1f}%", 
                     delta=f"{balance_error:.1f}%", 
                     delta_color="inverse" if balance_error > 5 else "normal")
        
        # Gráfico de temperaturas
        fig, ax = plt.subplots(figsize=(10, 6))
        positions = [0, 1]
        if flow_type == "Contracorrente":
            hot_temps = [T1_in, T1_out]
            cold_temps = [T2_out, T2_in]
        else:
            hot_temps = [T1_in, T1_out]
            cold_temps = [T2_in, T2_out]
            
        ax.plot(positions, hot_temps, 'r-o', linewidth=2, markersize=8, label='Lado Quente')
        ax.plot(positions, cold_temps, 'b-s', linewidth=2, markersize=8, label='Lado Frio')
        ax.fill_between(positions, hot_temps, cold_temps, alpha=0.2, color='gray')
        ax.set_xlabel('Posição no Trocador')
        ax.set_ylabel('Temperatura (°C)')
        ax.set_title('Perfil de Temperaturas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

elif calc_type == "Reynolds & Duty":
    st.header("📊 Cálculo de Reynolds e Duty")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Propriedades do Fluido")
        d = st.number_input("Diâmetro (m)", min_value=0.0, value=0.05)
        v = st.number_input("Velocidade (m/s)", min_value=0.0, value=1.5)
        rho = st.number_input("Densidade (kg/m³)", min_value=0.0, value=998.0)
        mu = st.number_input("Viscosidade (Pa·s)", min_value=0.0, value=0.001, format="%.6f")
        
        if st.button("Calcular Reynolds"):
            Re = calculate_reynolds(d, v, rho, mu)
            regime = "Turbulento" if Re > 4000 else "Laminar" if Re < 2300 else "Transição"
            
            st.metric("Número de Reynolds", f"{Re:,.0f}")
            st.metric("Regime de Escoamento", regime)
            
            # Informações adicionais
            st.info(f"""
            **Interpretação:**
            - Re < 2.300: Escoamento Laminar
            - 2.300 < Re < 4.000: Transição
            - Re > 4.000: Escoamento Turbulento
            """)
    
    with col2:
        st.subheader("Cálculo do Duty")
        m = st.number_input("Vazão mássica (kg/s)", min_value=0.0, value=10.0)
        cp = st.number_input("Calor específico (J/kg·K)", min_value=0.0, value=4180.0)
        T_in = st.number_input("T_in (°C)", value=80.0)
        T_out = st.number_input("T_out (°C)", value=50.0)
        
        if st.button("Calcular Duty"):
            Q = calculate_duty(m, cp, T_in - T_out)
            
            st.metric("Duty Térmico", f"{Q/1000:.2f} kW")
            st.metric("Por kg de fluido", f"{Q/m/1000:.2f} kJ/kg")
            
            # Equivalências
            st.info(f"""
            **Equivalências:**
            - {Q/1000:.1f} kW
            - {Q/3600000:.3f} MW
            - {Q*0.0009478:.0f} BTU/h
            """)

elif calc_type == "Fouling & Monitoramento":
    st.header("🔄 Monitoramento de Fouling por Queda de Pressão")
    
    tab1, tab2, tab3 = st.tabs(["📐 Cálculo de Fouling", "📊 Monitoramento ΔP", "📈 Tendências"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Condições Limpas (Projeto)")
            U_clean = st.number_input("U limpo (W/m²·K)", min_value=0.0, value=1000.0, key="Uc")
            dP_clean = st.number_input("ΔP limpa (Pa)", min_value=0.0, value=25000.0, key="dPc")
            F_clean = st.number_input("Vazão limpa (kg/s)", min_value=0.0, value=10.0, key="Fc")
            
        with col2:
            st.subheader("Condições Atuais (Operação)")
            U_dirty = st.number_input("U operacional (W/m²·K)", min_value=0.0, value=600.0, key="Ud")
            dP_current = st.number_input("ΔP atual (Pa)", min_value=0.0, value=40000.0, key="dPnow")
            F_current = st.number_input("Vazão atual (kg/s)", min_value=0.0, value=10.0, key="Fnow")
        
        if st.button("Calcular Fouling"):
            # Cálculo de fouling por U
            R_f = calculate_fouling(U_dirty, U_clean)
            
            # Cálculo por ΔP
            C_clean, F_estimated, fouling_percent, deviation = calculate_flow_coefficient(
                F_clean, dP_clean, dP_current, F_current
            )
            
            st.success("Resultados do Fouling")
            
            col_r1, col_r2 = st.columns(2)
            
            with col_r1:
                st.metric("Resistência Fouling", f"{R_f*1e4:.3f} ×10⁻⁴ m²·K/W")
                st.metric("Redução U", f"{(1 - U_dirty/U_clean)*100:.1f}%")
                st.metric("Coeficiente C_limpo", f"{C_clean:.6f}")
                
            with col_r2:
                st.metric("Vazão Estimada", f"{F_estimated:.2f} kg/s")
                st.metric("Fouling por ΔP", f"{founing_percent:.1f}%")
                st.metric("Desvio Vazão", f"{deviation:.1f}%", 
                         delta=f"{deviation:.1f}%", 
                         delta_color="inverse")
            
            # Recomendação
            st.subheader("🎯 Recomendação")
            if fouling_percent > 20:
                st.error("⚠️ **ALERTA:** Fouling severo (>20%). Programar limpeza imediata.")
            elif founing_percent > 10:
                st.warning("⚠️ **ATENÇÃO:** Fouling moderado (10-20%). Monitorar de perto.")
            else:
                st.success("✅ Condição aceitável (<10%). Continuar operação normal.")
    
    with tab2:
        st.subheader("Monitoramento Contínuo por ΔP")
        
        # Simulação de dados históricos
        days = list(range(0, 31, 3))
        dP_clean_ref = 25000
        fouling_growth = [0, 5, 8, 12, 18, 22, 28, 35, 42, 50, 58]
        
        dP_values = [dP_clean_ref * (1 + f/100) for f in fouling_growth]
        U_values = [1000 * (1 - f/100) for f in fouling_growth]
        
        # Criar dataframe
        df_monitoring = pd.DataFrame({
            'Dia': days,
            'ΔP (Pa)': dP_values,
            'U (W/m²K)': U_values,
            'Fouling %': fouling_growth
        })
        
        st.dataframe(df_monitoring.style.format({
            'ΔP (Pa)': '{:,.0f}',
            'U (W/m²K)': '{:.0f}',
            'Fouling %': '{:.1f}'
        }))
        
        # Gráfico
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color1 = 'tab:red'
        ax1.set_xlabel('Dias de Operação')
        ax1.set_ylabel('ΔP (Pa)', color=color1)
        ax1.plot(days, dP_values, 'o-', color=color1, linewidth=2)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.axhline(y=dP_clean_ref, color=color1, linestyle='--', alpha=0.5, label='ΔP Limpa')
        
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('U (W/m²K)', color=color2)
        ax2.plot(days, U_values, 's-', color=color2, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.axhline(y=1000, color=color2, linestyle='--', alpha=0.5, label='U Limpo')
        
        fig.tight_layout()
        st.pyplot(fig)

elif calc_type == "Vapor-Líquido":
    st.header("💨 Trocadores Vapor-Líquido / Condensadores")
    
    process_type = st.radio("Tipo de Processo:", 
                          ["Aquecimento com Vapor", "Condensação Total", "Superaquecedor"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Lado do Vapor")
        m_vapor = st.number_input("Vazão de vapor (kg/s)", min_value=0.0, value=2.0)
        T_vapor_in = st.number_input("T entrada vapor (°C)", value=150.0)
        P_vapor = st.number_input("Pressão vapor (bar)", value=4.0)
        
        if process_type == "Aquecimento com Vapor":
            x_vapor_out = st.slider("Título vapor saída", 0.0, 1.0, 0.0)
        elif process_type == "Condensação Total":
            T_cond_out = st.number_input("T condensado saída (°C)", value=110.0)
        else:
            T_vapor_out = st.number_input("T vapor saída (°C)", value=180.0)
    
    with col2:
        st.subheader("Lado do Líquido")
        m_liquid = st.number_input("Vazão líquido (kg/s)", min_value=0.0, value=20.0)
        cp_liquid = st.number_input("Cp líquido (J/kg·K)", min_value=0.0, value=4180.0)
        T_liq_in = st.number_input("T entrada líquido (°C)", value=25.0)
        T_liq_out = st.number_input("T saída líquido (°C)", value=85.0)
    
    # Propriedades do vapor (valores típicos)
    st.subheader("📊 Propriedades Termodinâmicas")
    col_p1, col_p2, col_p3 = st.columns(3)
    
    with col_p1:
        h_fg = st.number_input("Calor latente (kJ/kg)", value=2200.0) * 1000
    with col_p2:
        cp_vapor = st.number_input("Cp vapor (J/kg·K)", value=2000.0)
    with col_p3:
        cp_condensado = st.number_input("Cp condensado (J/kg·K)", value=4200.0)
    
    if st.button("Calcular Vapor-Líquido"):
        # Duty do lado líquido
        Q_liquid = calculate_duty(m_liquid, cp_liquid, T_liq_out - T_liq_in)
        
        # Duty do lado vapor (depende do processo)
        if process_type == "Aquecimento com Vapor":
            # Vapor condensa parcialmente
            Q_vapor = m_vapor * (h_fg * (1 - x_vapor_out) + 
                                cp_condensado * (T_vapor_in - 100) * (1 - x_vapor_out))
            process_desc = f"Vapor condensa de título 1.0 para {x_vapor_out:.2f}"
            
        elif process_type == "Condensação Total":
            # Condensação total + subresfriamento
            Q_vapor = m_vapor * (h_fg + cp_condensado * (100 - T_cond_out))
            process_desc = f"Condensação total + subresfriamento a {T_cond_out}°C"
            
        else:  # Superaquecedor
            # Apenas resfriamento do vapor superaquecido
            Q_vapor = m_vapor * cp_vapor * (T_vapor_in - T_vapor_out)
            process_desc = f"Resfriamento vapor de {T_vapor_in} para {T_vapor_out}°C"
        
        # Resultados
        st.success(f"Cálculo para: {process_type}")
        st.info(f"Processo: {process_desc}")
        
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.metric("Duty Líquido", f"{Q_liquid/1000:.1f} kW")
            st.metric("Duty Vapor", f"{Q_vapor/1000:.1f} kW")
            balance_error = abs(Q_liquid - Q_vapor) / max(Q_liquid, Q_vapor) * 100
            st.metric("Balanço", f"{balance_error:.1f}%")
            
        with col_r2:
            steam_rate = Q_liquid / h_fg if h_fg > 0 else 0
            st.metric("Consumo vapor teórico", f"{steam_rate:.3f} kg/s")
            st.metric("Vapor por kg líquido", f"{steam_rate/m_liquid*1000:.1f} g/kg")
            efficiency = (Q_liquid / Q_vapor) * 100 if Q_vapor > 0 else 0
            st.metric("Eficiência", f"{efficiency:.1f}%")

# Rodapé
st.markdown("---")
st.markdown("""
**📋 Instruções:**
1. Selecione o tipo de cálculo na sidebar
2. Insira os dados nos campos
3. Clique no botão calcular
4. Interprete os resultados
""")

st.caption("Desenvolvido para Engenharia de Processos - Cálculos de Trocadores de Calor")
