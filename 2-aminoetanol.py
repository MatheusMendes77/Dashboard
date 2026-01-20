import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Dados experimentais de viscosidade (cP) vs temperatura (°C) para 2-aminoetanol
# Fonte: Yaws' Handbook Properties & outros estudos
temperatura_exp = np.array([20, 25, 30, 40, 50, 60, 70, 80, 90, 100])  # °C
viscosidade_exp = np.array([24.1, 18.4, 14.2, 8.9, 5.9, 4.1, 3.0, 2.3, 1.8, 1.4])  # cP

# Interpolação para curva suave
interp_func = interp1d(temperatura_exp, viscosidade_exp, kind='cubic', fill_value='extrapolate')
temperatura_suave = np.linspace(20, 100, 200)
viscosidade_suave = interp_func(temperatura_suave)

# Configuração do gráfico
plt.figure(figsize=(10, 6))
plt.plot(temperatura_suave, viscosidade_suave, 'b-', linewidth=2.5, label='Curva interpolada')
plt.plot(temperatura_exp, viscosidade_exp, 'ro', markersize=8, label='Dados experimentais')

# Detalhes do gráfico
plt.title('Viscosidade do 2-Aminoetanol vs Temperatura', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Temperatura (°C)', fontsize=14)
plt.ylabel('Viscosidade Dinâmica (cP)', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=12, loc='upper right')
plt.xlim(20, 100)
plt.ylim(0, 26)

# Anotações importantes
plt.text(30, 22, '2-Aminoetanol\n(MEA - Monoetanolamina)', fontsize=12, 
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
plt.text(80, 8, 'Viscosidade diminui\nexponencialmente\ncom aumento da\ntemperatura', 
         fontsize=10, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Informações técnicas
props = dict(boxstyle='round', facecolor='lightgray', alpha=0.7)
info_text = 'Dados experimentais a 1 atm\nViscosidade em centipoise (cP)\n1 cP = 1 mPa·s'
plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
         fontsize=10, verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()
