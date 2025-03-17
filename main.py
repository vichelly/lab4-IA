import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl

# Variáveis de Entrada (Antecedent)
calorias_diarias = ctrl.Antecedent(np.arange(0, 4001, 1), 'calorias_diarias')
atividade_semana = ctrl.Antecedent(np.arange(0, 8, 1), 'atividade_semana')
tempo_atividade = ctrl.Antecedent(np.arange(0, 181, 1), 'tempo_atividade')

# Variável de saída (Consequent)
risco_obesidade = ctrl.Consequent(np.arange(0, 101, 1), 'risco_obesidade') # Risco de 0 a 100
imc_estimado = ctrl.Consequent(np.arange(15, 41, 1), 'imc_estimado') # Estimativa do IMC

# automf -> Atribuição de categorias automaticamente
calorias_diarias.automf(names=['baixo', 'moderado', 'alto'])
atividade_semana.automf(names=['nenhuma', 'pouca', 'moderada', 'alta'])
tempo_atividade.automf(names=['curto', 'medio', 'longo'])

# Funções de pertinência para risco de obesidade
risco_obesidade['baixo'] = fuzz.trimf(risco_obesidade.universe, [0, 25, 50])
risco_obesidade['medio'] = fuzz.trimf(risco_obesidade.universe, [25, 50, 75])
risco_obesidade['alto'] = fuzz.trimf(risco_obesidade.universe, [50, 75, 100])

# Funções de pertinência para IMC estimado
imc_estimado['normal'] = fuzz.trimf(imc_estimado.universe, [18.5, 22.5, 25])
imc_estimado['sobrepeso'] = fuzz.trimf(imc_estimado.universe, [25, 27.5, 30])
imc_estimado['obeso_I'] = fuzz.trimf(imc_estimado.universe, [30, 32.5, 35])
imc_estimado['obeso_II'] = fuzz.trimf(imc_estimado.universe, [35, 37.5, 40])

# Regras Fuzzy
regra_1 = ctrl.Rule(calorias_diarias['alto'] & atividade_semana['nenhuma'] & tempo_atividade['curto'], (risco_obesidade['alto'], imc_estimado['obeso_II']))
regra_2 = ctrl.Rule(calorias_diarias['baixo'] & atividade_semana['alta'] & tempo_atividade['longo'], (risco_obesidade['baixo'], imc_estimado['normal']))
regra_3 = ctrl.Rule(calorias_diarias['moderado'] & atividade_semana['moderada'] & tempo_atividade['medio'], (risco_obesidade['medio'], imc_estimado['sobrepeso']))
regra_4 = ctrl.Rule(calorias_diarias['alto'] & atividade_semana['moderada'] & tempo_atividade['medio'], (risco_obesidade['alto'], imc_estimado['obeso_I']))
regra_5 = ctrl.Rule(calorias_diarias['alto'] & atividade_semana['alta'] & tempo_atividade['curto'], (risco_obesidade['medio'], imc_estimado['sobrepeso']))
regra_6 = ctrl.Rule(calorias_diarias['moderado'] & atividade_semana['nenhuma'] & tempo_atividade['curto'], (risco_obesidade['alto'], imc_estimado['obeso_I']))
regra_7 = ctrl.Rule(calorias_diarias['baixo'] & atividade_semana['pouca'] & tempo_atividade['medio'], (risco_obesidade['baixo'], imc_estimado['normal']))
regra_8 = ctrl.Rule(calorias_diarias['moderado'] & atividade_semana['alta'] & tempo_atividade['longo'], (risco_obesidade['baixo'], imc_estimado['normal']))
regra_9 = ctrl.Rule(calorias_diarias['baixo'] & atividade_semana['moderada'] & tempo_atividade['medio'], (risco_obesidade['baixo'], imc_estimado['normal']))
regra_10 = ctrl.Rule(calorias_diarias['moderado'] & atividade_semana['pouca'] & tempo_atividade['curto'], (risco_obesidade['medio'], imc_estimado['sobrepeso']))

controlador = ctrl.ControlSystem([regra_1, regra_2, regra_3, regra_4, regra_5, regra_6, regra_7, regra_8, regra_9, regra_10])

# Simulando
CalculoRisco = ctrl.ControlSystemSimulation(controlador)

calorias = int(input('Calorias diárias: '))
dias_atividade = int(input('Dias de atividade física por semana: '))
tempo_atividade_min = int(input('Tempo de atividade física por sessão (minutos): '))

CalculoRisco.input['calorias_diarias'] = calorias
CalculoRisco.input['atividade_semana'] = dias_atividade
CalculoRisco.input['tempo_atividade'] = tempo_atividade_min

CalculoRisco.compute()

valor_risco = CalculoRisco.output['risco_obesidade']
valor_imc = CalculoRisco.output['imc_estimado']

print("\nCalorias diárias: %d \nDias de atividade física por semana: %d \nTempo de atividade física por sessão: %d minutos" % (calorias, dias_atividade, tempo_atividade_min))
print("Risco de Obesidade: %5.2f" % valor_risco)
print("IMC Estimado: %5.2f" % valor_imc)

# Visualização dos gráficos
calorias_diarias.view(sim=CalculoRisco, title='Calorias Diárias')
atividade_semana.view(sim=CalculoRisco, title='Atividade na Semana')
tempo_atividade.view(sim=CalculoRisco, title='Tempo de Atividade')
risco_obesidade.view(sim=CalculoRisco, title='Risco de Obesidade')
imc_estimado.view(sim=CalculoRisco, title='IMC Estimado')

plt.show()