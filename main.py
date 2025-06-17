import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

class CakeDivisionSimulator:
    def __init__(self):
        self.results = {
            'player_1': [],
            'player_2': [],
            'player_3': []
        }
        self.deviations = []
        
    def generate_random_preferences(self) -> Dict[str, Dict[str, float]]:
        """Genera preferencias aleatorias para cada jugador sobre los componentes A y B"""
        preferences = {}
        for player in ['player_1', 'player_2', 'player_3']:
            # Valores aleatorios para componentes A y B
            pref_A = random.uniform(0.1, 2.0)
            pref_B = random.uniform(0.1, 2.0)
            preferences[player] = {'A': pref_A, 'B': pref_B}
        return preferences
    
    def generate_cake(self, T: int) -> List[str]:
        """Genera una torta aleatoria de tamaño T con componentes A y B"""
        return [random.choice(['A', 'B']) for _ in range(T)]
    
    def calculate_utility(self, cake_slice: List[str], preferences: Dict[str, float]) -> float:
        """Calcula la utilidad de un trozo de torta para un jugador específico"""
        total_utility = 0
        for component in cake_slice:
            total_utility += preferences[component]
        return total_utility
    
    def normalize_utilities(self, cake: List[str], all_preferences: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Normaliza las utilidades para que la utilidad total de cada jugador sea 1"""
        normalized_prefs = {}
        for player, prefs in all_preferences.items():
            total_cake_utility = self.calculate_utility(cake, prefs)
            if total_cake_utility > 0:
                normalized_prefs[player] = {
                    'A': prefs['A'] / total_cake_utility,
                    'B': prefs['B'] / total_cake_utility
                }
            else:
                normalized_prefs[player] = prefs.copy()
        return normalized_prefs
    
    def steinhaus_algorithm(self, cake: List[str], preferences: Dict[str, Dict[str, float]]) -> Tuple[List[List[str]], List[float], float]:
        """
        Implementa el algoritmo de Steinhaus para 3 jugadores
        Retorna: (trozos_asignados, utilidades_obtenidas, desviación_del_corte_exacto)
        """
        T = len(cake)
        
        # Paso 1: Jugador 1 encuentra su corte para obtener 1/3 de utilidad
        cumulative_utility_1 = 0
        target_utility_1 = 1/3
        k1 = 0
        
        for i in range(T):
            cumulative_utility_1 += preferences['player_1'][cake[i]]
            if cumulative_utility_1 >= target_utility_1:
                k1 = i
                break
        
        # Paso 2: Jugador 2 examina y hace su corte
        # Calcula utilidad del primer trozo [0:k1+1]
        piece1_utility_2 = self.calculate_utility(cake[0:k1+1], preferences['player_2'])
        
        # Jugador 2 busca su segundo corte k2 >= k1 para el trozo [k1+1:k2+1]
        cumulative_utility_2 = 0
        target_utility_2 = 1/3
        k2 = k1
        
        if k1 + 1 < T:
            for i in range(k1 + 1, T):
                cumulative_utility_2 += preferences['player_2'][cake[i]]
                if cumulative_utility_2 >= target_utility_2:
                    k2 = i
                    break
            else:
                k2 = T - 1
        
        # Definir los tres trozos
        piece1 = cake[0:k1+1] if k1 >= 0 else []
        piece2 = cake[k1+1:k2+1] if k2 > k1 else []
        piece3 = cake[k2+1:] if k2 < T-1 else []
        
        pieces = [piece1, piece2, piece3]
        
        # Calcular utilidades de cada trozo para cada jugador
        utilities_matrix = []
        for player in ['player_1', 'player_2', 'player_3']:
            player_utilities = []
            for piece in pieces:
                if piece:
                    utility = self.calculate_utility(piece, preferences[player])
                    player_utilities.append(utility)
                else:
                    player_utilities.append(0)
            utilities_matrix.append(player_utilities)
        
        # Paso 3: Asignación (Jugador 3 elige primero, luego 2, luego 1)
        available_pieces = [0, 1, 2]
        assignments = [None, None, None]  # Para jugadores 1, 2, 3
        
        # Jugador 3 elige el trozo que más le conviene
        best_piece_3 = max(available_pieces, key=lambda p: utilities_matrix[2][p])
        assignments[2] = best_piece_3
        available_pieces.remove(best_piece_3)
        
        # Jugador 2 elige de los restantes
        if available_pieces:
            best_piece_2 = max(available_pieces, key=lambda p: utilities_matrix[1][p])
            assignments[1] = best_piece_2
            available_pieces.remove(best_piece_2)
        
        # Jugador 1 toma lo que queda
        if available_pieces:
            assignments[0] = available_pieces[0]
        
        # Construir resultado final
        final_pieces = [pieces[assignments[i]] if assignments[i] is not None else [] for i in range(3)]
        final_utilities = [utilities_matrix[i][assignments[i]] if assignments[i] is not None else 0 for i in range(3)]
        
        # Calcular desviación del corte exacto (1/3 para cada uno)
        ideal_utility = 1/3
        deviation = sum(abs(util - ideal_utility) for util in final_utilities) / 3
        
        return final_pieces, final_utilities, deviation
    
    def run_simulation(self, T: int, N: int, verbose: bool = True):
        """Ejecuta la simulación completa"""
        if verbose:
            print(f"=== SIMULACIÓN DE REPARTO PROPORCIONAL ===")
            print(f"Tamaño de torta (T): {T}")
            print(f"Número de iteraciones (N): {N}")
            print(f"Algoritmo: Steinhaus para 3 jugadores")
            print("=" * 50)
        
        for iteration in range(N):
            if verbose:
                print(f"\n--- ITERACIÓN {iteration + 1} ---")
            
            # Generar torta y preferencias aleatorias
            cake = self.generate_cake(T)
            raw_preferences = self.generate_random_preferences()
            preferences = self.normalize_utilities(cake, raw_preferences)
            
            if verbose:
                print(f"Torta generada: {''.join(cake)}")
                print("Preferencias normalizadas:")
                for player, prefs in preferences.items():
                    print(f"  {player}: A={prefs['A']:.3f}, B={prefs['B']:.3f}")
            
            # Aplicar algoritmo de Steinhaus
            pieces, utilities, deviation = self.steinhaus_algorithm(cake, preferences)
            
            # Guardar resultados
            self.results['player_1'].append(utilities[0])
            self.results['player_2'].append(utilities[1])
            self.results['player_3'].append(utilities[2])
            self.deviations.append(deviation)
            
            if verbose:
                print("Resultados del reparto:")
                for i, (piece, utility) in enumerate(zip(pieces, utilities)):
                    print(f"  Jugador {i+1}: trozo={''.join(piece)} (longitud={len(piece)}), utilidad={utility:.4f}")
                print(f"Desviación del corte exacto: {deviation:.4f}")
        
        # Mostrar estadísticas finales
        self.print_final_statistics()
        
        # Graficar resultados
        self.plot_results()
    
    def print_final_statistics(self):
        """Imprime estadísticas finales de la simulación"""
        print("\n" + "=" * 50)
        print("ESTADÍSTICAS FINALES")
        print("=" * 50)
        
        for i, player in enumerate(['player_1', 'player_2', 'player_3'], 1):
            utilities = self.results[player]
            mean_util = np.mean(utilities)
            var_util = np.var(utilities)
            std_util = np.std(utilities)
            
            print(f"\nJugador {i}:")
            print(f"  Media de utilidad: {mean_util:.4f}")
            print(f"  Varianza: {var_util:.6f}")
            print(f"  Desviación estándar: {std_util:.4f}")
            print(f"  Utilidad mínima: {min(utilities):.4f}")
            print(f"  Utilidad máxima: {max(utilities):.4f}")
        
        print(f"\nDesviaciones del corte exacto:")
        print(f"  Media: {np.mean(self.deviations):.4f}")
        print(f"  Desviación estándar: {np.std(self.deviations):.4f}")
        print(f"  Máxima desviación: {max(self.deviations):.4f}")
    
    def plot_results(self):
        """Genera gráficos de los resultados"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Gráfico 1: Utilidades por iteración
        iterations = range(1, len(self.results['player_1']) + 1)
        ax1.plot(iterations, self.results['player_1'], 'o-', label='Jugador 1', alpha=0.7)
        ax1.plot(iterations, self.results['player_2'], 's-', label='Jugador 2', alpha=0.7)
        ax1.plot(iterations, self.results['player_3'], '^-', label='Jugador 3', alpha=0.7)
        ax1.axhline(y=1/3, color='r', linestyle='--', alpha=0.5, label='Utilidad ideal (1/3)')
        ax1.set_xlabel('Iteración')
        ax1.set_ylabel('Utilidad')
        ax1.set_title('Utilidades por Iteración')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Histograma de utilidades
        all_utilities = (self.results['player_1'] + 
                        self.results['player_2'] + 
                        self.results['player_3'])
        ax2.hist(all_utilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=1/3, color='r', linestyle='--', label='Utilidad ideal (1/3)')
        ax2.set_xlabel('Utilidad')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Utilidades')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Desviaciones del corte exacto
        ax3.plot(iterations, self.deviations, 'o-', color='orange', alpha=0.7)
        ax3.set_xlabel('Iteración')
        ax3.set_ylabel('Desviación')
        ax3.set_title('Desviación del Corte Exacto por Iteración')
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Box plot de utilidades por jugador
        utilities_data = [self.results['player_1'], 
                         self.results['player_2'], 
                         self.results['player_3']]
        ax4.boxplot(utilities_data, labels=['Jugador 1', 'Jugador 2', 'Jugador 3'])
        ax4.axhline(y=1/3, color='r', linestyle='--', alpha=0.5, label='Utilidad ideal (1/3)')
        ax4.set_ylabel('Utilidad')
        ax4.set_title('Distribución de Utilidades por Jugador')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Función principal del programa"""
    print("SIMULADOR DE REPARTO PROPORCIONAL DE TORTA")
    print("Algoritmo de Steinhaus para 3 jugadores")
    print("-" * 40)
    
    # Leer parámetros de entrada
    try:
        T = int(input("Ingrese el tamaño de la torta (T): "))
        N = int(input("Ingrese el número de iteraciones (N): "))
        
        if T <= 0 or N <= 0:
            print("Error: T y N deben ser números positivos")
            return
            
    except ValueError:
        print("Error: Por favor ingrese números enteros válidos")
        return
    
    # Crear y ejecutar simulación
    simulator = CakeDivisionSimulator()
    simulator.run_simulation(T, N, verbose=True)
    
    print("\n" + "=" * 60)
    print("CRITERIO Y MÉTODO UTILIZADO:")
    print("=" * 60)
    print("""
ALGORITMO DE STEINHAUS (3 JUGADORES):
1. Jugador 1 hace un corte donde obtiene utilidad = 1/3
2. Jugador 2 examina los trozos y hace un segundo corte
3. Se forman 3 trozos contiguos
4. Jugador 3 elige primero, luego Jugador 2, luego Jugador 1

GARANTÍAS:
- Cada jugador obtiene al menos 1/3 de su utilidad percibida
- El algoritmo es estrategia-proof para el primer cortador
- Simple de implementar y computacionalmente eficiente

LIMITACIONES:
- No garantiza ausencia total de envidia
- La equidad depende del orden de elección
- Puede haber desviaciones del corte exacto (1/3 para todos)
    """)
    
    print("\nCONSIDERACIONES PARA INFORMACIÓN INCOMPLETA:")
    print("=" * 50)
    print("""
Si los jugadores no conocieran las preferencias de otros:
1. Usar algoritmos robustos como "I Cut, You Choose" generalizado
2. Implementar mecanismos de revelación de preferencias
3. Aplicar teoría de juegos con información incompleta
4. Considerar algoritmos aproximados con garantías worst-case
5. Usar métodos adaptativos que aprendan de las decisiones pasadas
    """)

if __name__ == "__main__":
    main()