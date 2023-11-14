from itertools import combinations
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyArrowPatch
import copy

# à refaire
def read_file(name):
    with open(name + '.txt', 'r') as input_file:
        nodes_position = []
        nodes_load = []
        for i, line in enumerate(input_file):
            values = line.split(' ')
            if i == 0:
                n_tours = int(values[0])
                init_capacity = int(values[1])
            else:
                if int(values[3]) == 0:
                    depot = int(values[0]) - 1
                nodes_position.append([int(values[1]), int(values[2])])
                nodes_load.append(int(values[3]))
    n_nodes = len(nodes_position)
    nodes_position = np.array(nodes_position)
    #nodes_load = np.array(nodes_load)
    return n_nodes, n_tours, init_capacity, depot, nodes_position, nodes_load

# UTILITIES
def get_distance(src, dest):
    return distance_mtrx[src][dest]

def is_empty_route(route: list):
    if len(route) == 2 and 0 in route and len(distance_mtrx) - 1 in route:
        return True
    return False


def contains(list, filter):
    for x in list:
        if filter(x):
            return True
    return False

class TabuListClass:
    def __init__(self, op, move, valid_for):
        self.op = op
        self.move = move
        self.valid_for = valid_for

    def checked(self):
        if self.valid_for > 0:
            self.valid_for -= 1
            return self.valid_for
        else:
            return -1

    def find(self, move, aspired, op):
        if self.op == op and self.move == move and self.valid_for > 0 and not aspired:
            print("found tabu match op : {0} move : {1}".format(self.op, self.move))
            return True
        return False


def is_move_allowed(move, soln_prev, soln_curr, op):
    if len(tabu_list) < 1:
        return True
    cost_prev = get_distance_for_solution(soln_prev)
    cost_curr = get_distance_for_solution(soln_curr)
    if cost_prev-cost_curr > aspiration:
        return not contains(tabu_list, lambda x: x.find(move, True, op))
    else:
        return not contains(tabu_list, lambda x: x.find(move, False, op))

def iteration_update_tabu_list():
    for i in tabu_list:
        if i.checked() < 0:
            tabu_list.remove(i)

def get_distance_for_solution(soln: list):
    d=0
    prev=soln[0]
    for client in soln[1:]:
        d += get_distance(prev, client)
        prev=client
    return d

def get_distances(nodes_position):
    n = len(nodes_position)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):  # Profitez de la symétrie de la distance
            if i != j:
                dx = nodes_position[i][0] - nodes_position[j][0]
                dy = nodes_position[i][1] - nodes_position[j][1]
                distances[i][j] = np.sqrt(dx * dx + dy * dy)
                distances[j][i] = distances[i][j]  # Utilisez la symétrie pour éviter des calculs redondants
    return distances

###Fin UTILITIES

def generate_initial_solution(distance_matrix, demands, vehicle_capacity):
    def find_best_neighbor(current, unserved, visited, remaining_capacity):
        min_cost = float('inf')
        best_neighbor = None

        for neighbor in range(1, len(distance_matrix)):
            if neighbor in unserved and neighbor not in visited and demands[neighbor] <= remaining_capacity:
                cost = distance_matrix[current][neighbor]
                if cost < min_cost:
                    min_cost = cost
                    best_neighbor = neighbor

        return best_neighbor, min_cost

    route = [0]  # Starting from the depot
    total_cost = []
    unserved = set(range(1, len(distance_matrix)))
    visited = set()

    while unserved:
        current_location = route[-1]  # Start from the last location
        loop_cost = 0
        loop_route = []
        remaining_capacity = vehicle_capacity

        while unserved:
            best_neighbor, travel_cost = find_best_neighbor(current_location, unserved, visited, remaining_capacity)
            if best_neighbor is None:
                break

            loop_route.append(best_neighbor)
            loop_cost += travel_cost
            remaining_capacity -= demands[best_neighbor]
            unserved.remove(best_neighbor)
            visited.add(best_neighbor)
            current_location = best_neighbor

            if remaining_capacity == 0:
                break

        # Return to the depot at the end of the loop or if the vehicle capacity is reached
        return_cost = distance_matrix[current_location][0]
        loop_cost += return_cost
        total_cost.append(loop_cost)

        # Update the main route for the next loop
        route += loop_route + [0]

    return total_cost, route

def get_exchange_neighbour(soln):
    neighbours = []
    n = len(soln)
    for i in range(n):
        for j in range(i + 1, n):
            _tmp = copy.deepcopy(soln)
            _tmp[i], _tmp[j] = _tmp[j], _tmp[i]  # Échange des points i et j

            # Définir le mouvement pour la vérification dans is_move_allowed
            move = (soln[i], soln[j], i, j)

            # Vérifier si le mouvement est autorisé
            if is_move_allowed(move, soln, _tmp, 3):
                neighbours.append((_tmp, get_distance_for_solution(_tmp)))

    neighbours.sort(key=lambda x: x[1])  # Tri par coût, supposé être la distance totale
    return neighbours[0][0] if neighbours else None

def get_relocate_neighbour(soln):
    neighbours = []
    n = len(soln)
    for i in range(n):
        for j in range(n):
            if i != j:
                _tmp = copy.deepcopy(soln)
                element = _tmp.pop(i)  # Retirer l'élément à déplacer
                _tmp.insert(j, element)  # Insérer l'élément à la nouvelle position

                # Définir le mouvement pour la vérification dans is_move_allowed
                move = (element, i, j)

                # Vérifier si le mouvement est autorisé
                if is_move_allowed(move, soln, _tmp, 1):
                    neighbours.append((_tmp, get_distance_for_solution(_tmp)))

    neighbours.sort(key=lambda x: x[1])  # Tri par coût, supposé être la distance totale
    return neighbours[0][0] if neighbours else None

def get_neighbours(op, soln):
    if op == 1:
        return get_relocate_neighbour(soln)
    elif op == 3:
        return get_exchange_neighbour(soln)

def tabu_search(routes: list, iterations):
    best_solution_ever = routes
    best_cost_ever = get_distance_for_solution(routes)
    best_solution_ever_not_changed_itr_count = 0
    best_soln = routes
    best_cost = float('inf')
    costs = [None]*iterations
    global tabu_list

    for i in range(iterations):
        print("iteration n°",i)
        if best_solution_ever_not_changed_itr_count > 7:
            break

        _sol1 = get_neighbours(1, best_soln)
        _sol2 = get_neighbours(3, best_soln)

        if _sol1 == -1 or _sol2 == -1:
            break

        _sol1_cost = get_distance_for_solution(_sol1)
        _sol2_cost = get_distance_for_solution(_sol2)

        if _sol1_cost < _sol2_cost:
            best_soln = _sol1
            best_cost = _sol1_cost
        else:
            best_soln = _sol2
            best_cost = _sol2_cost

        if best_cost < best_cost_ever:
            best_cost_ever = best_cost
            best_solution_ever = best_soln
            best_solution_ever_not_changed_itr_count = 0
            print("nouveau meilleur", best_cost)
        else:
            best_solution_ever_not_changed_itr_count += 1

        costs[i]=best_cost
        iteration_update_tabu_list()

    return best_solution_ever, costs

def tracer_itineraire(destinations, itineraire, charges):
    fig, ax = plt.subplots()

    # Création d'une colormap basée sur les charges
    cmap = matplotlib.colormaps.get_cmap('viridis')
    norm = plt.Normalize(vmin=min(charges), vmax=max(charges))

    # Tracer les destinations avec des couleurs basées sur les charges
    for i, dest in enumerate(destinations):
        color = cmap(norm(charges[i]))  # Obtention de la couleur basée sur la charge
        if i == 0:
            ax.plot(dest[0], dest[1], marker='^', color=color, markersize=10, label='Entrepôt')
        else:
            ax.plot(dest[0], dest[1], marker='o', color=color, markersize=8)

    # Fonction pour ajouter des flèches intermédiaires le long d'une ligne entre deux points
    """def add_arrows_along_line(ax, start, end):
        # Calculer la distance entre les points
        line = np.array([end[0] - start[0], end[1] - start[1]])
        line_len = np.sqrt(line[0]**2 + line[1]**2)

        # Calculer le nombre de flèches intermédiaires, en fonction de la longueur de la ligne
        num_arrows = max(int(line_len // 5), 1)  # Par exemple, une flèche pour chaque 5 unités de distance

        # Calculer la direction de la ligne
        line_dir = line / line_len

        # Créer des points le long de la ligne pour placer les flèches
        for i in range(1, num_arrows + 1):
            point = start + line_dir * (line_len * i / (num_arrows + 1))
            arrow = FancyArrowPatch(point, point + line_dir * 0.1, color='black', arrowstyle='->', mutation_scale=5, alpha=0.6)  # Taille et opacité réduites
            ax.add_patch(arrow)"""

    # Tracer les flèches principales et intermédiaires indiquant l'itinéraire
    for i in range(len(itineraire) - 1):
        start_idx = itineraire[i]
        end_idx = itineraire[i + 1]

        start = destinations[start_idx]
        end = destinations[end_idx]

        # Tracer la flèche principale droite
        arrow_main = FancyArrowPatch(start, end, color='black', arrowstyle='-|>', mutation_scale=15)
        ax.add_patch(arrow_main)

        # Ajout des flèches intermédiaires le long de la ligne
        """add_arrows_along_line(ax, np.array(start), np.array(end))"""

    # Ajout d'une légende pour la colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Charge')

    ax.set_xlim(min(dest[0] for dest in destinations) - 1, max(dest[0] for dest in destinations) + 1)
    ax.set_ylim(min(dest[1] for dest in destinations) - 1, max(dest[1] for dest in destinations) + 1)

    ax.legend()
    plt.show()

def tracer_evolution_longueurs(longueurs):
    """
    Trace l'évolution des longueurs d'une liste au fil du temps.

    Args:
    longueurs (list): La liste dont l'évolution des longueurs doit être tracée.
    """

    # Créer une liste d'indices pour les points de données
    indices = range(1, len(longueurs) + 1)

    # Tracer l'évolution des longueurs
    plt.plot(indices, longueurs, linestyle='-')
    plt.xlabel('Temps')
    plt.ylabel('Longueur')
    plt.title('Évolution des longueurs au fil du temps')
    plt.grid(True)
    plt.show()

#besoin d'utiliser quelques variables globales pour l'instant
tabu_list = []
aspiration = 100
def test(iterations, test_name):
    nb_dest, nb_camions, capacite, entrepot, pos_dest, quantite_dest = read_file(test_name)
    print(quantite_dest)
    #Génération de la solution initiale (on prend la destination la plus proche à chaque fois)
    global distance_mtrx
    distance_mtrx = get_distances(pos_dest)
    cost, routes = generate_initial_solution(distance_mtrx, quantite_dest, capacite)

    itineraire_final, longueurs = tabu_search(routes, iterations)
    itineraire_final = itineraire_final + [0]

    print("Itinéraire final :", itineraire_final)

    # Vous pouvez également vouloir tracer l'itinéraire et l'évolution des longueurs
    tracer_itineraire(pos_dest, itineraire_final, quantite_dest)
    tracer_evolution_longueurs(longueurs)


# Lors de l'exécution du test, les données seront maintenant sauvegardées dans 'data_vrp.pkl'.

test(1000,"archipels_small")