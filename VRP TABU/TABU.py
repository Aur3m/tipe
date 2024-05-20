import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyArrowPatch
#utilities

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
    nodes_load = np.array(nodes_load)
    return n_nodes, n_tours, init_capacity, depot, nodes_position, nodes_load

###
# Function to calculate distances between nodes
def calculate_distances(nodes_position):
    n = len(nodes_position)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dx = nodes_position[i][0] - nodes_position[j][0]
            dy = nodes_position[i][1] - nodes_position[j][1]
            distances[i][j] = distances[j][i] = np.sqrt(dx ** 2 + dy ** 2)
    return distances


# Greedy initial solution
def initial_solution(distances, demands, vehicle_capacity):
    n = len(distances)
    route = [0]  # Start at the depot
    remaining_capacity = vehicle_capacity

    visited = set([0])  # Start with the depot as visited
    while len(visited) < n:
        last_node = route[-1]
        nearest_neighbor = None
        min_distance = float('inf')

        for i in range(n):
            if i not in visited and demands[i] <= remaining_capacity:
                distance_to_i = distances[last_node][i]
                if distance_to_i < min_distance:
                    nearest_neighbor = i
                    min_distance = distance_to_i

        if nearest_neighbor is not None:
            route.append(nearest_neighbor)
            remaining_capacity -= demands[nearest_neighbor]
            visited.add(nearest_neighbor)
        else:
            route.append(0)  # Return to depot
            remaining_capacity = vehicle_capacity

    route.append(0)  # Return to depot at the end
    return route


# Calculate the total route length
def calculate_route_length(route, distances):
    length = 0
    for i in range(len(route) - 1):
        length += distances[route[i]][route[i + 1]]
    return length


# Check if the route is feasible with the given vehicle capacity
def is_feasible(route, demands, vehicle_capacity):
    current_capacity = vehicle_capacity
    for node in route:
        if node == 0:  # Reset capacity at the depot
            current_capacity = vehicle_capacity
        else:
            current_capacity -= demands[node]
            if current_capacity < 0:
                return False
    return True


# Génération NAIVE de voisins (inversements)
def generate_neighbors(route, distances, demands, vehicle_capacity):
    neighbors = []
    for i in range(1, len(route) - 1):
        for j in range(i + 1, len(route)):
            if i != j and not (route[i] == 0 or route[j] == 0):
                new_route = copy.deepcopy(route)
                new_route[i], new_route[j] = new_route[j], new_route[i]
                if is_feasible(new_route, demands, vehicle_capacity):
                    neighbors.append(new_route)
    return neighbors
#INsertion
def generate_insertion_neighbors(route, distances, demands, vehicle_capacity):
    neighbors = []
    for i in range(1, len(route) - 1):
        for j in range(1, len(route) - 1):
            if i != j and route[i] != 0:
                new_route = copy.deepcopy(route)
                node_to_move = new_route.pop(i)
                new_route.insert(j, node_to_move)
                if is_feasible(new_route, demands, vehicle_capacity):
                    neighbors.append(new_route)
    return neighbors

#inversion
def generate_inversion_neighbors(route, distances, demands, vehicle_capacity):
    neighbors = []
    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            new_route = copy.deepcopy(route)
            new_route[i:j] = new_route[i:j][::-1]  # Invert the sequence
            if is_feasible(new_route, demands, vehicle_capacity):
                neighbors.append(new_route)
    return neighbors
#echange de tournées
def generate_cross_tour_swap_neighbors(route, distances, demands, vehicle_capacity):
    neighbors = []
    n = len(route)
    tour_indices = [i for i, node in enumerate(route) if node == 0]  # Indices des dépôts

    # Parcourir chaque paire de tournées
    for i in range(len(tour_indices) - 1):
        for j in range(i + 1, len(tour_indices) - 1):
            tour_start1 = tour_indices[i]
            tour_end1 = tour_indices[i + 1]
            tour_start2 = tour_indices[j]
            tour_end2 = tour_indices[j + 1] if j + 1 < len(tour_indices) else n

            # Échanger des éléments entre les tournées
            for idx1 in range(tour_start1 + 1, tour_end1):  # Éviter les dépôts
                for idx2 in range(tour_start2 + 1, tour_end2):  # Éviter les dépôts
                    if idx1 != idx2:
                        new_route = copy.deepcopy(route)
                        new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
                        if is_feasible(new_route, demands, vehicle_capacity):
                            neighbors.append(new_route)

    return neighbors

def generate_3opt_neighbor(route, distances, demands, vehicle_capacity):
    # Identifier les indices des dépôts (0) pour séparer les tournées
    depot_indices = [i for i, node in enumerate(route) if node == 0]
    best_route = copy.deepcopy(route)
    best_cost = calculate_route_length(best_route, distances)

    # Appliquer 3-opt sur chaque tournée individuellement
    for i in range(len(depot_indices) - 1):
        tour_start = depot_indices[i]
        tour_end = depot_indices[i + 1]

        tour = route[tour_start:tour_end + 1]
        improved = True
        while improved:
            improved = False
            for i in range(1, len(tour) - 5):
                for j in range(i+2, len(tour) - 3):
                    for k in range(j+2, len(tour) - 1):
                        # Générer des tournées avec différents réarrangements de segments
                        new_tours = [
                            tour[:i] + tour[i:j+1][::-1] + tour[j+1:k+1][::-1] + tour[k+1:],
                            tour[:i] + tour[j+1:k+1] + tour[i:j+1] + tour[k+1:],
                            tour[:i] + tour[j+1:k+1] + tour[i:j+1][::-1] + tour[k+1:],
                            tour[:i] + tour[i:j+1][::-1] + tour[j+1:k+1] + tour[k+1:],
                            tour[:i] + tour[k:k+1] + tour[j+1:k] + tour[i:j+1] + tour[k+1:]
                        ]

                        for new_tour in new_tours:
                            if is_feasible(new_tour, demands, vehicle_capacity):
                                new_route = route[:tour_start] + new_tour + route[tour_end + 1:]
                                new_cost = calculate_route_length(new_route, distances)
                                if new_cost < best_cost:
                                    best_route = new_route
                                    best_cost = new_cost
                                    improved = True
                                    break
                    if improved:
                        break
                if improved:
                    break

    return best_route if is_feasible(best_route, demands, vehicle_capacity) else route


# Tabu Search Algorithm
def tabu_search(initial_route, distances, demands, vehicle_capacity, iterations, tabu_tenure):
    longueurs = []
    best_solution = initial_route
    best_cost = calculate_route_length(best_solution, distances)
    tabu_list = []

    for i in range(iterations):
        print("iteration n° ",i)

        neighbors = generate_neighbors(best_solution, distances, demands, vehicle_capacity)
        neighbors.extend(generate_insertion_neighbors(best_solution, distances, demands, vehicle_capacity))
        neighbors.extend(generate_inversion_neighbors(best_solution, distances, demands, vehicle_capacity))
        neighbors.extend(generate_cross_tour_swap_neighbors(best_solution, distances, demands, vehicle_capacity))

        neighbors.append(generate_3opt_neighbor(best_solution, distances, demands, vehicle_capacity))
        #on met 2-opt aussi?
        neighbors.sort(key=lambda x: calculate_route_length(x, distances))

        for neighbor in neighbors:
            if neighbor not in tabu_list:
                neighbor_cost = calculate_route_length(neighbor, distances)
                longueurs.append(neighbor_cost)
                if neighbor_cost < best_cost:
                    print("nouveau meilleur ", neighbor_cost, neighbor)
                    best_solution = neighbor
                    best_cost = neighbor_cost
                    break

        tabu_list.append(best_solution)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)

    return best_solution, best_cost, longueurs

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
def test(iterations, test_name):
    nb_dest, nb_camions, capacite, entrepot, pos_dest, quantite_dest = read_file(test_name)
    distances = calculate_distances(pos_dest)
    initial_route = initial_solution(distances, quantite_dest, capacite)
    #initial_route = [0, 59, 54, 58, 52, 56, 48, 50, 51, 0, 53, 60, 49, 55, 61, 0, 57, 62, 47, 40, 0, 26, 21, 30, 22, 0, 20, 19, 24, 28, 29, 27, 0, 18, 31, 25, 23, 0, 71, 66, 78, 63, 76, 70, 77, 67, 0, 69, 65, 35, 32, 46, 36, 0, 33, 38, 39, 43, 0, 74, 73, 75, 72, 68, 64, 0, 41, 1, 14, 17, 0, 4, 13, 7, 8, 0, 42, 45, 44, 37, 34, 0, 12, 3, 16, 2, 9, 15, 11, 0, 10, 6, 5]
    #initial_route = [0, 97, 99, 93, 98, 94, 95, 104, 100, 0, 96, 103, 92, 102, 101, 127, 121, 124, 125, 0, 173, 170, 182, 175, 176, 174, 181, 180, 178, 179, 0, 59, 53, 65, 55, 116, 56, 58, 61, 64, 0, 164, 129, 119, 162, 161, 168, 163, 167, 169, 165, 0, 172, 177, 171, 3, 2, 12, 11, 1, 10, 0, 123, 126, 158, 157, 159, 122, 120, 118, 128, 166, 130, 0, 73, 71, 66, 74, 75, 78, 69, 115, 113, 106, 108, 62, 68, 117, 57, 67, 0, 63, 112, 111, 110, 60, 105, 107, 54, 0, 114, 72, 76, 77, 70, 109, 141, 143, 0, 81, 84, 80, 85, 79, 86, 38, 91, 88, 89, 87, 82, 0, 4, 6, 9, 7, 8, 5, 13, 83, 27, 160, 0, 133, 136, 132, 134, 137, 131, 140, 138, 139, 135, 142, 0, 51, 40, 46, 49, 50, 45, 42, 47, 52, 44, 48, 0, 43, 41, 17, 20, 22, 192, 184, 152, 155, 0, 154, 147, 151, 146, 145, 149, 144, 148, 150, 153, 156, 194, 186, 0, 25, 19, 26, 16, 18, 23, 24, 21, 14, 15, 0, 34, 39, 35, 30, 28, 29, 37, 33, 32, 31, 90, 36, 0, 193, 191, 189, 188, 190, 195, 183, 187, 185, 0]
    initial_route = [0, 59, 54, 58, 52, 56, 48, 50, 51, 0, 49, 60, 53, 55, 61, 0, 20, 19, 24, 21, 28, 0, 57, 62, 33, 38, 47, 0, 26, 18, 29, 27, 0, 67, 65, 70, 78, 66, 71, 77, 0, 6, 10, 5, 8, 7, 0, 1, 14, 17, 13, 0, 34, 45, 35, 37, 44, 0, 69, 74, 75, 73, 72, 68, 0, 64, 63, 76, 42, 40, 0, 36, 46, 32, 41, 0, 3, 12, 16, 2, 9, 15, 11, 0, 31, 23, 25, 22, 30, 0, 39, 43, 4]
    best_route, best_cost, longueurs = tabu_search(initial_route, distances, quantite_dest, capacite, iterations, math.ceil(nb_dest/5))

    # Ajoutez le point de départ à la fin de l'itinéraire pour compléter le cycle
    itineraire_final = best_route + [0]

    print("Itinéraire final :", itineraire_final)

    # Lire les données du fichier qui a été utilisé dans la fonction VRP
    nb_dest, nb_camions, capacite, entrepot, pos_dest, quantite_dest = read_file(test_name)

    # Vous pouvez également vouloir tracer l'itinéraire et l'évolution des longueurs
    tracer_itineraire(pos_dest, itineraire_final, quantite_dest)
    tracer_evolution_longueurs(longueurs)

# Example usage

test(500,"archipels_small")


