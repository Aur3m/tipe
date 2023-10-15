import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

"""Problème de ce programme :
Il va chercher à utiliser les camions jusqu'au maximum de leur capacité,
Il faut donc démontrer par l'absurde au préalable que la solution optimale
est une solution qui utilise les camions jusqu'à la fin de leur charge
mais c'est bien le cas ici car on est sur un VRP qui essaie d'économiser
des coûts sans contrainte de temps

D'ailleurs ce qui est marrant c'est que si un camion a la capacité
de tout livrer on devrait retomber logiquement sur une solution du TSP

Dans une seconde version du VRP qui cherchera à exécuter le tout
dans une certaine fenêtre de temps, cela posera un problème


Gérer les cas où on n'aura pas de quoi tout livrer? privilegier la plus grande livraison ?
"""


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
    nodes_load = np.array(nodes_load)
    return n_nodes, n_tours, init_capacity, depot, nodes_position, nodes_load


# à refaire
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

#à refaire
def get_best_ant_distance(best_ants_location, distances):
    dist = 0.0
    for k in range(1, len(best_ants_location)):
        i = best_ants_location[k-1]
        j = best_ants_location[k]
        dist += distances[i][j]
    return dist

# à refaire
def calculate_visibility(n_nodes, distances):
    # on calcule ce qui correspondrait à la "visibilité" des fourmis
    visibility = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                visibility[i][j] = 1.0 / distances[i][j]
    return visibility


# à refaire
def init_mat_phero(n_nodes):
    pheromone = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            pheromone[i][j] = 0.1
    return pheromone


# à refaire
def init_itinerary(n_ants, depot, init_capacity):
    ants_location = {}
    ants_capacity = np.zeros(n_ants)
    for i in range(n_ants):
        ants_location[i] = [depot]
        ants_capacity[i] = init_capacity
    return ants_location, ants_capacity


# à refaire
def get_ant_distance(ants_location, ant, distances):
    dist = 0.0
    ant_location = ants_location[ant]
    for k in range(1, len(ant_location)):
        i = ant_location[k - 1]
        j = ant_location[k]
        dist += distances[i][j]
    return dist

def evaporation(pheromone, best_ants_location, n_nodes, distances, rho):
    for i in range(n_nodes):
        for j in range(n_nodes):
            pheromone[i][j] *= rho
    Lk = get_best_ant_distance(best_ants_location, distances)
    for i in range(n_nodes):
        for j in range(n_nodes):
            for k in range(1, len(best_ants_location)):
                if best_ants_location[k - 1] == i and best_ants_location[k] == j:
                    pheromone[i][j] += 1.0 / Lk
    return pheromone


# Fonction principale
def VRP(nb_iterations, alpha, beta, gamma,test):
    # alpha est le coefficient d'importance des distances dans la pondération
    # beta celui des phéromones
    # gamma le coefficient d'évaporation des phéromones
    nb_dest, nb_camions, capacite, entrepot, pos_dest, quantite_dest = read_file(test)
    distances = get_distances(pos_dest)
    visibility = calculate_visibility(nb_dest, distances)
    mat_phero = init_mat_phero(nb_dest)
    nb_fourmis = nb_dest # Il faut que ça soit proportionnel au nombre de destinations

    # en général on a "f_" qui est là pour "fourmis_"
    f_distances = np.zeros((nb_iterations, nb_fourmis))
    best_sol_dist = np.zeros(nb_iterations)

    min_dist = float('inf')
    best_itinerary = None

    for iteration in range(nb_iterations):
        print("iteration n°",iteration)
        f_chemin, f_capacite = init_itinerary(nb_fourmis, entrepot, capacite)
        for f in range(nb_fourmis):
            visited = [False] * nb_dest
            visited[entrepot] = True
            buffer = 0
            while buffer < nb_camions:
                start = f_chemin[f][-1]
                unvisited_dest = np.where(np.logical_not(visited))[0]  # regarder ce que fait cette fonction en détails pour améliorer la complexité
                if unvisited_dest.shape[0] > 0:  # si la matrice des dest non visités a plus de 0 ligne

                    # début choix du prochain chemin
                    proba_total = 0
                    for dest in unvisited_dest:
                        proba_total += ((mat_phero[start][dest]) ** alpha) * (visibility[start][dest] ** beta)
                    proba = np.zeros(nb_dest)
                    for dest_bis in range(nb_dest):
                        if dest_bis in unvisited_dest:
                            proba[dest_bis] = ((mat_phero[start][dest_bis]) ** alpha) * (
                                    visibility[start][dest_bis] ** beta) / proba_total

                    next_dest = np.random.choice(nb_dest, p=proba)
                    # fin choix du prochain chemin

                    if f_capacite[f] >= quantite_dest[next_dest] and next_dest != entrepot:
                        f_capacite[f] -= quantite_dest[next_dest]
                        f_chemin[f].append(next_dest)
                        visited[next_dest] = True
                    else:
                        f_chemin[f].append(entrepot)
                        f_capacite[f] = capacite
                        buffer += 1  # c'est comme si virtuellement on venait d'utiliser un camion
                else:  # Si on a déjà tout visité sans utiliser tout les camions
                    break
            f_distances[iteration][f] = get_ant_distance(f_chemin, f, distances)
            # Il y a un ACO alternatif où on modifie les phéromones aussi ici
            # pour que les fourmis au sein de la même itération puissent s'influencer et s'améliorer
            # ici on ne l'a pas fait

        # On regarde la meilleure fourmi sur toutes les fourmis pour cette itération
        best_f = np.argmin(f_distances[iteration])
        best_sol_dist[iteration] = f_distances[iteration][best_f]
        if best_sol_dist[iteration] < min_dist:
            min_dist = best_sol_dist[iteration]
            print("nouvelle distance ", min_dist)
            best_itinerary = f_chemin[best_f]

        # Evaporation des phéromones (pour éviter de converger vers des optimums locaux)
        mat_phero = evaporation(mat_phero, best_itinerary, nb_dest, distances, gamma)
    return best_sol_dist, best_itinerary #interessant de tout garder si on veut analyser la convergence

def tracer_itineraire(destinations, itineraire):
    # Créer une figure et un axe
    fig, ax = plt.subplots()

    # Tracer les destinations
    for i, dest in enumerate(destinations):
        if i == 0:
            # Tracer l'entrepôt comme un triangle orange
            ax.plot(dest[0], dest[1], marker='^', color='orange', markersize=10, label='Entrepôt')
        else:
            # Tracer les autres destinations comme des ronds bleus
            ax.plot(dest[0], dest[1], marker='o', color='blue', markersize=8)

    # Tracer les flèches orientées selon l'itinéraire
    for i in range(len(itineraire) - 1):
        start_idx = itineraire[i]
        end_idx = itineraire[i + 1]
        x1, y1 = destinations[start_idx]
        x2, y2 = destinations[end_idx]
        dx = x2 - x1
        dy = y2 - y1
        arrow = FancyArrowPatch((x1, y1), (x2, y2), color='black', arrowstyle='-|>', mutation_scale=15)
        ax.add_patch(arrow)

    # Définir les limites de l'axe
    ax.set_xlim(min(dest[0] for dest in destinations) - 1, max(dest[0] for dest in destinations) + 1)
    ax.set_ylim(min(dest[1] for dest in destinations) - 1, max(dest[1] for dest in destinations) + 1)

    # Ajouter une légende
    ax.legend()

    # Afficher le graphique
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
def test(iterations,test_name):
    longueurs, itineraire_final = VRP(iterations, 1.5, 4, 0.85,test_name)
    itineraire_final = itineraire_final+[0]
    print(itineraire_final)
    nb_dest, nb_camions, capacite, entrepot, pos_dest, quantite_dest = read_file(test_name)
    tracer_itineraire(pos_dest,itineraire_final)
    tracer_evolution_longueurs(longueurs)


test(100,"archipels_big")
