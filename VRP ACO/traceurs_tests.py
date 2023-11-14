import matplotlib.pyplot as plt
import os

def read_vrp_file(file_path):
    """
    Lit un fichier, ignore la première ligne, et extrait les destinations, coordonnées, et charges à partir des lignes suivantes.
    """
    destinations = []

    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()[1:]  # Ignorer la première ligne
            for line in lines:
                # Assumant que les valeurs sont séparées par des espaces
                _, x, y, charge = line.strip().split()
                destinations.append((float(x), float(y), float(charge)))
    except Exception as e:
        print(f"Une erreur est survenue lors de la lecture du fichier: {e}")
        return None

    return destinations

def plot_vrp_points(destinations):
    """
    Trace les points de coordonnées sur un graphique en colorant en fonction de la charge.
    """
    if not destinations:
        print("Aucune coordonnée à tracer.")
        return

    # Les coordonnées de l'entrepôt sont supposées être le premier point
    warehouse = destinations[0]
    other_points = destinations[1:]

    # Préparation des données pour le tracé
    coords_and_charge = [(x, y, charge) for x, y, charge in other_points]
    x_coords, y_coords, charges = zip(*coords_and_charge)

    # Conversion des charges en une gamme de couleurs
    colors = [plt.cm.viridis(charge / max(charges)) for charge in charges]

    plt.figure("Visualisation VRP")
    plt.scatter(warehouse[0], warehouse[1], c='red', s=100, marker='s', label='Entrepôt')  # Marquer l'entrepôt différemment
    scatter = plt.scatter(x_coords, y_coords, c=colors, marker='o', label='Points de livraison')

    # Ajout d'une barre de couleur pour représenter la charge
    cbar = plt.colorbar(scatter)
    cbar.set_label('Charge')

    # Étiquettes et légende
    plt.title('Points de destination VRP avec Entrepôt')
    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Récupère le nom de tous les fichiers de test dans le répertoire courant
    current_directory = os.path.dirname(os.path.realpath(__file__))
    test_files = [file for file in os.listdir(current_directory) if file.endswith('.txt')]

    # Pour chaque fichier de test, lisez et tracez les points de destination VRP
    for test_file in test_files:
        print(f"Traitement du fichier {test_file}...")
        full_file_path = os.path.join(current_directory, test_file)
        vrp_destinations = read_vrp_file(full_file_path)

        if vrp_destinations:
            print(f"Tracé des points pour {test_file}...")
            plot_vrp_points(vrp_destinations)
        else:
            print(f"Aucun point valide à tracer dans {test_file}.")

if __name__ == "__main__":
    main()
