import random
import math
import matplotlib.pyplot as plt

# Nombre d'archipels et de villes par archipel
num_archipels = 5
villes_par_archipel = 15

# Coordonnées des archipels
archipels = [(random.randint(0, 200), random.randint(0, 200)) for _ in range(num_archipels)]

# Génération des coordonnées des villes
villes = []

for i, (archipel_x, archipel_y) in enumerate(archipels):
    for _ in range(villes_par_archipel):
        angle = 2 * math.pi * random.random()
        radius = random.random() * 30
        x = int(archipel_x + radius * math.cos(angle))
        y = int(archipel_y + radius * math.sin(angle))
        villes.append((x, y))

# Calcul du barycentre
total_x = 0
total_y = 0

# Somme de toutes les coordonnées x et y
for x, y in villes:
    total_x += x
    total_y += y

# Calcul de la moyenne des coordonnées x et y
barycentre_x = total_x / len(villes)
barycentre_y = total_y / len(villes)

# Création de la batterie de test avec le barycentre comme première ville
test_battery = [(1, barycentre_x, barycentre_y, 0)]

# Ajout des autres villes avec leur valeur aléatoire
for i, (x, y) in enumerate(villes):
    random_num = random.randint(1, 35)
    test_battery.append((i + 2, x, y, random_num))

    # Affichage des résultats
for ville in test_battery:
    print(ville[0], ville[1], ville[2], ville[3])

print("\nBarycentre:", round(barycentre_x, 2), round(barycentre_y, 2))


# Fonction de visualisation des villes et du barycentre
def visualize_villes(test_battery, barycentre_x, barycentre_y):
    plt.figure(figsize=(10, 10))

    # Plot des villes et du barycentre
    for i, x, y, val in test_battery:
        if val == 0:
            plt.scatter(x, y, c='red', marker='x', s=100)
        else:
            plt.scatter(x, y, c='blue')

    # Ajout d'étiquettes et d'une légende
    plt.title('Visualisation des Villes et du Barycentre')
    plt.xlabel('Coordonnée X')
    plt.ylabel('Coordonnée Y')
    plt.legend(['Barycentre', 'Villes'], loc='upper left')

    # Affichage du plot
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# Appel de la fonction de visualisation
visualize_villes(test_battery, barycentre_x, barycentre_y)
