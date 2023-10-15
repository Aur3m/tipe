import math
import random
import matplotlib.pyplot as plt

# Nombre de villes
num_villes = 100

# Type de configuration
# ['cercle', 'spirale', 'gribouillage', 'etoile', 'flots', 'grille', 'clusters', 'ligne', 'border','croix','serpentins','cercles_concentriques']
config_type = 'cercles_concentriques'  # Choisir le type de configuration ici

# Coordonnées des villes
villes = []

# Générer les coordonnées en fonction du type de configuration
if config_type == 'clusters':
    num_clusters = 5
    for _ in range(num_clusters):
        cluster_center = (random.uniform(50, 250), random.uniform(50, 250))
        for _ in range(num_villes // num_clusters):
            angle = 2 * math.pi * random.random()
            radius = random.uniform(0, 30)
            x = cluster_center[0] + radius * math.cos(angle)
            y = cluster_center[1] + radius * math.sin(angle)
            villes.append((x, y))

elif config_type == 'cercles_concentriques':
    nombre_cercles = 4
    villes_par_cercle = num_villes // nombre_cercles

    for i in range(nombre_cercles):
        rayon = 20 * (i + 1)
        for j in range(villes_par_cercle):
            angle = 2 * math.pi * j / villes_par_cercle
            x = 150 + rayon * math.cos(angle)
            y = 150 + rayon * math.sin(angle)
            villes.append((x, y))

elif config_type == 'ligne':
    start_point = (50, 150)
    end_point = (250, 150)
    for i in range(num_villes):
        ratio = i / (num_villes-1)
        x = (1-ratio) * start_point[0] + ratio * end_point[0]
        y = (1-ratio) * start_point[1] + ratio * end_point[1]
        villes.append((x, y))

elif config_type == 'serpentins':
    for i in range(num_villes):
        x = i * 10
        y = math.sin(i) * 50 + 150  # Sinusoidal path
        villes.append((x, y))

elif config_type == 'border':
    for i in range(num_villes):
        position = random.uniform(0, 1)
        side = random.choice(['top', 'bottom', 'left', 'right'])
        if side == 'top':
            villes.append((position * 300, 290))
        elif side == 'bottom':
            villes.append((position * 300, 10))
        elif side == 'left':
            villes.append((10, position * 300))
        elif side == 'right':
            villes.append((290, position * 300))

elif config_type == 'croix':
    longueur_branche = num_villes // 2
    for i in range(longueur_branche):
        villes.append((150, i*6))  # Branche verticale
        villes.append((i*6, 150))  # Branche horizontale

# (les autres configurations peuvent être insérées ici)
elif config_type == 'cercle':
    centre_x, centre_y = 150, 150
    rayon = 100
    for i in range(num_villes):
        angle = 2 * math.pi * i / num_villes
        x = centre_x + rayon * math.cos(angle)
        y = centre_y + rayon * math.sin(angle)
        villes.append((x, y))

elif config_type == 'spirale':
    centre_x, centre_y = 150, 150
    for i in range(num_villes):
        angle = 0.3 * i  # coefficient pour ajuster l'écart entre les points
        x = centre_x + i * math.cos(angle)
        y = centre_y + i * math.sin(angle)
        villes.append((x, y))

elif config_type == 'gribouillage':
    centre_x, centre_y = 150, 150
    for _ in range(num_villes):
        angle = 2 * math.pi * random.random()
        dist = 150 * random.random()
        x = centre_x + dist * math.cos(angle)
        y = centre_y + dist * math.sin(angle)
        villes.append((x, y))

elif config_type == 'etoile':
    centre_x, centre_y = 150, 150
    for i in range(num_villes):
        angle = 2 * math.pi * i / num_villes
        dist = 150 * (i % 2)  # alternance entre un point proche et un point éloigné
        x = centre_x + dist * math.cos(angle)
        y = centre_y + dist * math.sin(angle)
        villes.append((x, y))

elif config_type == 'flots':
    for _ in range(num_villes):
        x = random.uniform(50, 250)
        y = random.uniform(50, 250) + 50 * math.sin(x/25)
        villes.append((x, y))

elif config_type == 'grille':
    grid_size = int(math.sqrt(num_villes))
    for i in range(grid_size):
        for j in range(grid_size):
            x = 30 * i + 50
            y = 30 * j + 50
            villes.append((x, y))
            if len(villes) == num_villes:
                break
        if len(villes) == num_villes:
            break

# Calcul du barycentre
total_x = sum(x for x, y in villes)
total_y = sum(y for x, y in villes)
barycentre = (total_x/num_villes, total_y/num_villes)

# Ajouter le barycentre comme première ville avec demande 0
villes.insert(0, barycentre)

# Affichage des données dans la console
for i, (x, y) in enumerate(villes):
    demande = 0 if i == 0 else random.randint(1, 35)  # demande est 0 pour le barycentre, aléatoire pour les autres
    print(f"{i + 1} {x:.2f} {y:.2f} {demande}")

# Affichage du graphique
plt.figure(figsize=(10, 10))
for i, (x, y) in enumerate(villes):
    plt.scatter(x, y, c='red' if i == 0 else 'blue')
plt.title(f'Configuration: {config_type.capitalize()}')
plt.xlabel('Coordonnée X')
plt.ylabel('Coordonnée Y')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
