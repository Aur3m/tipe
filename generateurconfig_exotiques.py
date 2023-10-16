import math
import random
import matplotlib.pyplot as plt

# Nombre de villes
num_villes = 250

# Type de configuration
# ['cercle', 'spirale', 'gribouillage', 'etoile', 'flots', 'grille', 'clusters', 'ligne', 'border','croix','serpentins','cercles_concentriques']
config_type = 'border'  # Choisir le type de configuration ici

# Coordonnées des villes
villes = []

# Fonction pour vérifier si la ville est trop proche d'une autre ville
def is_too_close(new_ville, villes, min_distance=5):
    for ville in villes:
        if math.dist(new_ville, ville) < min_distance:
            return True
    return False

# Générer les coordonnées en fonction du type de configuration
if config_type == 'clusters':
    num_clusters = 5
    for _ in range(num_clusters):
        cluster_center = (round(random.uniform(50, 250)), round(random.uniform(50, 250)))
        for _ in range(num_villes // num_clusters):
            angle = 2 * math.pi * random.random()
            radius = random.uniform(0, 30)
            x = round(cluster_center[0] + radius * math.cos(angle))
            y = round(cluster_center[1] + radius * math.sin(angle))
            if not is_too_close((x, y), villes):
                villes.append((x, y))

elif config_type == 'cercles_concentriques':
    nombre_cercles = 4
    villes_par_cercle = num_villes // nombre_cercles

    for i in range(nombre_cercles):
        rayon = 20 * (i + 1)
        for j in range(villes_par_cercle):
            angle = 2 * math.pi * j / villes_par_cercle
            x = round(150 + rayon * math.cos(angle))
            y = round(150 + rayon * math.sin(angle))
            # Verifier si les nouvelles coordonnées sont trop proches des villes existantes
            if not is_too_close((x, y), villes):
                villes.append((x, y))
            else:  # Si trop proche, essayer un autre angle
                new_angle = angle + (math.pi / villes_par_cercle)
                x = round(150 + rayon * math.cos(new_angle))
                y = round(150 + rayon * math.sin(new_angle))
                if not is_too_close((x, y), villes):  # Si le nouvel angle convient, ajouter la ville
                    villes.append((x, y))

elif config_type == 'ligne':
    start_point = (50, 150)
    end_point = (250, 150)
    for i in range(num_villes):
        ratio = i / (num_villes - 1)
        x = round((1 - ratio) * start_point[0] + ratio * end_point[0])
        y = round((1 - ratio) * start_point[1] + ratio * end_point[1])

        # Vérifier si les nouvelles coordonnées sont trop proches des villes existantes
        if not is_too_close((x, y), villes):
            villes.append((x, y))
        else:
            # Si trop proche, essayer d'ajuster légèrement la position
            adjust_ratio = 0.02  # pourcentage d'ajustement; à personnaliser selon le besoin
            x_adjusted = round((1 - ratio + adjust_ratio) * start_point[0] + (ratio - adjust_ratio) * end_point[0])
            y_adjusted = round((1 - ratio + adjust_ratio) * start_point[1] + (ratio - adjust_ratio) * end_point[1])

            if not is_too_close((x_adjusted, y_adjusted), villes):
                villes.append((x_adjusted, y_adjusted))

elif config_type == 'serpentins':
    for i in range(num_villes):
        x = i * 10
        y = math.sin(i) * 50 + 150  # Sinusoidal path
        x, y = round(x), round(y)

        # Vérifier si les nouvelles coordonnées ne sont pas trop proches des villes existantes
        if not is_too_close((x, y), villes):
            villes.append((x, y))
        else:
            # Si trop proche, essayer d'ajuster légèrement la position
            # Ajuster x et/ou y si nécessaire pour éviter les superpositions
            for adjust in [1, -1]:  # Ajustement de 1 unité dans les deux directions
                x_adjusted, y_adjusted = x + adjust, y
                if not is_too_close((x_adjusted, y_adjusted), villes):
                    villes.append((x_adjusted, y_adjusted))
                    break  # Sortir de la boucle une fois qu'une position valable est trouvée

elif config_type == 'border':
    for _ in range(num_villes):
        position = random.uniform(0, 1)
        side = random.choice(['top', 'bottom', 'left', 'right'])

        if side == 'top':
            x, y = round(position * 300), 290
        elif side == 'bottom':
            x, y = round(position * 300), 10
        elif side == 'left':
            x, y = 10, round(position * 300)
        elif side == 'right':
            x, y = 290, round(position * 300)

        # Vérifier si les nouvelles coordonnées ne sont pas trop proches des villes existantes
        if not is_too_close((x, y), villes):
            villes.append((x, y))
        else:
            # Si trop proche, essayer d'ajuster légèrement la position
            # Ajuster x et/ou y si nécessaire pour éviter les superpositions
            for adjust in [1, -1]:  # Ajustement de 1 unité dans les deux directions
                x_adjusted, y_adjusted = x + adjust, y + adjust
                if not is_too_close((x_adjusted, y_adjusted), villes):
                    villes.append((x_adjusted, y_adjusted))
                    break  # Sortir de la boucle une fois qu'une position valable est trouvée


elif config_type == 'croix':
    longueur_branche = num_villes // 2
    for i in range(longueur_branche):
        x1, y1 = 150, i * 6  # Branche verticale
        x2, y2 = i * 6, 150  # Branche horizontale

        # Vérifier si les nouvelles coordonnées ne sont pas trop proches des villes existantes
        if not is_too_close((x1, y1), villes):
            villes.append((x1, y1))
        else:
            # Ajuster y1 si nécessaire pour éviter les superpositions
            for adjust in [1, -1]:
                y_adjusted = y1 + adjust
                if not is_too_close((x1, y_adjusted), villes):
                    villes.append((x1, y_adjusted))
                    break

        if not is_too_close((x2, y2), villes):
            villes.append((x2, y2))
        else:
            # Ajuster x2 si nécessaire pour éviter les superpositions
            for adjust in [1, -1]:
                x_adjusted = x2 + adjust
                if not is_too_close((x_adjusted, y2), villes):
                    villes.append((x_adjusted, y2))
                    break


# (les autres configurations peuvent être insérées ici)
elif config_type == 'cercle':
    centre_x, centre_y = 150, 150
    rayon = 100
    for i in range(num_villes):
        angle = 2 * math.pi * i / num_villes
        x = int(round(centre_x + rayon * math.cos(angle)))
        y = int(round(centre_y + rayon * math.sin(angle)))

        # Vérifier si les nouvelles coordonnées ne sont pas trop proches des villes existantes
        if not is_too_close((x, y), villes):
            villes.append((x, y))
        else:
            # Ajuster x et y si nécessaire pour éviter les superpositions
            found = False
            for adjust_x in [1, -1]:
                for adjust_y in [1, -1]:
                    x_adjusted = x + adjust_x
                    y_adjusted = y + adjust_y
                    if not is_too_close((x_adjusted, y_adjusted), villes):
                        villes.append((x_adjusted, y_adjusted))
                        found = True
                        break
                if found:
                    break


elif config_type == 'spirale':
    centre_x, centre_y = 150, 150
    for i in range(num_villes):
        angle = 0.3 * i  # coefficient pour ajuster l'écart entre les points
        x = int(round(centre_x + i * math.cos(angle)))
        y = int(round(centre_y + i * math.sin(angle)))

        # Vérifier si les nouvelles coordonnées ne sont pas trop proches des villes existantes
        if not is_too_close((x, y), villes):
            villes.append((x, y))
        else:
            # Ajuster x et y si nécessaire pour éviter les superpositions
            found = False
            for adjust_x in [1, -1]:
                for adjust_y in [1, -1]:
                    x_adjusted = x + adjust_x
                    y_adjusted = y + adjust_y
                    if not is_too_close((x_adjusted, y_adjusted), villes):
                        villes.append((x_adjusted, y_adjusted))
                        found = True
                        break
                if found:
                    break


elif config_type == 'gribouillage':
    centre_x, centre_y = 150, 150
    for _ in range(num_villes):
        angle = 2 * math.pi * random.random()
        dist = 150 * random.random()
        x = int(round(centre_x + dist * math.cos(angle)))
        y = int(round(centre_y + dist * math.sin(angle)))

        # Vérifier si les nouvelles coordonnées ne sont pas trop proches des villes existantes
        if not is_too_close((x, y), villes):
            villes.append((x, y))
        else:
            # Ajuster x et y si nécessaire pour éviter les superpositions
            found = False
            for adjust_x in [1, -1]:
                for adjust_y in [1, -1]:
                    x_adjusted = x + adjust_x
                    y_adjusted = y + adjust_y
                    if not is_too_close((x_adjusted, y_adjusted), villes):
                        villes.append((x_adjusted, y_adjusted))
                        found = True
                        break
                if found:
                    break


elif config_type == 'etoile':
    centre_x, centre_y = 150, 150
    for i in range(num_villes):
        angle = 2 * math.pi * i / num_villes
        dist = 150 * (i % 2)  # alternance entre un point proche et un point éloigné
        x = int(round(centre_x + dist * math.cos(angle)))
        y = int(round(centre_y + dist * math.sin(angle)))

        # Vérifier si les nouvelles coordonnées ne sont pas trop proches des villes existantes
        if not is_too_close((x, y), villes):
            villes.append((x, y))
        else:
            # Ajuster x et y si nécessaire pour éviter les superpositions
            found = False
            for adjust_x in [1, -1]:
                for adjust_y in [1, -1]:
                    x_adjusted = x + adjust_x
                    y_adjusted = y + adjust_y
                    if not is_too_close((x_adjusted, y_adjusted), villes):
                        villes.append((x_adjusted, y_adjusted))
                        found = True
                        break
                if found:
                    break

elif config_type == 'flots':
    for _ in range(num_villes):
        x = random.uniform(50, 250)
        y = random.uniform(50, 250) + 50 * math.sin(x / 25)
        x, y = int(round(x)), int(round(y))

        # Vérifier si les nouvelles coordonnées ne sont pas trop proches des villes existantes
        if not is_too_close((x, y), villes):
            villes.append((x, y))
        else:
            # Ajuster x et y si nécessaire pour éviter les superpositions
            found = False
            for adjust_x in [1, -1]:
                for adjust_y in [1, -1]:
                    x_adjusted = x + adjust_x
                    y_adjusted = y + adjust_y
                    if not is_too_close((x_adjusted, y_adjusted), villes):
                        villes.append((x_adjusted, y_adjusted))
                        found = True
                        break
                if found:
                    break


elif config_type == 'grille':
    grid_size = int(math.sqrt(num_villes))
    for i in range(grid_size):
        for j in range(grid_size):
            x = 30 * i + 50
            y = 30 * j + 50
            x, y = int(round(x)), int(round(y))

            # Vérifier si les nouvelles coordonnées ne sont pas trop proches des villes existantes
            if not is_too_close((x, y), villes):
                villes.append((x, y))
            else:
                # Ajuster x et y si nécessaire pour éviter les superpositions
                found = False
                for adjust_x in [1, -1]:
                    for adjust_y in [1, -1]:
                        x_adjusted = x + adjust_x
                        y_adjusted = y + adjust_y
                        if not is_too_close((x_adjusted, y_adjusted), villes):
                            villes.append((x_adjusted, y_adjusted))
                            found = True
                            break
                    if found:
                        break

            # Arrêter de générer des points si nous avons atteint le nombre souhaité
            if len(villes) == num_villes:
                break
        # Arrêter la boucle externe si nous avons atteint le nombre souhaité de villes
        if len(villes) == num_villes:
            break

# Calcul du barycentre
total_x = sum(x for x, y in villes)
total_y = sum(y for x, y in villes)
barycentre = (round(total_x/num_villes), round(total_y/num_villes))

# Ajouter le barycentre comme première ville avec demande 0
villes.insert(0, barycentre)

# Affichage des données dans la console
for i, (x, y) in enumerate(villes):
    demande = 0 if i == 0 else random.randint(1, 35)  # demande est 0 pour le barycentre, aléatoire pour les autres
    print(f"{i + 1} {x} {y} {demande}")

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