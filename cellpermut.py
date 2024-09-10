from matplotlib.cbook import _premultiplied_argb32_to_unmultiplied_rgba8888
import pandas as pd
import math
import pickle
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import glob
import multiprocessing


def get_cell_to_voisin(p1, points, d_min, d_max):
    """ """
    neighbors = []
    for p2 in points:
        if p1 != p2:  # Ne pas comparer un point avec lui-même
            x = float(p1.split("_")[0])
            y = float(p1.split("_")[1])
            x_candidate = float(p2.split("_")[0])
            y_candidate = float(p2.split("_")[1])
            distance = math.sqrt((x - x_candidate)**2 + (y - y_candidate)**2)
            if distance <= radius_max and distance >= radius_min:
                neighbors.append(p2)
    return (p1, neighbors)


def read_cells_from_file(file_name):
    """ """
    points = []
    df = pd.read_csv(file_name)
    for index, row in df.iterrows():
        x = row['Centroid X µm']
        y = row['Centroid Y µm']
        points.append(f"{x}_{y}")
    return points


def get_cell_to_voisin_multiproc(points, d_min, d_max):
    with multiprocessing.Pool() as pool:
        # On envoie la recherche des voisins de chaque point en parallèle
        results = pool.starmap(get_cell_to_voisin, [(p1, points, d_min, d_max) for p1 in points])
    
    # Résultat sous forme de dictionnaire
    cell_to_voisin = {p1: neighbors for p1, neighbors in results}
    return cell_to_voisin


def map_cells_to_pop(data_file):
    """ """
    cell_to_pop = {}
    df = pd.read_csv(data_file)
    for index, row in df.iterrows():
        x = row['Centroid X µm']
        y = row['Centroid Y µm']
        population = row['OmiqFilter']
        cell_id = f"{x}_{y}"
        cell_to_pop[cell_id] = population

    return cell_to_pop


def get_pop_to_voisin(cell_to_voisin, cell_to_pop):
    """ """
    pop_to_voisin = {}
    for c in cell_to_voisin:
        p = cell_to_pop[c]
        if p not in pop_to_voisin:
            pop_to_voisin[p] = {}
        for cv in cell_to_voisin[c]:
            pv = cell_to_pop[cv]
            if pv not in pop_to_voisin[p]:
                pop_to_voisin[p][pv] = 1
            else:
                pop_to_voisin[p][pv] +=1
    return pop_to_voisin


    

def compute_proximity_matrix(radius_min, radius_max, data_file):
    """ """

    # load data
    cell_list = read_cells_from_file(data_file)
    
    # compute distances
    cell_to_voisin = get_cell_to_voisin_multiproc(cell_list, radius_min, radius_max)
    cell_to_pop = map_cells_to_pop(data_file)
    pop_to_voisin = get_pop_to_voisin(cell_to_voisin, cell_to_pop)

    # return proximity matrix
    return pop_to_voisin 


def compute_proximity_matrix_folder(folder, manifest, radius_min, radius_max):
    """ """

    # ectract class
    class_to_file = {}
    df = pd.read_csv(manifest)
    for index, row in df.iterrows():
        if row['Groupe'] not in class_to_file:
            class_to_file[row['Groupe']] = [f"{folder}/{row['file']}"]
        else:
            class_to_file[row['Groupe']].append(f"{folder}/{row['file']}")

    # extract pop list
    pop_list = []
    for tf in glob.glob(f"{folder}/*.csv"):
        if tf != manifest:
            local_list = pd.read_csv(tf)['OmiqFilter'].unique()
            for pop in local_list:
                if pop not in pop_list:
                    pop_list.append(pop)

    # for each class compute pop to voisin
    class_to_pop_to_voisin = {}
    for c in class_to_file:
        file_list = class_to_file[c]
        dict_list = []
        for f in file_list:
            pop_to_voisin = compute_proximity_matrix(radius_min, radius_max, f)
            dict_list.append(pop_to_voisin)

        # assemble results
        result = {}
        for d in dict_list:
            for pop in d:
                if pop not in result:
                    result[pop] = d[pop]
                else:
                    for p2 in result[pop]:
                        if p2 in d[pop]:
                            result[pop][p2] += d[pop][p2]

        # assign pop to voisin
        class_to_pop_to_voisin[c] = result

    # return class to pop_to_voisin
    return class_to_pop_to_voisin

    


def display_proximity_matrix(pop_to_voisin):
    """ """

    # craft vector list
    vector_list = []
    vector_list_percentage = []
    for k1 in pop_to_voisin:
        vector = []
        for k2 in pop_to_voisin:
            if k2 in pop_to_voisin[k1]:
                scalar = pop_to_voisin[k1][k2]
            else:
                scalar = 0
            vector.append(scalar)
        vector_list.append(vector)

        # deal with percentages
        total = sum(vector)
        percentages = [(value / total) * 100 for value in vector]
        vector_list_percentage.append(percentages)

    # plot graph
    ax = sns.heatmap(
        vector_list_percentage,
        linewidth=0.5,
        annot=True,
        xticklabels = list(pop_to_voisin.keys()),
        yticklabels = list(pop_to_voisin.keys())
    )
    plt.title('Proximity Matrix (%)')
    plt.show()



def display_voisin_bar(pop_to_voisins, pop):
    """ """

    # load data
    data = pop_to_voisins[pop]
    data = dict(sorted(data.items(), key=lambda item: item[1]))

    # compute percentage
    total = sum(data.values())
    data_percentage = {}
    for k in data:
        data_percentage[k] = (data[k] / total)*100
      
    # display
    plt.barh(data_percentage.keys(), data_percentage.values())
    plt.title(f"Voisinage de {pop}")
    plt.show()



def display_voisin_pie(pop_to_voisins, pop):
    """ """

    # load data
    data = pop_to_voisins[pop]
    data = dict(sorted(data.items(), key=lambda item: item[1]))

    # Extracting the labels and sizes from the dictionary
    labels = data.keys()
    sizes = data.values()

    # Plotting the pie chart
    plt.figure(figsize=(6, 6))  # Optional: Adjust the figure size
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)

    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Display the pie chart
    plt.title(f"Voisinage de {pop}")
    plt.show()



def display_multiclass_bar(group_to_voisin, pop):
    """ """

    data = []
    group_list = []
    for group in group_to_voisin:
        data.append(group_to_voisin[group][pop])
        group_list.append(group)

    # Conversion des valeurs en pourcentages
    pop_list = []
    for series in data:
        total = sum(series.values())
        for key in series:
            series[key] = (series[key] / total) * 100
            if key not in pop_list:
                pop_list.append(key)
    
    # fill missing pops
    for series in data:
        for pop in pop_list:
            if pop not in series:
                series[pop] = 0

    # Récupération des catégories et des valeurs
    categories = list(data[0].keys())
    n_categories = len(categories)
    bar_width = 0.2  # Largeur des barres
    index = np.arange(n_categories)  # Position des barres pour la première série

    # Création du barplot
    for i, series in enumerate(data):
        values = list(series.values())
        plt.bar(index + i * bar_width, values, bar_width, label=group_list[i])

    # Ajout des étiquettes et du titre
    plt.xlabel('Populations')
    plt.ylabel('Valeurs (%)')
    plt.title(f"Voisins de la population {pop} (valeurs en %)")
    plt.xticks(index + bar_width, categories, rotation=45)
    plt.legend()

    # Affichage du graphique
    plt.show()


    
def display_class_matrix(group_to_voisin, group):
    """ """
    pop_to_voisin = group_to_voisin[group]
    display_proximity_matrix(pop_to_voisin)    


if __name__ == "__main__":

    # parameters
    population = "CD8"
    cell_id = 3421
    radius_min = 5
    radius_max = 10
    data_file = "data/Ca15-measurements.csv"

    # Generate & save matrix
    matrix = compute_proximity_matrix(radius_min, radius_max, data_file)
    with open('matrix.pickle', 'wb') as handle:
        pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # load & display matrix
    with open('matrix.pickle', 'rb') as handle:
        pop_to_voisin = pickle.load(handle)

    # display_proximity_matrix(pop_to_voisin)
    # display_voisin_bar(pop_to_voisin, 'Tconv')
    # display_voisin_pie(pop_to_voisin, 'Tconv')

    result = compute_proximity_matrix_folder("data", "data/Groupe.csv", radius_min, radius_max)
    with open('multi_matrix.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


    # with open('matrix.pickle', 'rb') as handle:
    #     machin = pickle.load(handle)


    # display_multiclass_bar(machin, 'Tconv')
    # display_class_matrix(machin, 'EP')




    # Utilisation du multiprocessing pour trouver les points voisins

    



    

    
