import pandas as pd
import os

def extract_id_from_image(image_name):
    # Extraire le numéro de l'image à partir du nom du fichier
    return int(os.path.splitext(image_name)[0])

def combine_excel_files(grostest_path, classif_test_path):
    # Lire les fichiers Excel existants
    print("Reading grostest.xlsx")
    try:
        df_grostest = pd.read_excel(grostest_path)
        print("Grostest Data:")
        print(df_grostest.head())
    except Exception as e:
        print(f"Failed to read grostest.xlsx: {e}")
        return

    print("Reading classif_test.xlsx")
    try:
        df_classif_test = pd.read_excel(classif_test_path)
        print("Classif Test Data:")
        print(df_classif_test.head())
    except Exception as e:
        print(f"Failed to read classif_test.xlsx: {e}")
        return

    # Ajouter une colonne 'ID' dans df_grostest en extrayant le numéro d'image
    print("Extracting IDs from image names in grostest data")
    try:
        df_grostest['ID'] = df_grostest['Image'].apply(extract_id_from_image)
        df_grostest.drop(columns=['Image'], inplace=True)  # Supprimer la colonne 'Image' après extraction
        print("Updated Grostest Data with IDs:")
        print(df_grostest.head())
    except Exception as e:
        print(f"Failed to extract IDs from image names: {e}")
        return

    # Vérifier les noms des colonnes avant la fusion
    print("Columns in classif_test.xlsx:", df_classif_test.columns)
    print("Columns in grostest.xlsx after adding ID:", df_grostest.columns)

    # Vérifier les types de données des colonnes
    print("Data types in classif_test.xlsx:")
    print(df_classif_test.dtypes)
    print("Data types in grostest.xlsx:")
    print(df_grostest.dtypes)

    # Fusionner les nouvelles données avec les données existantes
    print("Merging data")
    try:
        df_combined = pd.merge(df_classif_test, df_grostest, on='ID', how='left')
        print("Combined Data:")
        print(df_combined.head())
    except Exception as e:
        print(f"Failed to merge data: {e}")
        return

    # Sauvegarder les données combinées dans le fichier Excel existant
    print("Saving combined data to classif_test.xlsx")
    try:
        df_combined.to_excel(classif_test_path, index=False)
        print(f"Combined results saved to {classif_test_path}")
    except Exception as e:
        print(f"Failed to save combined data: {e}")

if __name__ == "__main__":
    grostest_path = 'train/grostest.xlsx'
    classif_test_path = 'train/classif_test.xlsx'

    combine_excel_files(grostest_path, classif_test_path)