import pandas as pd

def combine_excel_files(grostest_path, classif_test_path, output_path):
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

    # Fusionner les nouvelles données avec les données existantes
    print("Merging data")
    try:
        df_combined = pd.merge(df_classif_test, df_grostest, on='ID', how='left')
        print("Combined Data:")
        print(df_combined.head())
    except Exception as e:
        print(f"Failed to merge data: {e}")
        return

    # Sauvegarder les données combinées dans un nouveau fichier Excel
    print("Saving combined data to new Excel file")
    try:
        df_combined.to_excel(output_path, index=False)
        print(f"Combined results saved to {output_path}")
    except Exception as e:
        print(f"Failed to save combined data: {e}")

if __name__ == "__main__":
    grostest_path = 'train/grostest.xlsx'
    classif_test_path = 'classif_test.xlsx'
    output_path = 'classif_combined.xlsx'

    combine_excel_files(grostest_path, classif_test_path, output_path)