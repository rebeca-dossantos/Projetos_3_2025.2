import pandas as pd
import numpy as np
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


def main(input_path='database.csv', output_path='database_balanced.csv', target='Diabetes_binary', random_state=42):
    p_in = Path(input_path)
    if not p_in.exists():
        print(f"Arquivo não encontrado: {input_path}")
        return

    df = pd.read_csv(p_in)

    # Garantir que valores da target estão binarizados (segue a transformação do EDA original)
    df = df.rename(columns={'Diabetes_012': 'Diabetes_binary'})
    if 'Diabetes_binary' in df.columns:
        df['Diabetes_binary'] = df['Diabetes_binary'].replace({2: 1})

    if target not in df.columns:
        print(f"Coluna alvo '{target}' não encontrada no dataset.")
        return

    X = df.drop(columns=[target])
    y = df[target]

    # Identificar colunas binárias/categóricas (nunique == 2) para usar com SMOTENC
    binary_cols = [col for col in X.columns if X[col].nunique() == 2]

    # Converter binárias para inteiros (SMOTENC espera categorias codificadas como inteiros)
    for col in binary_cols:
        X[col] = X[col].astype(int)

    try:
        from imblearn.over_sampling import SMOTENC, SMOTE
    except Exception as e:
        print("imblearn não encontrado. Instale via: pip install imbalanced-learn")
        raise

    print("Distribuição original:\n", y.value_counts())

    if len(binary_cols) > 0:
        cat_indices = [X.columns.get_loc(c) for c in binary_cols]
        smote = SMOTENC(categorical_features=cat_indices, random_state=random_state)
    else:
        smote = SMOTE(random_state=random_state)

    X_res, y_res = smote.fit_resample(X, y)

    df_res = pd.DataFrame(X_res, columns=X.columns)
    df_res[target] = y_res

    df_res.to_csv(output_path, index=False)

    print("Distribuição balanceada:\n", pd.Series(y_res).value_counts())
    print(f"Arquivo balanceado salvo em: {output_path}")


if __name__ == '__main__':
    main()
