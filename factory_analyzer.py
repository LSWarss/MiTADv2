from factor_analyzer.factor_analyzer import calculate_kmo
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pandas as pd
from sklearn.datasets import load_iris
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt

''' BFI(dataset based on personality assessment project),
# which were collected using a 6 point response scale:
# 1 Very Inaccurate,
# 2 Moderately Inaccurate,
# 3 Slightly Inaccurate
# 4 Slightly Accurate,
# 5 Moderately Accurate,
# 6 Very Accurate. from: https://vincentarelbundock.github.io/Rdatasets/datasets.html
'''
df = pd.read_csv("bfi.csv")

# Dropping unnecessary columns
df.drop(['gender', 'education', 'age'], axis=1, inplace=True)

# Dropping missing values rows
df.dropna(inplace=True)

# Test sferyczności Bartletta sprawdza,
# czy obserwowane zmienne w ogóle współrealizują
# się przy użyciu obserwowanej macierzy korelacji z macierzą tożsamości.
# Jeśli test okazał się statystycznie nieistotny, nie należy stosować analizy czynnikowej.
chi_square_value, p_value = calculate_bartlett_sphericity(df)
print(chi_square_value, p_value)

# W tym teście Bartletta wartość p wynosi 0.
# Test był statystycznie istotny, co wskazuje,
# że obserwowana macierz korelacji nie jest macierzą tożsamości.
kmo_all, kmo_model = calculate_kmo(df)
print(kmo_model)


fa = FactorAnalyzer()
fa.analyze(df, 25, rotation=None)
# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev