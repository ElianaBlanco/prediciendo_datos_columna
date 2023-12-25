from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Eliminamos las columnas DEATH_EVENT, age y categoria_edad para obtener la matriz X
X = df.drop(columns=['DEATH_EVENT', 'age', 'categoria_edad'])

# Definimos y como la columna de edades
y = df['age']

# Inicializamos y ajustamos el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X, y)

# Predecimos las edades
edades_predichas = modelo.predict(X)

# Calculamos el error cuadrático medio
mse = mean_squared_error(y, edades_predichas)
print(f'Error cuadrático medio: {mse}')

