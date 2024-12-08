from azureml.core import Workspace
from azureml.core.model import Model

# Conectarse al área de trabajo
ws = Workspace.from_config()

# Variables
ruta = "../models/random_forest_model.pkl"
nombre_modelo = 'rf_model'

# Registrar el modelo en el área de trabajo
model = Model.register(workspace = ws,
                       model_name = nombre_modelo,
                       model_path = "../models/random_forest_model.pkl")

# Resultado esperado: "Registering model rf_model"