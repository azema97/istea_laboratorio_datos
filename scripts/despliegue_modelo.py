from azureml.core import Workspace
from registro_modelo import nombre_modelo

# librerias de entorno
from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies
# librerias de despliegue
from azureml.core.model import Model, InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice

# Conectarse al área de trabajo
ws = Workspace.from_config()

# variables
nombre_servicio = 'rf-model-servicio'

# Cargar el modelo
model = Model(ws, nombre_modelo)

# Definir el entorno
env = Environment(name="proyecto_env")
conda_dep = CondaDependencies()
conda_dep.add_conda_package("scikit-learn")
conda_dep.add_conda_package("pandas")
conda_dep.add_conda_package("flask")
env.python.conda_dependencies = conda_dep

env.register(workspace=ws)

# Configurar la inferencia
inference_config = InferenceConfig(entry_script="score.py", environment=env)

# Configurar el despliegue ACI
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Desplegar el modelo como un servicio web
service = Model.deploy(ws, nombre_servicio, [model], inference_config=inference_config, deployment_config=aci_config)
service.wait_for_deployment(show_output=True)
print(service.state)

uri = service.scoring_uri
print(f'Dirección URI a la cual enviar las peticiones: {uri}')
