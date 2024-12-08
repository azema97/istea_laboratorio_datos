#!/bin/bash
cd scripts

echo "Registrando modelo en la nube..."
python registro_modelo.py

echo "Desplegando modelo..."
python despliegue_modelo.py