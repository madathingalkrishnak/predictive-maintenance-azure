#!/bin/bash
# Azure Resource Provisioning for Predictive Maintenance Project
# Run after: az login
# Replace <INITIALS> with your initials (e.g., kkm)

set -e

RESOURCE_GROUP="rg-predictive-maintenance"
LOCATION="northcentralus"
IOT_HUB_NAME="iot-hub-pred-maint-<INITIALS>"
STORAGE_ACCOUNT="sadlpredmaint<INITIALS>"
DATABRICKS_WORKSPACE="dbw-pred-maint-<INITIALS>"
DEVICE_ID="turbofan-engine-001"

echo "=== Registering resource providers ==="
az provider register --namespace Microsoft.Devices
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.Databricks

echo "=== Creating Resource Group ==="
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "=== Creating IoT Hub (Free Tier) ==="
az iot hub create \
  --resource-group $RESOURCE_GROUP \
  --name $IOT_HUB_NAME \
  --sku F1 \
  --partition-count 2

echo "=== Registering IoT Device ==="
az extension add --name azure-iot
az iot hub device-identity create \
  --hub-name $IOT_HUB_NAME \
  --device-id $DEVICE_ID

echo "=== Creating Data Lake Storage (Gen2) ==="
az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS \
  --kind StorageV2 \
  --hns true

STORAGE_KEY=$(az storage account keys list \
  --account-name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query "[0].value" -o tsv)

echo "=== Creating Storage Containers ==="
az storage container create --name raw-data --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY
az storage container create --name processed-data --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY
az storage container create --name ml-artifacts --account-name $STORAGE_ACCOUNT --account-key $STORAGE_KEY

echo "=== Setting Up IoT Hub Routing to Data Lake ==="
STORAGE_CONN=$(az storage account show-connection-string \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query "connectionString" -o tsv)

SUBSCRIPTION_ID=$(az account show --query "id" -o tsv)

az iot hub routing-endpoint create \
  --hub-name $IOT_HUB_NAME \
  --resource-group $RESOURCE_GROUP \
  --endpoint-name datalake-endpoint \
  --endpoint-type azurestoragecontainer \
  --endpoint-resource-group $RESOURCE_GROUP \
  --endpoint-subscription-id $SUBSCRIPTION_ID \
  --connection-string "$STORAGE_CONN" \
  --container raw-data \
  --encoding json

az iot hub route create \
  --hub-name $IOT_HUB_NAME \
  --resource-group $RESOURCE_GROUP \
  --route-name route-to-datalake \
  --endpoint-name datalake-endpoint \
  --source devicemessages \
  --condition true

echo "=== Creating Databricks Workspace ==="
az databricks workspace create \
  --resource-group $RESOURCE_GROUP \
  --name $DATABRICKS_WORKSPACE \
  --location $LOCATION \
  --sku premium

echo ""
echo "=== SETUP COMPLETE ==="
echo ""
echo "IoT Hub:     $IOT_HUB_NAME"
echo "Storage:     $STORAGE_ACCOUNT"
echo "Databricks:  $DATABRICKS_WORKSPACE"
echo ""
echo "Next steps:"
echo "  1. Get device connection string:"
echo "     az iot hub device-identity connection-string show --hub-name $IOT_HUB_NAME --device-id $DEVICE_ID -o tsv"
echo "  2. Get Databricks workspace URL:"
echo "     az databricks workspace show --resource-group $RESOURCE_GROUP --name $DATABRICKS_WORKSPACE --query workspaceUrl -o tsv"