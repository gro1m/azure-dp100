# azure-dp100

Resources: 
- https://github.com/MicrosoftLearning/DP100/tree/master/labdocs
- https://docs.microsoft.com/en-us/learn/certifications/exams/dp-100
- https://www.itexams.com/exam/DP-100

## 1 Set up an Azure Machine Learning Workspace (30-35%)
### 1.1 Create an Azure Machine Learning workspace
#### 1.1.1 create an Azure Machine Learning workspace 
Reference: https://docs.microsoft.com/en-us/learn/modules/intro-to-azure-machine-learning-service/2-azure-ml-workspace

You can create a workspace in any of the following ways:

* In the Microsoft Azure portal, create a new Machine Learning resource, specifying the subscription, resource group and workspace name.
* Use the Azure Machine Learning Python SDK to run code that creates a workspace. For example, the following code creates a workspace named aml-workspace (assuming the Azure ML SDK for Python is installed and a valid subscription ID is specified):

```python
    from azureml.core import Workspace
    
    ws = Workspace.create(name='aml-workspace', 
                      subscription_id='123456-abc-123...',
                      resource_group='aml-resources',
                      create_resource_group=True,
                      location='eastus'
                     )
 ```

Use the Azure Command Line Interface (CLI) with the Azure Machine Learning CLI extension. For example, you could use the following command (which assumes a resource group named aml-resources has already been created):
```bash
    az ml workspace create -w 'aml-workspace' -g 'aml-resources'
```

Create an Azure Resource Manager template, see the[Azure Machine Learning documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-workspace-template?tabs=azcli).

- configure workspace settings
- manage a workspace by using Azure Machine Learning studio
### 1.2 Manage data objects in an Azure Machine Learning workspace
- register and maintain data stores
- create and manage datasets
### 1.3 Manage experiment compute contexts
- create a compute instance
- determine appropriate compute specifications for a training workload
- create compute targets for experiments and training

## 2 Run Experiments and Train Models (25-30%)
### 2.1 Create models by using Azure Machine Learning Designer
- create a training pipeline by using Azure Machine Learning designer
- ingest data in a designer pipeline
- use designer modules to define a pipeline data flow
- use custom code modules in designer
### 2.2 Run training scripts in an Azure Machine Learning workspace
- create and run an experiment by using the Azure Machine Learning SDK
- consume data from a data store in an experiment by using the Azure Machine Learning
SDK
- consume data from a dataset in an experiment by using the Azure Machine Learning
SDK
- choose an estimator for a training experiment
### 2.3 Generate metrics from an experiment run
- log metrics from an experiment run
- retrieve and view experiment outputs
- use logs to troubleshoot experiment run errors
### 2.4 Automate the model training process
- create a pipeline by using the SDK
- pass data between steps in a pipeline
- run a pipeline
- monitor pipeline runs

## 3 Optimize and Manage Models (20-25%)
### 3.1 Use Automated ML to create optimal models
- use the Automated ML interface in Azure Machine Learning studio
- use Automated ML from the Azure Machine Learning SDK
- select scaling functions and pre-processing options
- determine algorithms to be searched
- define a primary metric
- get data for an Automated ML run
- retrieve the best model
### 3.2 Use Hyperdrive to tune hyperparameters
- select a sampling method
- define the search space
- define the primary metric
- define early termination options
- find the model that has optimal hyperparameter values
### 3.3 Use model explainers to interpret models
- select a model interpreter
- generate feature importance data
Manage models
- register a trained model
- monitor model history
- monitor data drift 

## 4 Deploy and Consume Models (20-25%)
### 4.1 Create production compute targets
- consider security for deployed services
- evaluate compute options for deployment
### 4.2 Deploy a model as a service
- configure deployment settings
- consume a deployed service
- troubleshoot deployment container issues
### 4.3 Create a pipeline for batch inferencing
- publish a batch inferencing pipeline
- run a batch inferencing pipeline and obtain outputs
### 4.4 Publish a designer pipeline as a web service
- create a target compute resource
- configure an Inference pipeline
- consume a deployed endpoint
