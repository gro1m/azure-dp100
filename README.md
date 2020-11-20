# azure-dp100

Resources: 
- https://github.com/MicrosoftLearning/DP100/tree/master/labdocs
- https://docs.microsoft.com/en-us/learn/certifications/exams/dp-100
- https://www.itexams.com/exam/DP-100
- https://docs.microsoft.com/en-us/azure/machine-learning/

## 1 Set up an Azure Machine Learning Workspace (30-35%)
### 1.1 Create an Azure Machine Learning workspace
A workspace defines the boundary for a set of related machine learning assets. You can use workspaces to group machine learning assets based on projects, deployment environments (for example, test and production), teams, or some other organizing principle. The assets in a workspace include:

* Compute targets for development, training, and deployment.
* Data for experimentation and model training.
* Notebooks containing shared code and documentation.
* Experiments, including run history with logged metrics and outputs.
* Pipelines that define orchestrated multi-step processes.
* Models that you have trained.
* Workspaces as Azure Resources

Workspaces are Azure resources, and as such they are defined within a resource group in an Azure subscription, along with other related Azure resources that are required to support the workspace.
![](workspace_overview.png)

The Azure resources created alongside a workspace include:
* A storage account - used to store files used by the workspace as well as data for experiments and model training.
* An Application Insights instance, used to monitor predictive services in the workspace.
* An Azure Key Vault instance, used to manage secrets such as authentication keys and credentials used by the workspace.
* A container registry, created as-needed to manage containers for deployed models.

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

**NOTE**
*Basic* edition as opposed to *Enterprise* edition of the workspace have lower cost, but do not include capabilities like AutoML, the Visual Designer, and data drift monitoring. For the details, see: https://azure.microsoft.com/en-us/pricing/details/machine-learning/.

#### 1.1.2 configure workspace settings
fter installing the SDK package in your Python environment, you can write code to connect to your workspace and perform machine learning operations. The easiest way to connect to a workspace is to use a workspace configuration file, which includes the Azure subscription, resource group, and workspace details as shown here:

```json
{
    "subscription_id": "1234567-abcde-890-fgh...",
    "resource_group": "aml-resources",
    "workspace_name": "aml-workspace"
}
```

Tip: You can download a configuration file for a workspace from the Overview page of its blade in the Azure portal or from Azure Machine Learning studio.

To connect to the workspace using the configuration file, you can use the from_config method of the Workspace class in the SDK, as shown here:

```python
from azureml.core import Workspace

ws = Workspace.from_config()
```

By default, the from_config method looks for a file named config.json in the folder containing the Python code file, but you can specify another path if necessary.

As an alternative to using a configuration file, you can use the get method of the Workspace class with explicitly specified subscription, resource group, and workspace details as shown here - though the configuration file technique is generally preferred due to its greater flexibility when using multiple scripts:

```python
from azureml.core import Workspace

ws = Workspace.get(name='aml-workspace',
                   subscription_id='1234567-abcde-890-fgh...',
                   resource_group='aml-resources')
```
Whichever technique you use, if there is no current active session with your Azure subscription, you will be prompted to authenticate.

#### 1.1.3 manage a workspace by using Azure Machine Learning studio
You can manage the assets in your Azure Machine Learning workspace in the Azure portal, but as this is a general interface for managing all kinds of resources in Azure, data scientists and other users involved in machine learning operations may prefer to use a more focused, dedicated interface.

![](aml_studio.png)

Azure Machine Learning studio is a web-based tool for managing an Azure Machine Learning workspace. It enables you to create, manage, and view all of the assets in your workspace and provides the following graphical tools:

Designer: A drag and drop interface for "no code" machine learning model development.
Automated Machine Learning: A wizard interface that enables you to train a model using a combination of algorithms and data preprocessing techniques to find the best model for your data.
 Note

A previously released tool named Azure Machine Learning Studio provided a free service for drag and drop machine learning model development. The studio interface for the Azure Machine Learning service includes this capability in the designer tool, as well as other workspace asset management capabilities.

To use Azure Machine Learning studio, use a a web browser to navigate to https://ml.azure.com and sign in using credentials associated with your Azure subscription. You can then select the subscription and workspace you want to manage.

### 1.2 Manage data objects in an Azure Machine Learning workspace
#### 1.2.1 register and maintain data stores
#### 1.2.2 create and manage datasets
2. In the *Studio* interface, view the **Datasets** page. Datasets represent specific data files or tables that you plan to work with in Azure ML.
3. Create a new dataset from web files, using the following settings:
    * **Basic Info**:
        * **Web URL**: https://aka.ms/diabetes-data
        * **Name**: diabetes dataset (*be careful to match the case and spacing*)
        * **Dataset type**: Tabular
        * **Description**: Diabetes data
    * **Settings and preview**:
        * **File format**: Delimited
        * **Delimiter**: Comma
        * **Encoding**: UTF-8
        * **Column headers**: Use headers from first file
        * **Skip rows**: None
    * **Schema**:
        * Include all columns other than **Path**
        * Review the automatically detected types
    * **Confirm details**:
        * Do not profile the dataset after creation
4. After the dataset has been created, open it and view the **Explore** page to see a sample of the data. This data represents details from patients who have been tested for diabetes, and you will use it in many of the subsequent labs in this course.

    > **Note**: You can optionally generate a *profile* of the dataset to see more details. You'll explore datasets in more detail later in the course.

### 1.3 Manage experiment compute contexts
#### 1.3.1 create a compute instance
1. In the Azure Machine Learning studio web interface for your workspace, view the **Compute** page. This is where you'll manage all the compute targets for your data science activities.
2. On the **Compute Instances** tab, add a new compute instance with the following settings. You'll use this as a workstation from which to test your model:
    - **Compute name**: *enter a unique name*
    - **Virtual Machine type**: CPU
    - **Virtual Machine size**: Standard_DS1_v2
3. While the compute instance is being created, switch to the **Compute Clusters** tab, and add a new compute cluster with the following settings. You'll use this to train a machine learning model:
    - **Compute name**: *enter a unique name*
    - **Virtual Machine type**: CPU
    - **Virtual Machine priority**: Dedicated
    - **Virtual Machine size**: Standard_DS2_v2
    - **Minimum number of nodes**: 0
    - **Maximum number of nodes**: 2
    - **Idle seconds before scale down**: 120
4. Note the **Inference Clusters** tab. This is where you can create and manage compute targets on which to deploy your trained models as web services for client applications to consume.
5. Note the **Attached Compute** tab. This is where you could attach a virtual machine or Databricks cluster that exists outside of your workspace.
#### 1.3.2 determine appropriate compute specifications for a training workload
#### 1.3.3 create compute targets for experiments and training

### --- Introduction Azure ML SDK --- ###
Azure Machine Learning (Azure ML) is a cloud-based service for creating and managing machine learning solutions. It's designed to help data scientists leverage their existing data processing and model development skills and frameworks, and help them scale their workloads to the cloud. The Azure ML SDK for Python provides classes you can use to work with Azure ML in your Azure subscription.

Check the Azure ML SDK Version
Let's start by importing the azureml-core package and checking the version of the SDK that is installed.

```python
import azureml.core
print("Ready to use Azure ML", azureml.core.VERSION)
```
View Azure ML resources
First connect to workspace, like so:
```python
from azureml.core import Workspace

ws = Workspace.from_config()
print(ws.name, "loaded")
```
and then view resources:
```python
from azureml.core import ComputeTarget, Datastore, Dataset

print("Compute Targets:")
for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print("\t", compute.name, ':', compute.type)
    
print("Datastores:")
for datastore_name in ws.datastores:
    datastore = Datastore.get(ws, datastore_name)
    print("\t", datastore.name, ':', datastore.datastore_type)
    
print("Datasets:")
for dataset_name in list(ws.datasets.keys()):
    dataset = Dataset.get_by_name(ws, dataset_name)
    print("\t", dataset.name)
```


## 2 Run Experiments and Train Models (25-30%)
### 2.1 Create models by using Azure Machine Learning Designer
#### 2.1.1 create a training pipeline by using Azure Machine Learning designer

With the data flow steps defined, you're now ready to run the training pipeline and train the model.

1. Verify that your pipeline looks similar to the following:

    ![Visual Training Pipeline](visual-training.png)

2. At the top right, click **Submit**. Then when prompted, create a new *experiment* named **visual-training**, and run it.  This will initialize the compute cluster and then run the pipeline, which may take 10 minutes or longer. You  can see the status of the pipeline run above the top right of the design canvas.

    **Tip**: While it's running, you can view the pipeline and experiment that have been created in the **Pipelines** and **Experiments** pages. Switch back to the **Visual Diabetes Training** pipeline on the **Designer** page when you're done.

3. After the **Normalize Data** module has completed, select it, and in the **Settings** pane, on the **Outputs + logs** tab, under **Data outputs** in the **Transformed dataset** section, click the **Visualize** icon, and note that you can view statistics and distribution visualizations for the transformed columns.
4. Close the **Normalize Data** visualizations and wait for the rest of the modules to complete. Then visualize the output of the **Evaluate Model** module to see the performance metrics for the model.

    **Note**: The performance of this model isn't all that great, partly because we performed only minimal feature engineering and pre-processing. You could try some different classification algorithms and compare the results (you can connect the outputs of the **Split Data** module to multiple **Train Model** and **Score Model** modules, and you can connect a second scored model to the **Evaluate Model** module to see a side-by-side comparison). The point of the exercise is simply to introduce you to the Designer interface, not to train a perfect model!

#### 1.2.2 ingest data in a designer pipeline
#### 1.2.3 use designer modules to define a pipeline data flow
#### 1.2.4 use custom code modules in designer

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
