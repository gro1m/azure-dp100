# Summary for Azure DP100 Exam

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

**Azure portal**
1. In the *Studio* interface, view the **Datasets** page. Datasets represent specific data files or tables that you plan to work with in Azure ML.
2. Create a new dataset from web files, using the following settings:
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
3. After the dataset has been created, open it and view the **Explore** page to see a sample of the data. This data represents details from patients who have been tested for diabetes, and you will use it in many of the subsequent labs in this course.

    > **Note**: You can optionally generate a *profile* of the dataset to see more details. You'll explore datasets in more detail later in the course.
    
Datasets are versioned packaged data objects that can be easily consumed in experiments and pipelines. Datasets are the recommended way to work with data, and are the primary mechanism for advanced Azure Machine Learning capabilities like data labeling and data drift monitoring.

**Types of Dataset**
Datasets are typically based on files in a datastore, though they can also be based on URLs and other sources. You can create the following types of dataset:

* Tabular: The data is read from the dataset as a table. You should use this type of dataset when your data is consistently structured and you want to work with it in common tabular data structures, such as Pandas dataframes.

* File: The dataset presents a list of file paths that can be read as though from the file system. Use this type of dataset when your data is unstructured, or when you need to process the data at the file level (for example, to train a convolutional neural network from a set of image files).

You can create datasets from individual files or multiple file paths. The paths can include wildcards (for example, /files/*.csv) making it possible to encapsulate data from a large number of files in a single dataset.

You can create a dataset and work with it immediately, and you can then register the dataset in the workspace to make it available for use in experiments and data processing pipelines later.

You can create datasets by using the visual interface in Azure Machine Learning studio, or you can use the Azure Machine Learning SDK.

**Creating and Registering Tabular Datasets**
To create a tabular dataset using the SDK, use the from_delimited_files method of the Dataset.Tabular class, like this:
```python
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
csv_paths = [(blob_ds, 'data/files/current_data.csv'),
             (blob_ds, 'data/files/archive/*.csv')]
tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
tab_ds = tab_ds.register(workspace=ws, name='csv_table')
``` 

The dataset in this example includes data from two file paths within the default datastore:
* The *current_data.csv* file in the *data/files* folder.
* All *.csv* files in the *data/files/archive/* folder.

After creating the dataset, the code registers it in the workspace with the name *csv_table*.

**Creating and Registering File Datasets**
To create a file dataset using the SDK, use the from_files method of the Dataset.File class, like this:
```python
from azureml.core import Dataset

blob_ds = ws.get_default_datastore()
file_ds = Dataset.File.from_files(path=(blob_ds, 'data/files/images/*.jpg'))
file_ds = file_ds.register(workspace=ws, name='img_files')
``` 

The dataset in this example includes all *.jpg* files in the *data/files/images* path within the default datastore:

After creating the dataset, the code registers it in the workspace with the name *img_files*.

**Retrieving a Registered Dataset**
After registering a dataset, you can retrieve it by using any of the following techniques:

The datasets dictionary attribute of a Workspace object.

The *get_by_name* or *get_by_id* method of the Dataset class.

Both of these techniques are shown in the following code:
```python
import azureml.core
from azureml.core import Workspace, Dataset

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Get a dataset from the workspace datasets collection
ds1 = ws.datasets['csv_table']

# Get a dataset by name from the datasets class
ds2 = Datasets.get_by_name(ws, 'img_files')

# Get the files in the dataset
for file_path in ds2.to_path():
   print(file_path)
``` 

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
The most common ways to create or attach a compute target are to use the Compute page in Azure Machine Learning studio, or to use the Azure Machine Learning SDK to provision compute targets in code. Additionally, you can create compute targets using the Azure Machine Learning extension in Visual Studio Code, or by using the Azure command line interface (CLI) extension for Azure Machine Learning.

**Creating a Managed Compute Target with the SDK*3
A managed compute target is one that is managed by Azure Machine Learning, such as an Azure Machine Learning training cluster.

To create an Azure Machine Learning compute cluster compute target, use the azureml.core.compute.ComputeTarget class and the AmlCompute class, like this:
```python
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Specify a name for the compute (unique within the workspace)
compute_name = 'aml-cluster'

# Define compute configuration
compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2',
                                                       min_nodes=0, max_nodes=4,
                                                       vm_priority='dedicated')

# Create the compute
aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)
aml_cluster.wait_for_completion(show_output=True)
``` 

In this example, a cluster with up to four nodes based on the STANDARD_DS12_v2 virtual machine image will be created. The priority for the virtual machines (VMs) is set to dedicated, meaning they are reserved for use in this cluster (the alternative is to specify lowpriority, which has a lower cost but means that the VMs can be pre-empted if a higher-priority workload requires the compute).

Note: For a full list of AmlCompute configuration options, see the AmlCompute class SDK documentation.

**Attaching an Unmanaged Compute Target with the SDK**
An unmanaged compute target is one that is defined and managed outside of the Azure Machine Learning workspace; for example, an Azure virtual machine or an Azure Databricks cluster.

The code to attach an existing unmanaged compute target is similar to the code used to create a managed compute target, except that you must use the *ComputeTarget.attach()* method to attach the existing compute based on its target-specific configuration settings.

For example, the following code can be used to attach an existing Azure Databricks cluster:
```python
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, DatabricksCompute

# Load the workspace from the saved config file
ws = Workspace.from_config()

# Specify a name for the compute (unique within the workspace)
compute_name = 'db_cluster'

# Define configuration for existing Azure Databricks cluster
db_workspace_name = 'db_workspace'
db_resource_group = 'db_resource_group'
db_access_token = '1234-abc-5678-defg-90...'
db_config = DatabricksCompute.attach_configuration(resource_group=db_resource_group,
                                                   workspace_name=db_workspace_name,
                                                   access_token=db_access_token)

# Create the compute
databricks_compute = ComputeTarget.attach(ws, compute_name, db_config)
databricks_compute.wait_for_completion(True)

Checking for an Existing Compute Target
In many cases, you will want to check for the existence of a compute target, and only create a new one if there isn't already one with the specified name. To accomplish this, you can catch the ComputeTargetException exception, like this:

from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

compute_name = "aml-cluster"

# Check if the compute target exists
try:
    aml_cluster = ComputeTarget(workspace=ws, name=compute_name)
    print('Found existing cluster.')
except ComputeTargetException:
    # If not, create it
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS12_V2',
                                                           max_nodes=4)
    aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)

aml_cluster.wait_for_completion(show_output=True)
``` 

More Information: For more information about creating compute targets, see Set up and use compute targets for model training in the Azure Machine Learning documentation.

**Using Compute Targets**
After you've created or attached compute targets in your workspace, you can use them to run specific workloads; such as experiments.

To use a particular compute target, you can specify it in the appropriate parameter for an experiment run configuration or estimator. For example, the following code configures an estimator to use the compute target named aml-cluster:
```python
from azureml.core import Environment
from azureml.train.estimator import Estimator

compute_name = 'aml-cluster'

training_env = Environment.get(workspace=ws, name='training_environment')

estimator = Estimator(source_directory='experiment_folder',
                      entry_script='training_script.py',
                      environment_definition=training_env,
                      compute_target=compute_name)
 ``` 

When an experiment for the estimator is submitted, the run will be queued while the aml-cluster compute target is started and the specified environment created on it, and then the run will be processed on the compute environment.

Instead of specifying the name of the compute target, you can specify a ComputeTarget object, like this:
```python
from azureml.core import Environment
from azureml.train.estimator import Estimator
from azureml.core.compute import ComputeTarget

compute_name = "aml-cluster"

training_cluster = ComputeTarget(workspace=ws, name=compute_name)

training_env = Environment.get(workspace=ws, name='training_environment')

estimator = Estimator(source_directory='experiment_folder',
                      entry_script='training_script.py',
                      environment_definition=training_env,
                      compute_target=training_cluster)
```

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
#### 2.2.1 create and run an experiment by using the Azure Machine Learning SDK
In Azure Machine Learning, an experiment is a named process, usually the running of a script or a pipeline, that can generate metrics and outputs and be tracked in the Azure Machine Learning workspace. An experiment can be run multiple times, with different data, code, or settings; and Azure Machine Learning tracks each run, enabling you to view run history and compare results for each run.

**The Experiment Run Context**
When you submit an experiment, you use its run context to initialize and end the experiment run that is tracked in Azure Machine Learning, as shown in the following code sample:
```python
from azureml.core import Experiment

# create an experiment variable
experiment = Experiment(workspace = ws, name = "my-experiment")

# start the experiment
run = experiment.start_logging()

# experiment code goes here

# end the experiment
run.complete()
```
After the experiment run has completed, you can view the details of the run in the Experiments tab in Azure Machine Learning studio.

**Via Script**
```python
%%writefile $folder_name/diabetes_experiment.py
from azureml.core import Run
import pandas as pd
import os

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
data = pd.read_csv('diabetes.csv')

# Count the rows and log the result
row_count = (len(data))
run.log('observations', row_count)
print('Analyzing {} rows of data'.format(row_count))

# Count and log the label counts
diabetic_counts = data['Diabetic'].value_counts()
print(diabetic_counts)
for k, v in diabetic_counts.items():
    run.log('Label:' + str(k), v)
      
# Save a sample of the data in the outputs folder (which gets uploaded automatically)
os.makedirs('outputs', exist_ok=True)
data.sample(100).to_csv("outputs/sample.csv", index=False, header=True)

# Complete the run
run.complete()
``` 

This code is a simplified version of the inline code used before. However, note the following:

It uses the Run.get_context() method to retrieve the experiment run context when the script is run.
It loads the diabetes data from the folder where the script is located.
It creates a folder named outputs and writes the sample file to it - this folder is automatically uploaded to the experiment run
Now you're almost ready to run the experiment. To run the script, you must create a ScriptRunConfig that identifies the Python script file to be run in the experiment, and then run an experiment based on it.

Note: The ScriptRunConfig also determines the compute target and Python environment. If you don't specify these, a default environment is created automatically on the local compute where the code is being run (in this case, where this notebook is being run).

The following cell configures and submits the script-based experiment:
```python
import os
import sys
from azureml.core import Experiment, ScriptRunConfig
from azureml.widgets import RunDetails


# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder, 
                      script='diabetes_experiment.py') 

# submit the experiment
experiment = Experiment(workspace=ws, name='diabetes-experiment')
run = experiment.submit(config=script_config)
RunDetails(run).show()
run.wait_for_completion()
```

**View experiment run history**
```python
from azureml.core import Experiment, Run

diabetes_experiment = ws.experiments['diabetes-experiment']
for logged_run in diabetes_experiment.get_runs():
    print('Run ID:', logged_run.id)
    metrics = logged_run.get_metrics()
    for key in metrics.keys():
        print('-', key, metrics.get(key))
```

#### 2.2.2 consume data from a data store in an experiment by using the Azure Machine Learning SDK

Working with Datastores
Although it's fairly common for data scientists to work with data on their local file system, in an enterprise environment it can be more effective to store the data in a central location where multiple data scientists can access it. In this lab, you'll store data in the cloud, and use an Azure Machine Learning datastore to access it.
```python
# Connect to Your Workspace
import azureml.core
from azureml.core import Workspace
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))

# View Datastores in the Workspace
from azureml.core import Datastore
default_ds = ws.get_default_datastore() # get default datastore

for ds_name in ws.datastores: # enumerate all datastores, indicating which is the default
    print(ds_name, "- Default =", ds_name == default_ds.name)

# Get a Datastore to Work With
aml_datastore = Datastore.get(ws, 'aml_data')
print(aml_datastore.name,":", aml_datastore.datastore_type + " (" + aml_datastore.account_name + ")")

# Set the Default Datastore (for convenience)
ws.set_default_datastore('aml_data')
default_ds = ws.get_default_datastore()
print(default_ds.name)

# Upload Data to a Datastore
default_ds.upload_files(files=['./data/diabetes.csv', './data/diabetes2.csv'], # Upload the diabetes csv files in /data
                       target_path='diabetes-data/', # Put it in a folder path in the datastore
                       overwrite=True, # Replace existing files of the same name
                       show_progress=True)
                       
``` 

#### 2.2.3 consume data from a dataset in an experiment by using the Azure Machine Learning SDK
You can read data directly from a dataset, or you can pass a dataset as a named input to a script configuration or estimator.
If you have a reference to a dataset, you can access its contents directly.

For tabular datasets, most data processing begins by reading the dataset as a Pandas dataframe:
```python
df = tab_ds.to_pandas_dataframe()
# code to work with dataframe goes here
``` 

When working with a file dataset, you can use the to_path() method to return a list of the file paths encapsulated by the dataset:
```python
for file_path in file_ds.to_path():
    print(file_path)
``` 

**Passing a Dataset to an Experiment Script**
When you need to access a dataset in an experiment script, you can pass the dataset as an input to a ScriptRunConfig or an Estimator. For example, the following code passes a tabular dataset to an estimator:
```python 
estimator = SKLearn( source_directory='experiment_folder',
                     entry_script='training_script.py'
                     compute_target='local',
                     inputs=[tab_ds.as_named_input('csv_data')],
                     pip_packages=['azureml-dataprep[pandas]')
```
**> Note **: since the script will need to work with a Dataset object, you must include either the full azureml-sdk package or the azureml-dataprep package with the pandas extra library in the script's compute environment.

In the experiment script itself, you can access the input and work with the Dataset object it references like this:
```python
run = Run.get_context()
data = run.input_datasets['csv_data'].to_pandas_dataframe()
``` 

When passing a file dataset, you must specify the access mode. For example:
```python
estimator = Estimator( source_directory='experiment_folder',
                     entry_script='training_script.py',
                     compute_target='local',
                     inputs=[img_ds.as_named_input('img_data').as_download(path_on_compute='data')],
                     pip_packages=['azureml-dataprep[pandas]')
``` 
When a file dataset is passed to the estimator, a mount point from which the script can read the files has to be defined:
- for large volumes of data, you would generally use the *as_mount* method to stream the files directly from the dataset source
- When running on local compute though, you need to use the *as_download* option to download the dataset files to a local folder.

**Dataset Versioning**
Datasets can be versioned, enabling you to track historical versions of datasets that were used in experiments, and reproduce those experiments with data in the same state.

*Creating a New Version of a Dataset*
You can create a new version of a dataset by registering it with the same name as a previously registered dataset and specifying the create_new_version property:
```python 
img_paths = [(blob_ds, 'data/files/images/*.jpg'),
             (blob_ds, 'data/files/images/*.png')]
file_ds = Dataset.File.from_files(path=img_paths)
file_ds = file_ds.register(workspace=ws, name='img_files', create_new_version=True)
``` 

In this example, the .png files in the images folder have been added to the definition of the img_paths dataset example used in the previous topic.

*Retrieving a Specific Dataset version*
You can retrieve a specific version of a dataset by specifying the version parameter in the get_by_name method of the Dataset class.
```python 
img_ds = Dataset.get_by_name(workspace=ws, name='img_files', version=2)
``` 

#### 2.2.4 choose an estimator for a training experiment
```python
%%writefile $training_folder/diabetes_training.py
# Import libraries
from azureml.core import Run
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
print("Loading Data...")
diabetes = pd.read_csv('diabetes.csv')

# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Set regularization hyperparameter
reg = 0.01

# Train a logistic regression model
print('Training a logistic regression model with regularization rate of', reg)
run.log('Regularization Rate',  np.float(reg))
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/diabetes_model.pkl')

run.complete()
``` 

```python
from azureml.train.estimator import Estimator
from azureml.core import Experiment

# Create an estimator
estimator = Estimator(source_directory=training_folder,
                      entry_script='diabetes_training.py',
                      compute_target='local',
                      conda_packages=['scikit-learn']
                      )

# Create an experiment
experiment_name = 'diabetes-training'
experiment = Experiment(workspace = ws, name = experiment_name)

# Run the experiment based on the estimator
run = experiment.submit(config=estimator)
run.wait_for_completion(show_output=True)
``` 

**Use a Framework-Specific Estimator**
You used a generic Estimator class to run the training script, but you can also take advantage of framework-specific estimators that include environment definitions for common machine learning frameworks. In this case, you're using Scikit-Learn, so you can use the SKLearn estimator. This means that you don't need to specify the scikit-learn package in the configuration.
```python
from azureml.train.sklearn import SKLearn
from azureml.widgets import RunDetails

# Create an estimator
estimator = SKLearn(source_directory=training_folder,
                    entry_script='diabetes_training.py',
                    script_params = {'--reg_rate': 0.1},
                    compute_target='local'
                    )

# Create an experiment
experiment_name = 'diabetes-training'
experiment = Experiment(workspace = ws, name = experiment_name)

# Run the experiment
run = experiment.submit(config=estimator)

# Show the run details while running
RunDetails(run).show()
run.wait_for_completion()
```
### 2.3 Generate metrics from an experiment run
#### 2.3.1 log metrics from an experiment run
Every experiment generates log files that include the messages that would be written to the terminal during interactive execution. This enables you to use simple print statements to write messages to the log. However, if you want to record named metrics for comparison across runs, you can do so by using the Run object; which provides a range of logging functions specifically for this purpose. These include:

* **log**: Record a single named value.
Example experiment code:
```python
import pandas
data = pd.read_csv('data/diabetes.csv')
row_count = len(data)
run.log('observations', row_count)
```
* **log_list**: Record a named list of values.
```python
run.log_list('pregnancy categories', data.Pregnancies.unique())
```  
* **log_row**: Record a row with multiple columns.
```python
# Log summary statistics for numeric columns
med_columns = ['PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'SerumInsulin', 'BMI']
summary_stats = data[med_columns].describe().to_dict()
for col in summary_stats:
    keys = list(summary_stats[col].keys())
    values = list(summary_stats[col].values())
    for index in range(len(keys)):
        run.log_row(col, stat=keys[index], value = values[index])
```
* **log_table**: Record a dictionary as a table.
* **log_image**: Record an image file or a plot.
```python
diabetic_counts = data['Diabetic'].value_counts()
fig = plt.figure(figsize=(6,6))
ax = fig.gca()    
diabetic_counts.plot.bar(ax = ax) 
ax.set_title('Patients with Diabetes') 
ax.set_xlabel('Diagnosis') 
ax.set_ylabel('Patients')
plt.show()
run.log_image(name = 'label distribution', plot = fig)
```

More Information: For more information about logging metrics during experiment runs, see https://docs.microsoft.com/en-us/azure/machine-learning/how-to-track-experiments.

For example, following code records the number of observations (records) in a CSV file:
```python
from azureml.core import Experiment
import pandas as pd

# Create an Azure ML experiment in your workspace
experiment = Experiment(workspace = ws, name = 'my-experiment')

# Start logging data from the experiment
run = experiment.start_logging()

# load the dataset and count the rows
data = pd.read_csv('data.csv')
row_count = (len(data))

# Log the row count
run.log('observations', row_count)

# Complete the experiment
run.complete()
```

**Logging with ML Flow**
ML Flow is an Open Source library for managing machine learning experiments, and includes a tracking component for logging. If your organization already includes ML Flow, you can continue to use it to track metrics in Azure Machine Learning.

More Information: For more information about using ML Flow with Azure Machine Learning, see Track metrics and deploy models with MLflow and Azure Machine Learning in the documentation.

**Retrieving and Viewing Logged Metrics**
You can view the metrics logged by an experiment run in Azure Machine Learning studio or by using the RunDetails widget in a notebook, as shown here:
```python
from azureml.widgets import RunDetails

RunDetails(run).show()
```

You can also retrieve the metrics using the Run object's get_metrics method, which returns a JSON representation of the metrics, as shown here:
```python
import json

# Get logged metrics
metrics = run.get_metrics()
print(json.dumps(metrics, indent=2))
``` 

The previous code produces output similar to this:
```json
{
  "observations": 15000
}
```
#### 2.3.2 retrieve and view experiment outputs
In addition to logging metrics, an experiment can generate output files. Often these are trained machine learning models, but you can save any sort of file and make it available as an output of your experiment run. The output files of an experiment are saved in its outputs folder.

The technique you use to add files to the outputs of an experiment depend on how your running the experiment. The examples shown so far control the experiment lifecycle inline in your code, and when taking this approach you can upload local files to the run's outputs folder by using the Run object's upload_file method in your experiment code as shown here:
```python
run.upload_file(name='outputs/sample.csv', path_or_stream='./sample.csv')
``` 

When running an experiment in a remote compute context (which we'll discuss later in this course), any files written to the outputs folder in the compute context are automatically uploaded to the run's outputs folder when the run completes.

Whichever approach you use to run your experiment, you can retrieve a list of output files from the Run object like this:
```python
import json

files = run.get_file_names()
print(json.dumps(files, indent=2))
```

The previous code produces output similar to this:
```json
[
  "outputs/sample.csv"
]
```
#### 2.3.3 use logs to troubleshoot experiment run errors

#### --- Environments --- ####
What are Environments?
Python code runs in the context of a virtual environment that defines the version of the Python runtime to be used as well as the installed packages available to the code. In most Python installations, packages are installed and managed in environments using Conda or pip.

Environments in Azure Machine Learning
In general, Azure Machine Learning handles environment creation and package installation for you - usually through the creation of Docker containers. You can specify the Conda or pip packages you need, and have Azure Machine Learning create an environment for the experiment.

In an enterprise machine learning solution, where experiments may be run in a variety of compute contexts, it can be important to be aware of the environments in which your experiment code is running. Environments are encapsulated by the Environment class; which you can use to create environments and specify runtime configuration for an experiment.

You can have Azure Machine Learning manage environment creation and package installation to define an environment, and then register it for reuse. Alternatively, you can manage your own environments and register them. This makes it possible to define consistent, reusable runtime contexts for your experiments - regardless of where the experiment script is run.

**Creating an Environment from a Specification File**
You can use a Conda or pip specification file to define the packages required in a Python evironment, and use it to create an Environment object.

For example, you could save the following Conda configuration settings in a file named conda.yml:
```yaml
name: py_env
dependencies:
  - numpy
  - pandas
  - scikit-learn
  - pip:
    - azureml-defaults
```

The you could use the following code creates an Azure Machine Learning environment from the saved specification file:
```python
from azureml.core import Environment

env = Environment.from_conda_specification(name='training_environment',
                                           file_path='./conda.yml')
``` 

**Creating an Environment from an Existing Conda Environment**
If you have an existing Conda environment defined on your workstation, you can use it to define an Azure Machine Learning environment:
```python
from azureml.core import Environment

env = Environment.from_existing_conda_environment(name='training_environment',
                                                  conda_environment_name='py_env')
``` 

**Creating an Environment by Specifying Packages**
You can define an environment by specifying the Conda and pip packages you need in a CondaDependencies object, like this:
```python 
from azureml.core import Environment
from azureml.core.conda_dependencies import CondaDependencies

env = Environment('training_environment')

env.python.user_managed_dependencies = False # Let Azure ML manage dependencies, for custom docker images this has to be set to True
env.docker.enabled = True # Use a docker container

deps = CondaDependencies.create(conda_packages=['scikit-learn','pandas','numpy'],
                                pip_packages=['azureml-defaults'])
env.python.conda_dependencies = deps
``` 

After you've created an environment, you can register it in your workspace and reuse it for future experiments that have the same Python dependencies.

**Registering an Environment**
Use the register method of an Environment object to register an environment:
```python
env.register(workspace=ws)
``` 

You can view the registered environments in your workspace like this:
```python
from azureml.core import Environment

env_names = Environment.list(workspace=ws)
for env_name in env_names:
    print('Name:',env_name)
``` 

**Retrieving and using an Environment**
You can retrieve a registered environment by using the get method of the Environment class, and then assign it to a ScriptRunConfig or Estimator.
For example, the following code sample retrieves the training_environment registered environment, and assigns it to an estimator:
```python
from azureml.core import Environment
from azureml.train.estimator import Estimator

training_env = Environment.get(workspace=ws, name='training_environment')
estimator = Estimator(source_directory='experiment_folder'
                      entry_script='training_script.py',
                      compute_target='local',
                      environment_definition=training_env)
``` 

When an experiment based on the estimator is run, Azure Machine Learning will look for an existing environment that matches the definition, and if none is found a new environment will be created based on the registered environment specification. In the *azureml_logs/60_control_log.txt* you will see the conda environment being built.

**Curated Environments**
Azure Machine Learning includes a selection of pre-defined curated environments that reflect common usage scenarios. These include environments that are pre-configured with package dependencies for common frameworks, such as Scikit-Learn, PyTorch, Tensorflow, and others.

Curated environments are registered in all Azure Machine Learning workspaces with a name that begins AzureML-.

**> Note: You can't register your own environments with an “AzureML-” prefix. **

*Viewing Curated Environments*
To view curated environments and the dependencies they contain, you can run the following code:
```python
from azureml.core import Environment

envs = Environment.list(workspace=ws)
for env in envs:
    if env.startswith("AzureML"):
        print("Name",env)
        print("packages", envs[env].python.conda_dependencies.serialize_to_string())
``` 

### 2.4 Automate the model training process
**What is a Pipeline?**
In Azure Machine Learning, a pipeline is a workflow of machine learning tasks in which each task is implemented as a step.

**> Note:** The term pipeline is used extensively in machine learning, often with different meanings. For example, in Scikit-Learn, you can define pipelines that combine data preprocessing transformations with a training algorithm; and in Azure DevOps, you can define a build or release pipeline to perform the build and configuration tasks required to deliver software. The focus of this module is on Azure Machine Learning pipelines, which encapsulate steps that can be run as an experiment. However, bear in mind that it's perfectly feasible to have an Azure DevOps pipeline with a task that that initiates an Azure Machine Learning pipeline, which in turn includes a step that trains a model based on a Scikit-Learn pipeline!

Steps can be arranged sequentially or in parallel, enabling you to build sophisticated flow logic to orchestrate machine learning operations. Each step can be run on a specific compute target, making it possible to combine different types of processing as required to achieve an overall goal.

*Pipelines as Executable Processes*
A pipeline can be executed as a process by running the pipeline as an experiment. Each step in the pipeline runs on its allocated compute target as part of the overall experiment run.

You can publish a pipeline as a REST endpoint, enabling client applications to initiate a pipeline run. You can also define a schedule for a pipeline, and have it run automatically at periodic intervals.

*Pipelines and DevOps for Machine Learning*
As machine learning becomes increasingly ubiquitous in the enterprise, IT organizations are finding a need to integrate model training, management, and deployment into their standard development/operations (DevOps) practices through automation and policy-based release management. The implementation of a continuous integration/continuous delivery (CI/CD) solution for machine learning models is often referred to as “MLOps”, and pipelines are a core element of this.
#### 2.4.1 create a pipeline by using the SDK
**Pipeline Steps**
An Azure Machine Learning pipeline consists of one or more steps that perform tasks. There are many kinds of step supported by Azure Machine Learning pipelines, each with its own specialized purpose and configuration options.

*Types of Step*
Common kinds of step in an Azure Machine Learning pipeline include:

* PythonScriptStep: Runs a specified Python script.
* EstimatorStep: Runs an estimator.
* DataTransferStep: Uses Azure Data Factory to copy data between data stores.
* DatabricksStep: Runs a notebook, script, or compiled JAR on a databricks cluster.
* AdlaStep: Runs a U-SQL job in Azure Data Lake Analytics.

**Note:** For a full list of supported step types, see azure.pipeline.steps package documentation.

*Defining Steps in a Pipeline*
To create a pipeline, you must first define each step and then create a pipeline that includes the steps. The specific configuration of each step depends on the step type. For example the following code defines a PythonScriptStep step that runs a script, and an EstimatorStep step that runs an estimator.
```python
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep

# Step to run a Python script
step1 = PythonScriptStep(name = 'prepare data',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster',
                         runconfig = run_config)

# Step to run an estimator
step2 = EstimatorStep(name = 'train model',
                      estimator = sk_estimator,
                      compute_target = 'aml-cluster')
``` 

After defining the steps, you can assign them to a pipeline, and run it as an experiment:
```python
from azureml.pipeline.core import Pipeline
from azureml.core import Experiment

# Construct the pipeline
train_pipeline = Pipeline(workspace = ws, steps = [step1,step2])

# Create an experiment and run the pipeline
experiment = Experiment(workspace = ws, name = 'training-pipeline')
pipeline_run = experiment.submit(train_pipeline)
``` 

#### 2.4.2 pass data between steps in a pipeline
To use a PipelineData object to pass data between steps, you must:
* Define a named PipelineData object that references a location in a datastore.
* Specify the PipelineData object as an input or output for the steps that use it.
* Pass the PipelineData object as a script argument in steps that run scripts (and include code in those scripts to read or write data)

For example, the following code defines a PipelineData object that for the preprocessed data that must be passed between the steps.
```python
from azureml.pipeline.core import PipelineData
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep

# Get a dataset for the initial data
raw_ds = Dataset.get_by_name(ws, 'raw_dataset')

# Define a PipelineData object to pass data between steps
data_store = ws.get_default_datastore()
prepped_data = PipelineData('prepped',  datastore=data_store)

# Step to run a Python script
step1 = PythonScriptStep(name = 'prepare data',
                         source_directory = 'scripts',
                         script_name = 'data_prep.py',
                         compute_target = 'aml-cluster',
                         runconfig = run_config,
                         # Specify dataset as initial input
                         inputs=[raw_ds.as_named_input('raw_data')],
                         # Specify PipelineData as output
                         outputs=[prepped_data],
                         # Also pass as data reference to script
                         arguments = ['--folder', prepped_data])

# Step to run an estimator
step2 = EstimatorStep(name = 'train model',
                      estimator = sk_estimator,
                      compute_target = 'aml-cluster',
                      # Specify PipelineData as input
                      inputs=[prepped_data],
                      # Pass as data reference to estimator script
                      estimator_entry_script_arguments=['--folder', prepped_data])
 ``` 

By default the step output from a previous pipeline run is reused without re-running the step as long as the script, source directory, and other parameters for the step have not changed. To change that include `allow_reuse=False` as argument to the step.

In the scripts themselves, you can obtain a reference to the PipelineData object from the script argument, and use it like a local folder.
```python
# code in data_prep.py
from azureml.core import Run
import argparse
import os

# Get the experiment run context
run = Run.get_context()

# Get input dataset as dataframe
raw_df = run.input_datasets['raw_data'].to_pandas_dataframe()

# Get PipelineData argument
parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str, dest='folder')
args = parser.parse_args()
output_folder = args.folder

# code to prep data (in this case, just select specific columns)
prepped_df = raw_df[['col1', 'col2', 'col3']]

# Save prepped data to the PipelineData location
os.makedirs(output_folder, exist_ok=True)
output_path = os.path.join(output_folder, 'prepped_data.csv')
prepped_df.to_csv(output_path)
```

#### 2.4.3 run a pipeline
see also 2.4.1:
```python
# Construct the pipeline
train_pipeline = Pipeline(workspace = ws, steps = [step1,step2])

# Create an experiment and run the pipeline
experiment = Experiment(workspace = ws, name = 'training-pipeline')
pipeline_run = experiment.submit(train_pipeline)
``` 
If you want to force all steps to be re-run, modify last line like so:
```python
pipeline_run = experiment.submit(train_pipeline, regenerate_outputs=True)
```
#### 2.4.4 monitor pipeline runs

#### --- publishing pipelines --- ####
After you have created a pipeline, you can publish it to create a REST endpoint through which the pipeline can be run on demand.
To publish a pipeline, you can call its publish method, as shown here:
```python
published_pipeline = pipeline.publish(name='training_pipeline',
                                          description='Model training pipeline',
                                          version='1.0')
``` 

Alternatively, you can call the publish method on a successful run of the pipeline:
```python
# Get the most recent run of the pipeline
pipeline_experiment = ws.experiments.get('training-pipeline')
run = list(pipeline_experiment.get_runs())[0]

# Publish the pipeline from the run
published_pipeline = run.publish_pipeline(name='training_pipeline',
                                          description='Model training pipeline',
                                          version='1.0')

After the pipeline has been published, you can view it in Azure Machine Learning studio. You can also determine the URI of its endpoint like this:

rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)
``` 

**Using a Published Pipeline**
To initiate a published endpoint, you make an HTTP request to its REST endpoint, passing an authorization header with a token for a service principal with permission to run the pipeline, and a JSON payload specifying the experiment name. The pipeline is run asynchronously, so the response from a successful REST call includes the run ID. You can use this to track the run in Azure Machine Learning studio.

For example, the following Python code makes a REST request to run a pipeline and displays the returned run ID.
```python 
import requests
from azureml.core.authentication import InteractiveLoginAuthentication

rest_endpoint = published_pipeline.endpoint
interactive_auth = InteractiveLoginAuthentication()
auth_header = interactive_auth.get_authentication_header()
print("Authentication header ready.")

response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "run_training_pipeline"})
run_id = response.json()["Id"]
print(run_id)
``` 

**Defining Parameters for a Pipeline**
You can increase the flexibility of a pipeline by defining parameters. To define parameters for a pipeline, create a PipelineParameter object for each parameter, and specify each parameter in at least one step.

For example, you could use the following code to include a parameter for a regularization rate in the script used by an estimator:
```python
from azureml.pipeline.core.graph import PipelineParameter
reg_param = PipelineParameter(name='reg_rate', default_value=0.01)

...

step2 = EstimatorStep(name = 'train model',
                      estimator = sk_estimator,
                      compute_target = 'aml-cluster',
                      inputs=[prepped],
                      estimator_entry_script_arguments=['--folder', prepped,
                                                        '--reg', reg_param])
``` 

Note: You must define parameters for a pipeline before publishing it.

*Running a Pipeline with a Parameter*
After you publish a parameterized pipeline, you can pass parameter values in the JSON payload for the REST interface:
```python
response = requests.post(rest_endpoint,
                         headers=auth_header,
                         json={"ExperimentName": "run_training_pipeline",
                               "ParameterAssignments": {"reg_rate": 0.1}})
``` 

*Scheduling a Pipeline*
After you have published a pipeline, you can initiate it on demand through its REST endpoint, or you can have the pipeline run automatically based on a periodic schedule or in response to data updates. To schedule a pipeline to run at periodic intervals, you must define a ScheduleRecurrance that determines the run frequency, and use it to create a Schedule.
For example, the following code schedules a daily run of a published pipeline:
```python
from azureml.pipeline.core import ScheduleRecurrence, Schedule

daily = ScheduleRecurrence(frequency='Day', interval=1)
pipeline_schedule = Schedule.create(ws, name='Daily Training',
                                        description='trains model every day',
                                        pipeline_id=published_pipeline.id,
                                        experiment_name='Training_Pipeline',
                                        recurrence=daily)
``` 
*Triggering a Pipeline Run on Data Changes*
To schedule a pipeline to run whenever data changes, you must create a Schedule that monitors a specified path on a datastore, like this:
```python
from azureml.core import Datastore
from azureml.pipeline.core import Schedule

training_datastore = Datastore(workspace=ws, name='blob_data')
pipeline_schedule = Schedule.create(ws, name='Reactive Training',
                                    description='trains model on data change',
                                    pipeline_id=published_pipeline_id,
                                    experiment_name='Training_Pipeline',
                                    datastore=training_datastore,
                                    path_on_datastore='data/training')
 ```

## 3 Optimize and Manage Models (20-25%)
### 3.1 Use Automated ML to create optimal models
#### 3.1.1 use the Automated ML interface in Azure Machine Learning studio
#### 3.1.2 use Automated ML from the Azure Machine Learning SDK
#### 3.1.3 select scaling functions and pre-processing options
#### 3.1.4 determine algorithms to be searched
#### 3.1.5 define a primary metric
#### 3.1.6 get data for an Automated ML run
#### 3.1.7 retrieve the best model
### 3.2 Use Hyperdrive to tune hyperparameters
#### 3.2.1 select a sampling method
#### 3.2.2 define the search space
#### 3.2.3 define the primary metric
#### 3.2.4 define early termination options
#### 3.2.5 find the model that has optimal hyperparameter values
### 3.3 Use model explainers to interpret models
#### 3.3.1 select a model interpreter
#### 3.3.2 generate feature importance data
### 3.4 Manage models
#### 3.4.1 register a trained model
```python
from azureml.core import Model

# Register the model
run.register_model(model_path='outputs/diabetes_model.pkl', 
                   model_name='diabetes_model',
                   tags={'Training context':'Parameterized SKLearn Estimator'},
                   properties={'AUC': run.get_metrics()['AUC'], 
                               'Accuracy': run.get_metrics()['Accuracy']})

# List registered models
for model in Model.list(ws):
    print(model.name, 'version:', model.version)
    for tag_name in model.tags:
        tag = model.tags[tag_name]
        print ('\t',tag_name, ':', tag)
    for prop_name in model.properties:
        prop = model.properties[prop_name]
        print ('\t',prop_name, ':', prop)
    print('\n')
```
#### 3.4.2 monitor model history
#### 3.4.3 monitor data drift 

## 4 Deploy and Consume Models (20-25%)
### 4.1 Create production compute targets
#### 4.1.1 consider security for deployed services
#### 4.1.2 evaluate compute options for deployment
### 4.2 Deploy a model as a service
#### 4.2.1 configure deployment settings
#### 4.2.2 consume a deployed service
#### 4.2.3 troubleshoot deployment container issues
### 4.3 Create a pipeline for batch inferencing
#### 4.3.1 publish a batch inferencing pipeline
#### 4.3.2 run a batch inferencing pipeline and obtain outputs
### 4.4 Publish a designer pipeline as a web service
#### 4.4.1 create a target compute resource
#### 4.4.2 configure an Inference pipeline
#### 4.4.3 consume a deployed endpoint
