from diagrams import Diagram, Cluster
from diagrams.aws.compute import Lambda
from diagrams.aws.storage import S3
from diagrams.aws.analytics import EMR
from diagrams.aws.database import RDS
from diagrams.aws.ml import Sagemaker
from diagrams.aws.analytics import DataPipeline
from diagrams.aws.enduser import Workdocs
from diagrams.aws.security import Inspector
from diagrams.aws.database import Redshift
from diagrams.aws.analytics import Glue

with Diagram("MRI Processing & Analysis Pipeline", show=False, direction="LR"):

    # 1. File Management
    with Cluster("Data Sourcing"):
        find_files = Redshift("Import MRI Data")
        find_files - Redshift("Import Harvard Atlas")

    # 2-4. Preprocessing
    with Cluster("Preprocess MRI Images"):
        n4_correction = DataPipeline("N4 Bias Field Correction")
        n4_correction - Glue("Noise Smoothing")
        n4_correction - DataPipeline("Register to Atlas")
        
    # 5-5.5. Region & Volume Analysis
    with Cluster("Region & Volume Analysis"):
        region_isolation = Sagemaker("Brain Region Isolation")
        region_isolation - [Sagemaker("Volume Calculation"),
                            Workdocs("Export as Excel")]

    # 6. Statistical Analysis
    with Cluster("Statistical Analysis"):
        stat_analysis = Inspector("Statistical Analysis")

    # Flow Connections
    find_files >> n4_correction
    n4_correction >> region_isolation
    region_isolation >> stat_analysis