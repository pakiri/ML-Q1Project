# ML-Q1Project
Steps to Reproduce Our Most Effective Model: SVC with Correlation Attribute Selection
1.   Download the mushrooms.csv from the [official UCI repository](https://archive.ics.uci.edu/dataset/73/mushroom)
2.   Place the mushrooms.csv file in a new directory named “Q1 Project”
3.   Open Weka and load the mushrooms.csv file
4.   Go to the “Select Attributes” tab and ensure the class variable is set to class
5.   Select CorrelationAttributeEval as Attribute Evaluator and Ranker as Search Method
6.   Begin the attribute selection and take inventory of features in the output that have a ratio not 
      greater than or equal to 0.25 (our cutoff)
7.   Open a Python IDE (Jetbrains DataSpell was used for this paper) with CWD as “Q1 Project”.
	If the Python environment is not in the “Q1 Project” directory, type
			    `cd [Path of Q1 Project directory]`
9.   Copy the split.py, model.py, and load.py programs from the paper’s source to the directory
10.   Install required dependencies using the shell terminal:
`pip install numpy pandas scikit-learn`
11. Open the split.py program in the IDE
12. Configure the `attributes_to_remove` list in the split.py program to contain the names
      of the features in the earlier inventory as strings
13. Change the contents of the `folder_name` string to “correlation”
14. Run the split.py program to remove irrelevant attributes, encode data, perform KNN
      imputation, and test-train-split into separate directories
14. Open the model.py program in the IDE
15. Run the model.py program to derive metrics for Correlation Attribute Selection with SVC  
      and save the model
16. Model can be found here: **Q1 Project/models/svc_correlation.pkl**
17. Verify the stored model as functional by running load.py and observing prediction accuracy
