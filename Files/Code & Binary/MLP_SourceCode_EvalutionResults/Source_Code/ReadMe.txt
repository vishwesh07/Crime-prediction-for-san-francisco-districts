Programs are written in object oriented fashion with functions for each phase of data science life cycle as one function.

This folder contains two files:
	1) ipython notebook
	2) python file

Libraries required:
	python 3.X
	sklearn
	numpy
	scipy
	matplotlib
	sodapy

To run problem on test data(in ipython notebook):
	- Fetch data from API via uncommenting first 7 lines and then follow below listed steps.
	- Execute all the cells up to sign that tells 'Program Strating Point for Data Science Life Cycle Phases'. 
	- Uncomment 'break' line:
		accuracy_dictionary = dict()

		for key in district_data:
    			district = district_data[key]
    			process_district(district, key)
			#     break 		
	- Execute functions in following order:
		preprocess_district()
		convert_date_to_day()
		convert_date_to_month()
		convert_date_to_year()
		convert_date_to_hour()
		dataframetoCSRmatrix()
		reduce_dimentions_CSR_matrix()
		feature_engineer_district()
		report()
		train_test_district_model()
		process_district() - with dictionary of district as parameter
	- Uncommenting break line will only execute pipeline for one district - 'BAYVIEW'. 

To run whole program(in ipython notebook):
	- Fetch data from API via uncommenting first 7 lines and then follow below listed steps.
	- Execute all the cells up to sign that tells 'Program Strating Point for Data Science Life Cycle Phases' 
	- Execute functions in following order:
		preprocess_district()
		convert_date_to_day()
		convert_date_to_month()
		convert_date_to_year()
		convert_date_to_hour()
		dataframetoCSRmatrix()
		reduce_dimentions_CSR_matrix()
		feature_engineer_district()
		report()
		train_test_district_model()
		process_district() - with dictionary of district as parameter
		process_district() - with dictionary of city as parameter