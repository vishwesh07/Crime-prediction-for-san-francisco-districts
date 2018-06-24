Line not needed for teacher testing:To run code on full dataset first run the CSVExports.py file which generates CSV files for each district with police incident data from SFGov site. (sodapy must be installed).

I have provided a subsample of the data in the CSVs folder where only 10000 samples are extracted from site.

IMPORTANT 2 LINES BELOW TO CHANGE CODE FOR YOUR SYSTEM:
n_threads = 36 line at start of each source file must also be changed to however many physical cores available on your environment..
Before running district files must change directory where CSV is taken from to match your own directory(e.g. mine is '/home/ubuntu/CSVs/BAYVIEW_data.csv').

If you are only testing my code I suggest you run on only Bayview district as it has all my testing/modelling, but if you want to see all district results change filepath on each to match sample
CSVs and execute them individually.

CSV files are in named as follows after CSV export files are run:
BAYVIEW_data.csv
CENTRAL_data.csv
CITY_data.csv
INGLESIDE_data.csv
MISSION_data.csv
NORTHERN_data.csv
PARK_data.csv
RICHMOND_data.csv
SOUTHERN_data.csv
TARAVAL_data.csv
TENDERLOIN_data.csv



After that each district/the whole city can be run individually(long computation times so files separated) to generate
f1 scores on .dat files that are exported to the same folder along with the district/city name.  District/city Files include:

Bayview XGB.py (Most important, contains all model comparison/testing done!)
Central.py
ingleside.py
Mission.py
Northern.py
Park.py
Richmond.py
Southern.py
Taraval.py
Tenderloin.py
Whole City.py

BAYVIEW XGB HAS ALL TESTING/MODEL COMPARISON, MOST IMPORTANT FILE!!!

Finally, the Comp_Districts_City.py file can be run to compare and bar plot each district(after each district has been run and respective .dat file generated)  


All my work was done in Notebook folder on EC2 server on a jupyter notebook so they're in ipynb format also where executions can already be seen
