import json, os,sys,time
import numpy as np

# List of different alpha values to run
alpha_values = np.array([0.70,0.72,0.75,0.77,0.8,0.9,0.95,1.0])
# alpha_values = [0.7,1.0]

# Load the parameters from the JSON file
with open('parameters.json', 'r') as file:
    parameters = json.load(file)

for ii,alpha in enumerate(alpha_values[::2]):
    # Modify the alpha value
    parameters['alph'] = alpha
    
    # Save the modified parameters to the JSON file
    with open('parameters.json', 'w') as file:
        json.dump(parameters, file, indent=4)
    print(f'Running with alpha={alpha}')
    # Run the Python script
    output_file = f'postproc_{alpha*100:.0f}.out'
    # output_file = f'1024_spline_{alpha*100:.0f}.out'
    os.system(f'nohup time python -u postproc.py > {output_file} &')
    # os.system(f'nohup python -u 2DV_spline.py > {output_file} &')
    print(f"saved in {output_file}")
    time.sleep(2)
    # Wait for n seconds before running the next iteration