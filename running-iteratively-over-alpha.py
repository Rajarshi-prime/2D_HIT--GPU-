import json
import os,sys,time

# List of different alpha values to run
# alpha_values = [0.70,0.75,0.77,0.8,0.83,0.85,0.9,0.95]
alpha_values = [0.70,0.72,0.8,0.95,1.0]

# Load the parameters from the JSON file
with open('parameters.json', 'r') as file:
    parameters = json.load(file)

for alpha in alpha_values:
    # Modify the alpha value
    parameters['alph'] = alpha
    
    # Save the modified parameters to the JSON file
    with open('parameters.json', 'w') as file:
        json.dump(parameters, file, indent=4)
    print(f'Running with alpha={alpha}')
    # Run the Python script
    os.system(f'nohup python -u test.py > postproc_{alpha*100:.0f}.out &')
    print(f"saved in postproc_{alpha*100:.0f}.out")
    # os.system(f'nohup python -u 2DV_spline.py > 1024_spline_{alpha*100:.0f}.out &')
    # print(f"saved in 1024_spline_{alpha*100:.0f}.out")
    # Wait for n seconds before running the next iteration
    time.sleep(5) 