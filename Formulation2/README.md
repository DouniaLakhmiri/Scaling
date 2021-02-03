# Running Formulation 2


![Formulation2](Formulation2.png)


## Inside the blackbox

### Inputs:

* d: depth factor
* w: width factor
* r: resolution factor

### Outputs:

* Objective function: Number of MACS of the current network
* Constraint value: (Validation accuracy of ResNet18) - (Validation accuracy of current network) < 0 


### Testing one blackbox evaluation

To test one signe blackbox evaluation, we first need to choose values for d, w, and r. Knowing that the triplet (1, 1, 1) 
is equivalent to running ResNet18() with the default image resolution of the database.

A blackbox call can be run with the following command:

```
$ python blackbox2.py d w r
```

## Defining the HPO 

The file parameter_file.txt allows to define the Hyperparameter Optimization problem: 

```
DIMENSION               3                               # NOMAD optimizes 3 hyperparameters
BB_EXE                  "$python ./nomad_linker2.py"    # The script that links NOMAD to this blackbox

BB_OUTPUT_TYPE          OBJ   EB  -                     # The blackbox returns 3 outputs: 
                                                        # OBJ : number of MACS
                                                        # EB : Extreme Barrier for the constraint:
                                                        #     (Validation accuracy of ResNet18) - (Validation accuracy of current network) < 0 
                                                        #  - : Number of parameters
                                                        
BB_INPUT_TYPE           ( R  R  R )                     # d, w and r and Real hyperparameters

X0                      (  1  1  1)                     # NOMAD needs an initial starting point       

LOWER_BOUND             ( 0.5 0.5 0.5 )                 # Lower bound on d, w and r
UPPER_BOUND             ( 2.5  2.5  2.5 )               # Upper bound on d, w and r

MAX_BB_EVAL             200                             # Number of blackbox evaluations
DISPLAY_DEGREE          3                               # Display all the logs of NOMAD
```



## Running an optimization

To run the NOMAD optimization, we need to execute: 

```
$ nomad parameter_file2.txt
```


