All the solvers are run in the same way by executing python main.py, which by default runs the 40 instances present in the folder res/instances/ one after another using the solver stated by the argument --solver. The instances can be run with the following arguments:

    -P, --plot Plots the solution after its computation
    -X, --stats Disables automatic plot display of the instances times at the end of execution
    -A, --average Solves each instance 5 times and computes an average of the tries
    -T TIMEOUT, --timeout TIMEOUT Timeout for the execution of the solvers in seconds (default: 300)
    -R, --rotation Enables rotation mode (disabled by default)
    -S {null,cp,sat,mip}, --solver {null,cp,sat,mip} Select solver used for execution
    -I INSTANCES, --instances INSTANCES Instances to be solved. Can be a directory, a file or all, which searches by default for the files from ins-0.txt to ins-40.txt in the path res/instances (default: all)
    -O OUTPUT, --output OUTPUT Output directory of the files containing the solutions
