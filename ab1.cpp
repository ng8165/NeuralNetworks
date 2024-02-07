/**
 * A-B-1 Network
 * Author: Nelson Gou
 * Creation Date: 1/30/23
 * Functional Description: This is an A-B-1 multilayer perceptron network that uses gradient
 * descent learning. The network has two execution modes, running and training.
 * Other configuration parameters can be modified in the setConfig method.
 */

#include <iostream>
#include <cmath>
#include <cfloat>

using namespace std;

/**
 * These enums allow for increased readibility for certain configuration parameters.
 */
enum PopulationMode
{
   RANDOM,  // denotes random weight population
   CUSTOM   // denotes custom weight population
};

enum ExecutionMode
{
   TRAIN,   // trains the network
   RUN      // runs the network
};

/**
 * These variables are the configuration parameters.
 */
int numInputs;                // the number of input layers
int numHidden;                // the number of hidden layers
double lambda;                // learning rate
int maxIterations;            // maximum iteration limit
double errorThreshold;        // average error threshold
PopulationMode populateMode;  // weight population mode (either RANDOM or CUSTOM)
double randomMin;             // random number generation minimum bound
double randomMax;             // random number generation maximum bound
bool printTruthTables;        // print truth tables
ExecutionMode executionMode;  // execution mode (either TRAIN or RUN)

/**
 * These variables represent the test cases, the truth tables, and the stored results.
 */
const int TEST_CASES = 4;  // the number of test cases to run
double** inputs;           // the inputs to the truth table
double* truth;             // the truth table
double* results;           // the results of running each test case

/**
 * a, h, and F0 represent the activation layers. T0 represents the truth.
 */
double* a;  // the a layer is the input layer
double* h;  // the h layer is the hidden layer
double F0;  // F0 is the output node (since there is only one output node in the output layer)
double T0;  // T0 is the truth for the test case

/**
 * W_kj and W_j0 represent the weight arrays for the kj and j0 layers, respectively.
 */
double** W_kj; // the weights for the k-j layer
double* W_j0;  // the weights for the j-0 layer

/**
 * Theta_j and Theta_0 represent intermediate arrays when running the network.
 */
double* Theta_j;  // the Theta for the j layer
double Theta_0;   // the Theta for the i layer

/**
 * omega_0, psi_0, dE_dWj0, and deltaW_j0 represent intermediates when training the ji weights.
 */
double omega_0;      // the omega for the j-0 layer
double psi_0;        // the psi for the j-0 layer
double* dE_dWj0;     // the dE/dW for the j-0 layer
double* deltaW_j0;   // the deltaW for the j-0 layer

/**
 * Omega_j, Psi_j, dE_dWkj, and deltaW_kj represent intermediates when training the kj weights.
 */
double* Omega_j;     // the Omega for the k-j layer
double* Psi_j;       // the Psi for the k-j layer
double** dE_dWkj;    // the dE/dW for the k-j layer
double** deltaW_kj;  // the deltaW for the k-j layer

/**
 * These are used to track training status.
 */
int iterations;         // the number of iterations already trained
double averageError;    // the average error of each iteration

/**
 * f is the activation function (currently set to sigmoid).
 */
double f(double x)
{
   return 1.0 / (1.0 + exp(-x));
}

/**
 * fPrime, or f'(x), is the derivative of the activation function f(x).
 */
double fPrime(double x)
{
   double fOfX = f(x);
   return fOfX * (1.0 - fOfX);
}

/**
 * setConfig sets the configuration parameters for the network.
 * Parameters include the network configuration (how many input layers and how many hidden layers), lambda (learning
 * rate), maximum number of iterations, the error threshold, the mode for population of weights, random number
 * minimum and maximum boundaries, whether truth tables should be printed, and the execution mode of the network.
 */
void setConfig()
{
   numInputs = 2;             // the number of input layers
   numHidden = 1;             // the number of hidden layers
   lambda = 0.3;              // learning rate
   maxIterations = 100000;    // maximum iteration limit
   errorThreshold = 0.0002;   // average error threshold
   populateMode = RANDOM;     // weight population mode (either RANDOM or CUSTOM)
   randomMin = -1.5;          // random number generation minimum bound
   randomMax = 1.5;           // random number generation maximum bound
   printTruthTables = true;   // print truth tables
   executionMode = RUN;     // execution mode (either TRAIN or RUN)

   return;
} // void setConfig()

/**
 * echoConfig prints out the configuration parameters for the network.
 * This is used as a sanity check to ensure all parameters are as expected.
 */
void echoConfig()
{
   cout << "\nCONFIGURATION PARAMETERS:\n";

   cout << "Network Configuration: " << numInputs << "-" << numHidden << "-1" << endl;
   cout << "Execution Mode: " << (executionMode == TRAIN ? "training" : "running") << endl;
   cout << "Weight Population Mode: " << (populateMode == RANDOM ? "random" : "custom") << endl;
   cout << "Print Truth Tables: " << (printTruthTables ? "enabled" : "disabled") << endl;

   if (populateMode == RANDOM) // only print the random number bounds if populateMode is RANDOM
         cout << "Random Number Bounds: [" << randomMin << ", " << randomMax << "]\n";

   if (executionMode == TRAIN) // only print training-related parameters if training mode is selected
   {
      cout << "Lambda: " << lambda << endl;
      cout << "Maximum Iterations: " << maxIterations << endl;
      cout << "Average Error Threshold: " << errorThreshold << endl;
   }

   return;
} // void echoConfig()

/**
 * allocateMemory allocates memory for the network arrays (weights, activations, deltas, omegas, psis,
 * dE/dW, truth tables, inputs, results, etc.).
 */
void allocateMemory()
{
/**
 * Allocate memory for input arrays, truth table, and results array.
 */
   inputs = new double*[TEST_CASES];
   for (int test = 0; test < TEST_CASES; test++)
      inputs[test] = new double[numInputs];
   
   truth = new double[TEST_CASES];

   results = new double[TEST_CASES];

/**
 * Allocate memory for the activation nodes.
 */
   a = new double[numInputs];
   h = new double[numHidden];

/**
 * Allocate memory for the weights for both the k-j and j-0 layer.
 */
   W_j0 = new double[numHidden];

   W_kj = new double*[numInputs];
   for (int k = 0; k < numInputs; k++)
      W_kj[k] = new double[numHidden];

   if (executionMode == TRAIN)
   {
/**
 * Allocate memory for the dE/dW and the delta of weights for both the k-j and j-0 layer.
*/
      dE_dWj0 = new double[numHidden];

      deltaW_j0 = new double[numHidden];

      dE_dWkj = new double*[numInputs];
      for (int k = 0; k < numInputs; k++)
         dE_dWkj[k] = new double[numHidden];

      deltaW_kj = new double*[numInputs];
      for (int k = 0; k < numInputs; k++)
         deltaW_kj[k] = new double[numHidden];

/**
 * Allocate memory for the Theta, Omega, and Psi arrays.
 */
      Theta_j = new double[numHidden];

      Omega_j = new double[numHidden];

      Psi_j = new double[numHidden];
   } // if (executionMode == TRAIN)

   return;
} // void allocateMemory()

/**
 * deallocateMemory performs garbage-collection by deallocating memory for the network arrays.
 * Deletes all arrays dynamically allocated in the allocateMemory function.
 */
void deallocateMemory()
{
/**
 * Allocate memory for input arrays, truth table, and results array.
 */
   for (int test = 0; test < TEST_CASES; test++)
      delete[] inputs[test];
   
   delete[] inputs;

   delete[] truth;

   delete[] results;

/**
 * Deallocate memory for the activation nodes.
 */
   delete[] a;
   delete[] h;

/**
 * Deallocate memory for the weights for both the k-j and j-0 layer.
 */
   for (int k = 0; k < numInputs; k++)
      delete[] W_kj[k];

   delete[] W_kj;

   delete[] W_j0;

   if (executionMode == TRAIN)
   {
/**
 * Deallocate memory for the dE/dW and the delta of weights for both the k-j and j-0 layer.
*/
      for (int k = 0; k < numInputs; k++)
         delete[] dE_dWkj[k];
      
      delete[] dE_dWkj;
      
      for (int k = 0; k < numInputs; k++)
         delete[] deltaW_kj[k];
      
      delete[] deltaW_kj;

      delete[] dE_dWj0;

      delete[] deltaW_j0;
      
/**
 * Deallocate memory for the Theta, Omega, and Psi arrays.
 */
      delete[] Theta_j;

      delete[] Omega_j;

      delete[] Psi_j;
   } // if (executionMode == TRAIN)

   return;
} // void deallocateMemory()

/**
 * randomize returns a random double between the range of randomMin and randomMax, inclusive.
 */
double randomize()
{
   return ((double) rand() / (double) RAND_MAX) * (randomMax - randomMin) + randomMin;
}

/**
 * populateArrays populates the truth table and test case inputs (hardcoded), as well as the weights.
 * There are two population modes for weights.
 * If populateMode is RANDOM, all weights are initialized using the randomize() function.
 * If populateMode is CUSTOM, the weights are manually set in the function to anything of the user's choice.
 */
void populateArrays()
{
/**
 * Populate the truth table (inputs and truths).
 */
   inputs[0][0] = 0.0;
   inputs[0][1] = 0.0;

   inputs[1][0] = 0.0;
   inputs[1][1] = 1.0;

   inputs[2][0] = 1.0;
   inputs[2][1] = 0.0;

   inputs[3][0] = 1.0;
   inputs[3][1] = 1.0;

   truth[0] = 0.0;
   truth[1] = 1.0;
   truth[2] = 1.0;
   truth[3] = 0.0;

/**
 * Populate the weights (options: random and custom).
 */
   if (populateMode == RANDOM)
   {
      srand(time(NULL)); // seeds the random number generator
      rand();            // needed to return random numbers correctly

      for (int k = 0; k < numInputs; k++)
         for (int j = 0; j < numHidden; j++)
            W_kj[k][j] = randomize();

      for (int j = 0; j < numHidden; j++)
         W_j0[j] = randomize();
   } // if (populateMode == RANDOM)
   else // custom weight population
   {
      W_kj[0][0] = 0.4;
      W_kj[0][1] = 0.3;

      W_kj[1][0] = 0.3;
      W_kj[1][1] = 0.4;

      W_j0[0] = 0.5;
      W_j0[1] = 0.5;
   } // if (populateMode == RANDOM) ... else

   return;
} // void populateArrays()

/**
 * runNetworkForTraining runs the network by computing the hidden layers based on the input layer and the k-j weights,
 * and the output node (F0) based on the hidden layers and the j-0 weights.
 * This version of runNetwork is only used during training, as it stores Theta_j values in
 * a global array that is only allocated during training.
 * This function assumes that the input layer a has already been set.
 */
void runNetworkForTraining()
{
/**
 * Compute the hidden layer activations (h).
 */
   for (int j = 0; j < numHidden; j++)
   {
      Theta_j[j] = 0.0;

      for (int K = 0; K < numInputs; K++)
         Theta_j[j] += a[K] * W_kj[K][j];
      
      h[j] = f(Theta_j[j]);
   } // for (int j = 0; j < numHidden; j++)

/**
 * Compute the output layer activations (F0).
 */
   Theta_0 = 0.0;

   for (int J = 0; J < numHidden; J++)
      Theta_0 += h[J] * W_j0[J];
   
   F0 = f(Theta_0);

   return;
} // void runNetworkForTraining()

/**
 * runNetwork runs the network by computing the hidden layers based on the input layer and the k-j weights,
 * and the output node (F0) based on the hidden layers and the j-0 weights.
 * This version of runNetwork is only used when running (not training), as it creates temporary Theta
 * variables to use as accumulators. This function assumes that the input layer a has already been set.
 */
void runNetwork()
{
/**
 * Compute the hidden layer activations (h).
 */
   for (int j = 0; j < numHidden; j++)
   {
      double Theta_j = 0.0;

      for (int K = 0; K < numInputs; K++)
         Theta_j += a[K] * W_kj[K][j];
      
      h[j] = f(Theta_j);
   } // for (int j = 0; j < numHidden; j++)

/**
 * Compute the output layer activations (F0).
 */
   Theta_0 = 0.0;

   for (int J = 0; J < numHidden; J++)
      Theta_0 += h[J] * W_j0[J];
   
   F0 = f(Theta_0);

   return;
} // void runNetwork()


/**
 * calculateDeltaWeights computes the delta weights for each layer (the k-j layer and the j-0 layer).
 * This function uses gradient descent to train the network.
 * It calculates omega, psi, and dE/dW to compute deltaW.
 */
void calculateDeltaWeights()
{
/**
 * Calculate the delta for the j-0 weights.
 */
   omega_0 = T0 - F0;
   psi_0 = omega_0 * fPrime(Theta_0);
   
   for (int j = 0; j < numHidden; j++)
   {
      dE_dWj0[j] = -h[j] * psi_0;
      deltaW_j0[j] = -lambda * dE_dWj0[j];
   } // for (int j = 0; j < numHidden; j++)

/**
 * Calculate the delta for the k-j weights.
 */
   for (int j = 0; j < numHidden; j++)
   {
      Omega_j[j] = psi_0 * W_j0[j];
      Psi_j[j] = Omega_j[j] * fPrime(Theta_j[j]);

      for (int k = 0; k < numInputs; k++)
      {
         dE_dWkj[k][j] = -a[k] * Psi_j[j];
         deltaW_kj[k][j] = -lambda * dE_dWkj[k][j];
      } // for (int k = 0; k < numInputs; k++)
   } // for (int j = 0; j < numHidden; j++)

   return;
} // void calculateDeltaWeights()

/**
 * applyDeltaWeights applies the delta weights to each layer (the k-j layer and the j-0 layer).
 * This function must be called directly after the calculateDeltaWeights function.
 * This will also eventually be absorbed into the calculateDeltaWeights function.
 */
void applyDeltaWeights()
{
/**
 * Update the k-j weights.
 */
   for (int k = 0; k < numInputs; k++)
      for (int j = 0; j < numHidden; j++)
         W_kj[k][j] += deltaW_kj[k][j];

/**
 * Update the j-0 weights.
 */
   for (int j = 0; j < numHidden; j++)
      W_j0[j] += deltaW_j0[j];
   
   return;
} // void applyDeltaWeights()

/**
 * For each iteration, the network loops over each test case and uses
 * gradient descent to update the weights on the k-j and j-0 layers. The math can be found in
 * Design Document 1 (Minimization of the Error Function for a Single Output and One Hidden Layer).
 * Training stops either when the maximum iteration limit is reached or when the average error
 * goes under the error threshold.
 */
void trainNetwork()
{
   iterations = 0;
   averageError = DBL_MAX; // initialize averageError to be larger than the error threshold
   
   while (iterations < maxIterations && averageError > errorThreshold)
   {
      double totalError = 0.0;

      for (int test = 0; test < TEST_CASES; test++)
      {
         a[0] = inputs[test][0];
         a[1] = inputs[test][1];
         T0 = truth[test];

         runNetworkForTraining();
         totalError += 0.5 * (T0 - F0) * (T0 - F0);

         calculateDeltaWeights();

         applyDeltaWeights();
      } // for (int test = 0; test < TEST_CASES; test++)

      iterations++;
      averageError = totalError / (double) TEST_CASES;
   } // while (iterations < maxIterations && averageError > errorThreshold)

   return;
} // void trainNetwork()

/**
 * Uses the executionMode to either train the network by calling the trainNetwork() function
 * or run the network by calling runNetwork() for each test case.
 * When running the network, average error is also calculated.
 */
void trainOrRun()
{
   if (executionMode == TRAIN)
      trainNetwork();
   
/**
 * The network is always run, even after training. This is to ensure that the results
 * array is correct so that reportResults will function correctly.
 */
   double totalError = 0.0; // will be used to find averageError later

   for (int test = 0; test < TEST_CASES; test++)
   {
      a[0] = inputs[test][0];
      a[1] = inputs[test][1];
      T0 = truth[test];

      runNetwork();

      results[test] = F0; // store F0 in results to report later
      totalError += 0.5 * (T0-F0) * (T0-F0);
   } // for (int test = 0; test < TEST_CASES; test++)

   averageError = totalError / (double) TEST_CASES;

   return;
} // void trainOrRun()

/**
 * Prints results from the network's execution.
 * If the network was trained, explains the reason why training was stopped and prints the number
 * of iterations and average error.
 * If the network was run, prints the average error.
 * Only prints truth tables if specified in the configuration parameters.
 */
void reportResults()
{
   cout << "\nRESULTS:\n";

   if (executionMode == TRAIN)
   {
      cout << "The network trained on " << TEST_CASES << " test cases.\n";
      cout << "Training stopped because ";

      if (averageError <= errorThreshold)
         cout << "average error was less than the error threshold (" << errorThreshold << ")";
      else
         cout << "the network reached the maximum iteration limit of " << maxIterations << " iterations";
      cout << ".\n\n";

      cout << "Iterations: " << iterations << endl;
   } // if (executionMode == TRAIN)
   else // executionMode was RUN
   {
      cout << "The network ran " << TEST_CASES << " test cases.\n";
   }

   cout << "Average Error: " << averageError << "\n\n";

   if (printTruthTables)
   {
      cout << "Case\t| a[0]\t| a[1]\t| F0\t\t| T0\t| Error\n";
      cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

      for (int test = 0; test < TEST_CASES; test++)
      {
         cout << test + 1 << "\t| "; // print a 1-indexed test case number for readability

         cout << inputs[test][0] << "\t| " << inputs[test][1] << "\t| ";
         printf("%.8f\t| ", results[test]); // prints F0 with 8 decimal places
         cout << truth[test] << "\t| ";

         omega_0 = truth[test] - results[test];
         printf("%.8f\n", 0.5 * omega_0 * omega_0); // prints error with 8 decimal places
      } // for (int test = 0; test < TEST_CASES; test++)

      cout << endl;
   } // if (printTruthTables)

   return;
} // void reportResults()

/**
 * The main function sets and echoes the configuration parameters.
 * It then allocates memory and populates the weight arrays based on the configuration parameters.
 * It then either runs or trains (again based on configuration parameters).
 * Then, results are reported (also based on configuration parameters).
 * Finally, memory management is performed as large arrays are garbage-collected and deleted.
 */
int main()
{
   setConfig();
   echoConfig();

   allocateMemory();
   populateArrays();

   trainOrRun();

   reportResults();

   deallocateMemory();

   return 0;
} // int main()