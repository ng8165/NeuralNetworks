/**
 * A-B-1 Network
 * Author: Nelson Gou
 * Creation Date: 1/30/23
 * Functional Description: This is an A-B-1 multilayer perceptron network that uses gradient
 * descent learning. The network has two execution modes, running and training.
 * Other configuration parameters can be modified in the setConfig method.
 */

#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

/**
 * These enums allow for increased readibility for certain configuration parameters.
 */
enum populateMode { RANDOM, CUSTOM };
enum executionMode { TRAIN, RUN };

/**
 * These variables are the configuration parameters.
 */
int NUM_INPUTS, NUM_HIDDEN;
double LAMBDA;
int MAX_ITERATIONS;
double ERROR_THRESHOLD;
populateMode POPULATE_MODE;
double RANDOM_MIN, RANDOM_MAX;
bool PRINT_TRUTH_TABLES;
executionMode EXECUTION_MODE;

/**
 * These variables represent the test cases, the truth tables, and the stored results.
 */
const int TEST_CASES = 4;
double** INPUTS;
double* TRUTH;
double* results;

/**
 * a, h, and F0 represent the activation layers. T0 represents the truth.
 */
double* a;
double* h;
double F0;
double T0;

/**
 * W_kj and W_j0 represent the weight arrays for the kj and j0 layers, respectively.
 */
double** W_kj;
double* W_j0;

/**
 * Theta_j and Theta_0 represent intermediate arrays when running the network.
 */
double* Theta_j;
double Theta_0;

/**
 * omega_0, psi_0, dE_dWj0, and deltaW_j0 represent intermediates when training the ji weights.
 */
double omega_0;
double psi_0;
double* dE_dWj0;
double* deltaW_j0;

/**
 * Omega_j, Psi_j, dE_dWkj, and deltaW_kj represent intermediates when training the kj weights.
 */
double* Omega_j;
double* Psi_j;
double** dE_dWkj;
double** deltaW_kj;

/**
 * These are used to track training status.
 */
int iterations;
double averageError;

/**
 * f is the activation function (currently set to sigmoid).
 */
double f(double x)
{
   return 1 / (1 + exp(-x));
}

/**
 * fPrime, or f'(x), is the derivative of the activation function f(x).
 */
double fPrime(double x)
{
   return f(x) * (1 - f(x));
}

/**
 * setConfig sets the configuration parameters for the network.
 * Parameters include the network configuration (how many input layers and how many hidden layers), lambda (learning
 * rate), maximum number of iterations, the error threshold, the mode for population of weights, random number
 * minimum and maximum boundaries, whether truth tables should be printed, and the execution mode of the network.
 */
void setConfig()
{
   NUM_INPUTS = 2, NUM_HIDDEN = 5;
   LAMBDA = 0.3;
   MAX_ITERATIONS = 100000;
   ERROR_THRESHOLD = 0.0002;
   POPULATE_MODE = RANDOM;
   RANDOM_MIN = -1.5, RANDOM_MAX = 1.5;
   PRINT_TRUTH_TABLES = true;
   EXECUTION_MODE = TRAIN;

   return;
} // void setConfig()

/**
 * echoConfig prints out the configuration parameters for the network.
 * This is used as a sanity check to ensure all parameters are as expected.
 */
void echoConfig()
{
   cout << "\nCONFIGURATION PARAMETERS:\n";

   cout << "Network Configuration: " << NUM_INPUTS << "-" << NUM_HIDDEN << "-1" << endl;
   cout << "Execution Mode: " << (EXECUTION_MODE == TRAIN ? "training" : "running") << endl;
   cout << "Weight Population Mode: " << (POPULATE_MODE == RANDOM ? "random" : "custom") << endl;
   cout << "Print Truth Tables: " << (PRINT_TRUTH_TABLES ? "enabled" : "disabled") << endl;

   if (EXECUTION_MODE == TRAIN) // only print training-related parameters if training mode is selected
   {
      cout << "Lambda: " << LAMBDA << endl;
      cout << "Maximum Iterations: " << MAX_ITERATIONS << endl;
      cout << "Average Error Threshold: " << ERROR_THRESHOLD << endl;
      cout << "Random Number Bounds: [" << RANDOM_MIN << ", " << RANDOM_MAX << ")\n";
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
   INPUTS = new double*[TEST_CASES];
   for (int test = 0; test < TEST_CASES; test++)
      INPUTS[test] = new double[NUM_INPUTS];
   TRUTH = new double[TEST_CASES];
   results = new double[TEST_CASES];

/**
 * Allocate memory for the activation nodes.
 */
   a = new double[NUM_INPUTS];
   h = new double[NUM_HIDDEN];

/**
 * Allocate memory for the kj layer arrays (weights, dE/dW, and delta of weights).
 */
   W_kj = new double*[NUM_INPUTS];
   dE_dWkj = new double*[NUM_INPUTS];
   deltaW_kj = new double*[NUM_INPUTS];
   for (int k = 0; k < NUM_INPUTS; k++)
   {
      W_kj[k] = new double[NUM_HIDDEN];
      dE_dWkj[k] = new double[NUM_HIDDEN];
      deltaW_kj[k] = new double[NUM_HIDDEN];
   }

/**
 * Allocate memory for the Theta, Omega, and Psi arrays.
 */
   Theta_j = new double[NUM_HIDDEN];
   Omega_j = new double[NUM_HIDDEN];
   Psi_j = new double[NUM_HIDDEN];

/**
 * Allocate memory for the j0 layer arrays (weights, dE/dW, and delta of weights).
 */
   W_j0 = new double[NUM_HIDDEN];
   dE_dWj0 = new double[NUM_HIDDEN];
   deltaW_j0 = new double[NUM_HIDDEN];

   return;
} // void allocateMemory()

/**
 * deallocateMemory performs garbage-collection by deallocating memory for the network arrays.
 * Deletes all arrays dynamically allocated in the allocateMemory function.
 */
void deallocateMemory()
{
   for (int test = 0; test < TEST_CASES; test++)
      delete[] INPUTS[test];
   delete[] INPUTS;
   delete[] TRUTH;
   delete[] results;

   delete[] a;
   delete[] h;

   for (int k = 0; k < NUM_INPUTS; k++)
   {
      delete[] W_kj[k];
      delete[] dE_dWkj[k];
      delete[] deltaW_kj[k];
   }
   delete[] W_kj;
   delete[] dE_dWkj;
   delete[] deltaW_kj;
   
   delete[] W_j0;
   delete[] dE_dWj0;
   delete[] deltaW_j0;

   delete[] Theta_j;
   delete[] Omega_j;
   delete[] Psi_j;

   return;
} // void deallocateMemory()

/**
 * randomize returns a random double between the range of RANDOM_MIN (inclusive) and RANDOM_MAX (exclusive).
 */
double randomize()
{
   return ((double) rand() / (double) RAND_MAX) * (RANDOM_MAX - RANDOM_MIN) + RANDOM_MIN;
}

/**
 * populateArrays populates the truth table and test case inputs (hardcoded), as well as the weights.
 * There are two population modes for weights.
 * If POPULATE_MODE is RANDOM, all weights are initialized using the randomize() function.
 * If POPULATE_MODE is CUSTOM, the weights are manually set in the function to anything of the user's choice.
 */
void populateArrays()
{
/**
 * Populate the truth table (inputs and truths).
 */
   INPUTS[0][0] = 0.0, INPUTS[0][1] = 0.0;
   INPUTS[1][0] = 0.0, INPUTS[1][1] = 1.0;
   INPUTS[2][0] = 1.0, INPUTS[2][1] = 0.0;
   INPUTS[3][0] = 1.0, INPUTS[3][1] = 1.0;

   TRUTH[0] = 0.0;
   TRUTH[1] = 1.0;
   TRUTH[2] = 1.0;
   TRUTH[3] = 0.0;

/**
 * Populate the weights (options: random and custom).
 */
   if (POPULATE_MODE == RANDOM)
   {
      srand(time(NULL)); // seeds the random number generator

      for (int k = 0; k < NUM_INPUTS; k++)
         for (int j = 0; j < NUM_HIDDEN; j++)
            W_kj[k][j] = randomize();

      for (int j = 0; j < NUM_HIDDEN; j++)
         W_j0[j] = randomize();
   } // if (POPULATE_MODE == RANDOM)
   else // custom weight population
   {
      W_kj[0][0] = 0.4;
      W_kj[0][1] = 0.3;
      W_kj[1][0] = 0.3;
      W_kj[1][1] = 0.4;
      W_j0[0] = 0.5;
      W_j0[1] = 0.5;
   } // if (POPULATE_MODE == RANDOM) ... else

   return;
}

/**
 * runNetwork runs the network by computing the hidden layers based on the input layer and the k-j weights,
 * and the output node (F0) based on the hidden layers and the j-0 weights.
 * This function assumes that the input layer a has already been set.
 */
void runNetwork()
{
/**
 * Compute the hidden layer activations (h).
 */
   for (int j = 0; j < NUM_HIDDEN; j++)
   {
      Theta_j[j] = 0.0;
      for (int K = 0; K < NUM_INPUTS; K++)
         Theta_j[j] += a[K] * W_kj[K][j];
      h[j] = f(Theta_j[j]);
   } // for (int j = 0; j < NUM_HIDDEN; j++)

/**
 * Compute the output layer activations (F0).
 */
   Theta_0 = 0.0;
   for (int J = 0; J < NUM_HIDDEN; J++)
      Theta_0 += h[J] * W_j0[J];
   F0 = f(Theta_0);

   return;
} // void runNetwork()

/**
 * trainNetwork trains the network. For each iteration, the network loops over each test case and uses
 * gradient descent to update the weights on the k-j and j-0 layers. The math can be found in
 * the design document. Training stops either when the maximum iteration limit is reached or when the
 * average error goes under the error threshold.
 */
void trainNetwork()
{
   iterations = 0;
   
   do // while (iterations < MAX_ITERATIONS && averageError > ERROR_THRESHOLD);
   {
      double totalError = 0.0;

      for (int test = 0; test < TEST_CASES; test++)
      {
         a[0] = INPUTS[test][0];
         a[1] = INPUTS[test][1];
         T0 = TRUTH[test];

         runNetwork();

/**
 * Calculate the delta for the j-0 weights.
 */
         omega_0 = T0 - F0;
         psi_0 = omega_0 * fPrime(Theta_0);
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            dE_dWj0[j] = -h[j] * psi_0;
            deltaW_j0[j] = -LAMBDA * dE_dWj0[j];
         } // for (int j = 0; j < NUM_HIDDEN; j++)

/**
 * Calculate the delta for the k-j weights.
 */
         for (int j = 0; j < NUM_HIDDEN; j++)
         {
            Omega_j[j] = psi_0 * W_j0[j];
            Psi_j[j] = Omega_j[j] * fPrime(Theta_j[j]);

            for (int k = 0; k < NUM_INPUTS; k++)
            {
               dE_dWkj[k][j] = -a[k] * Psi_j[j];
               deltaW_kj[k][j] = -LAMBDA * dE_dWkj[k][j];
            } // for (int k = 0; k < NUM_INPUTS; k++)
         } // for (int j = 0; j < NUM_HIDDEN; j++)

/**
 * Update the k-j and j-0 weights.
 */
         for (int k = 0; k < NUM_INPUTS; k++)
            for (int j = 0; j < NUM_HIDDEN; j++)
               W_kj[k][j] += deltaW_kj[k][j];
         
         for (int j = 0; j < NUM_HIDDEN; j++)
            W_j0[j] += deltaW_j0[j];

         totalError += 0.5 * omega_0 * omega_0;
      } // for (int test = 0; test < TEST_CASES; test++)

      iterations++;
      averageError = totalError / (double) TEST_CASES;
   }
   while (iterations < MAX_ITERATIONS && averageError > ERROR_THRESHOLD);

   return;
} // void trainNetwork()

/**
 * Uses the EXECUTION_MODE to either train the network by calling the trainNetwork() function
 * or run the network by calling runNetwork() for each test case.
 * When running the network, average error is also calculated.
 */
void trainOrRun()
{
   if (EXECUTION_MODE == TRAIN)
      trainNetwork();
   
/**
 * The network is always run, even after training. This is to ensure that the results
 * array is correct so that reportResults will function correctly.
 */
   double totalError = 0.0; // will be used to find averageError later

   for (int test = 0; test < TEST_CASES; test++)
   {
      a[0] = INPUTS[test][0];
      a[1] = INPUTS[test][1];
      T0 = TRUTH[test];

      runNetwork();

      results[test] = F0; // store F0 in results to report later
      totalError += 0.5*(T0-F0)*(T0-F0);
   } // for (int test = 0; test < TEST_CASES; test++)

   averageError = totalError / (double) TEST_CASES;

   return;
}

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

   if (EXECUTION_MODE == TRAIN)
   {
      cout << "The network trained on " << TEST_CASES << " test cases.\n";
      cout << "Training stopped because ";

      if (averageError <= ERROR_THRESHOLD)
         cout << "average error was less than the error threshold (" << ERROR_THRESHOLD << ")";
      else
         cout << "the network reached the maximum iteration limit of " << MAX_ITERATIONS << " iterations";
      cout << ".\n\n";

      cout << "Iterations: " << iterations << endl;
   } // if (EXECUTION_MODE == TRAIN)
   else // EXECUTION_MODE was RUN
   {
      cout << "The network ran " << TEST_CASES << " test cases.\n";
   }

   cout << "Average Error: " << averageError << "\n\n";

   if (PRINT_TRUTH_TABLES)
   {
      cout << "Case\t| a_0\t| a_1\t| F0\t\t| T0\t| Error\n";
      cout << "--------------------------------------------------------------\n";

      for (int test = 0; test < TEST_CASES; test++)
      {
         cout << test+1 << "\t| "; // print a 1-indexed test case number for readability

         cout << INPUTS[test][0] << "\t| " << INPUTS[test][1] << "\t| ";
         printf("%.8f\t| ", results[test]); // prints F0 with 8 decimal places
         cout << TRUTH[test] << "\t| ";

         omega_0 = TRUTH[test] - results[test];
         printf("%.8f\n", 0.5*omega_0*omega_0); // prints error with 8 decimal places
      } // for (int test = 0; test < TEST_CASES; test++)

      cout << endl;
   } // if (PRINT_TRUTH_TABLES)

   return;
}

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