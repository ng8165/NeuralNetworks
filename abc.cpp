/**
 * A-B-C Network
 * Author: Nelson Gou
 * Creation Date: 2/21/24
 * Functional Description: This is an A-B-C multilayer perceptron network that uses gradient
 * descent learning. The network has two execution modes, running and training.
 * Other configuration parameters can be modified in the setConfig method.
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cfloat>

using namespace std;

#define WEIGHTS_FILE "weights.bin" // name of the file where weights are loaded and saved

/**
 * These enums allow for increased readability for certain configuration parameters.
 */
enum PopulationMode
{
   RANDOM,  // denotes random weight population
   CUSTOM,  // denotes custom weight population
   IMPORT   // denotes file weight population
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
int numOutputs;               // the number of output layers
double lambda;                // learning rate
int maxIterations;            // maximum iteration limit
double errorThreshold;        // average error threshold
PopulationMode populateMode;  // weight population mode (either RANDOM, CUSTOM, or IMPORT)
double randomMin;             // random number generation minimum bound
double randomMax;             // random number generation maximum bound
bool printTruthTables;        // print truth tables
bool exportWeights;           // exporting weights
ExecutionMode executionMode;  // execution mode (either TRAIN or RUN)

/**
 * These variables represent the test cases, the truth tables, and the stored results.
 */
int testCases;    // the number of test cases to run
double** inputs;  // the inputs to the truth table
double** truth;   // the truth table
double** results; // the results of running each test case

/**
 * a, h, and F represent the activation layers. T represents the truth.
 */
double* a;  // the a layer is the input layer
double* h;  // the h layer is the hidden layer
double* F;  // the F layer is the output layer
double* T;  // T represents the truth

/**
 * W_kj and W_ji represent the weight arrays for the k-j and j-i layers, respectively.
 */
double** W_kj; // the weights for the k-j layer
double** W_ji; // the weights for the j-i layer

/**
 * Theta_j and Theta_i represent intermediate arrays when running the network.
 */
double* Theta_j;  // the Theta for the j layer
double* Theta_i;  // the Theta for the i layer

/**
 * omega_i, psi_i, dE_dWji, and deltaW_ji represent intermediates when training the j-i weights.
 */
double* omega_i;     // the omega for the j-i layer
double* psi_i;       // the psi for the j-i layer
double** dE_dWji;    // the dE/dW for the j-i layer
double** deltaW_ji;  // the deltaW for the j-i layer

/**
 * Omega_j, Psi_j, dE_dWkj, and deltaW_kj represent intermediates when training the k-j weights.
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
 * The sigmoid function returns the sigmoid of x.
 */
double sigmoid(double x)
{
   return 1.0 / (1.0 + exp(-x));
}

/**
 * The sigmoidDerivative function returns the derivative of sigmoid of x.
 */
double sigmoidDerivative(double x)
{
   double fx = sigmoid(x);
   return fx * (1.0 - fx);
}

/**
 * f is the activation function (currently set to sigmoid).
 */
double f(double x)
{
   return sigmoid(x);
}

/**
 * fDerivative, or f'(x), is the derivative of the activation function f(x).
 * It is currently set to the derivative of sigmoid.
 */
double fDerivative(double x)
{
   return sigmoidDerivative(x);
}

/**
 * setConfig sets the configuration parameters for the network.
 * Parameters include the number of test cases, network structure (how many input layers, how many hidden
 * layers, and how many output layers), lambda (learning rate), maximum number of iterations, the error threshold,
 * the mode for population of weights, random number minimum and maximum boundaries, whether truth tables should be
 * printed, whether weights should be exported, and the execution mode of the network.
 */
void setConfig()
{
   testCases = 4;             // the number of test cases

   numInputs = 2;             // the number of input layers
   numHidden = 5;             // the number of hidden layers
   numOutputs = 3;            // the number of output layers

   lambda = 0.3;              // learning rate

   maxIterations = 100000;    // maximum iteration limit
   errorThreshold = 0.0002;   // average error threshold

   populateMode = RANDOM;     // weight population mode (either RANDOM, CUSTOM, or IMPORT)
   randomMin = 0.1;           // random number generation minimum bound
   randomMax = 1.5;           // random number generation maximum bound

   printTruthTables = true;   // print truth tables
   exportWeights = true;      // export weights

   executionMode = TRAIN;     // execution mode (either TRAIN or RUN)

   return;
} // void setConfig()

/**
 * echoConfig prints out the configuration parameters for the network specified in setConfig.
 * This is used as a sanity check to ensure all parameters are as expected.
 */
void echoConfig()
{
   cout << "\nCONFIGURATION PARAMETERS:\n";

   cout << "Network Configuration: " << numInputs << "-" << numHidden << "-" << numOutputs << endl;

   cout << "Execution Mode: " << (executionMode == TRAIN ? "training" : "running") << endl;

   cout << "Weight Population Mode: ";

   if (populateMode == RANDOM)
      cout << "random\n";
   else if (populateMode == CUSTOM)
      cout << "custom\n";
   else
      cout << "file\n";

   cout << "Print Truth Tables: " << (printTruthTables ? "enabled" : "disabled") << endl;

   cout << "Export Weights: " << (exportWeights ? "enabled" : "disabled") << endl;

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
 * loadWeights loads the weights from the WEIGHTS_FILE into W_kj and W_ji.
 * WEIGHTS_FILE is assumed to be a binary file generated using the saveWeights function.
 * If the file does not exist, an error message is printed and execution is aborted.
 */
void loadWeights()
{
   ifstream fileIn(WEIGHTS_FILE, ios::binary | ios::in); // set an input stream to read in binary from WEIGHTS_FILE

   if (!fileIn)
      throw runtime_error("ERROR: " WEIGHTS_FILE " could not be opened to load weights.");
   
   for (int k = 0; k < numInputs; k++)
      fileIn.read(reinterpret_cast<char*>(W_kj[k]), sizeof(double) * numHidden);    // read into W_kj
   
   for (int j = 0; j < numHidden; j++)
      fileIn.read(reinterpret_cast<char*>(W_ji[j]), sizeof(double) * numOutputs);   // read into W_ji

   fileIn.close(); // close the input stream

   return;
} // void loadWeights()

/**
 * saveWeights saves W_kj and W_ji into WEIGHTS_FILE as binary.
 * WEIGHTS_FILE can then be imported into the network with the loadWeights function.
 */
void saveWeights()
{
   ofstream fileOut(WEIGHTS_FILE, ios::binary | ios::out); // set an output stream to write binary to WEIGHTS_FILE

   for (int k = 0; k < numInputs; k++)
      fileOut.write(reinterpret_cast<char*>(W_kj[k]), sizeof(double) * numHidden);  // write to W_kj
   
   for (int j = 0; j < numHidden; j++)
      fileOut.write(reinterpret_cast<char*>(W_ji[j]), sizeof(double) * numOutputs); // write to W_ji

   fileOut.close(); // close the output stream

   return;
} // void saveWeights()

/**
 * allocateMemory allocates memory for the network arrays (weights, activations, deltas, omegas, psis,
 * dE/dW, truth tables, inputs, results, etc.).
 */
void allocateMemory()
{
/**
 * Allocate memory for input arrays, truth table, and results array.
 */
   inputs = new double*[testCases];
   truth = new double*[testCases];
   results = new double*[testCases];

   for (int test = 0; test < testCases; test++)
   {
      inputs[test] = new double[numInputs];
      truth[test] = new double[numOutputs];
      results[test] = new double[numOutputs];
   }

/**
 * Allocate memory for the layers and T.
 */
   a = new double[numInputs];
   h = new double[numHidden];
   F = new double[numOutputs];
   T = new double[numOutputs];

/**
 * Allocate memory for the weights for both the k-j and j-i layer.
 */
   W_kj = new double*[numInputs];

   for (int k = 0; k < numInputs; k++)
      W_kj[k] = new double[numHidden];
   
   W_ji = new double*[numHidden];
   
   for (int j = 0; j < numHidden; j++)
      W_ji[j] = new double[numOutputs];

   if (executionMode == TRAIN)
   {
/**
 * Allocate memory for the dE/dW and the delta of weights for both the k-j and j-i layer.
 */
      dE_dWji = new double*[numHidden];
      deltaW_ji = new double*[numHidden];

      for (int j = 0; j < numHidden; j++)
      {
         dE_dWji[j] = new double[numOutputs];
         deltaW_ji[j] = new double[numOutputs];
      }

      dE_dWkj = new double*[numInputs];
      deltaW_kj = new double*[numInputs];

      for (int k = 0; k < numInputs; k++)
      {
         dE_dWkj[k] = new double[numHidden];
         deltaW_kj[k] = new double[numHidden];
      }

/**
 * Allocate memory for the Theta, Omega, and Psi arrays.
 */
      Theta_i = new double[numOutputs];
      Theta_j = new double[numHidden];

      omega_i = new double[numOutputs];
      Omega_j = new double[numHidden];

      psi_i = new double[numOutputs];
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
   for (int test = 0; test < testCases; test++)
   {
      delete[] inputs[test];
      delete[] truth[test];
      delete[] results[test];
   }
   
   delete[] inputs;
   delete[] truth;
   delete[] results;

/**
 * Deallocate memory for the activation layers and T.
 */
   delete[] a;
   delete[] h;
   delete[] F;
   delete[] T;

/**
 * Deallocate memory for the weights for both the k-j and j-i layer.
 */
   for (int k = 0; k < numInputs; k++)
      delete[] W_kj[k];

   delete[] W_kj;

   for (int j = 0; j < numHidden; j++)
      delete[] W_ji[j];

   delete[] W_ji;

   if (executionMode == TRAIN)
   {
/**
 * Deallocate memory for the dE/dW and the delta of weights for both the k-j and j-i layer.
 */
      for (int j = 0; j < numHidden; j++)
      {
         delete[] dE_dWji[j];
         delete[] deltaW_ji[j];
      }

      delete[] dE_dWji;
      delete[] deltaW_ji;

      for (int k = 0; k < numInputs; k++)
      {
         delete[] dE_dWkj[k];
         delete[] deltaW_kj[k];
      }
      
      delete[] dE_dWkj;
      delete[] deltaW_kj;
      
/**
 * Deallocate memory for the Theta, Omega, and Psi arrays.
 */
      delete[] Theta_i;
      delete[] Theta_j;

      delete[] omega_i;
      delete[] Omega_j;

      delete[] psi_i;
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
 * If populateMode is IMPORT, the weights are inputted from the file as binary and initialized into the weight arrays.
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

   truth[0][0] = 0.0;
   truth[1][0] = 0.0;
   truth[2][0] = 0.0;
   truth[3][0] = 1.0;

   truth[0][1] = 0.0;
   truth[1][1] = 1.0;
   truth[2][1] = 1.0;
   truth[3][1] = 1.0;

   truth[0][2] = 0.0;
   truth[1][2] = 1.0;
   truth[2][2] = 1.0;
   truth[3][2] = 0.0;

/**
 * Populate the weights (options: RANDOM, CUSTOM, or IMPORT).
 */
   if (populateMode == RANDOM)
   {
      srand(time(NULL)); // seeds the random number generator
      rand();            // needed to return random numbers correctly

      for (int k = 0; k < numInputs; k++)
         for (int j = 0; j < numHidden; j++)
            W_kj[k][j] = randomize();

      for (int j = 0; j < numHidden; j++)
         for (int i = 0; i < numOutputs; i++)
            W_ji[j][i] = randomize();
   } // if (populateMode == RANDOM)
   else if (populateMode == CUSTOM)
   {
      W_kj[0][0] = 0.4;
      W_kj[0][1] = 0.3;

      W_kj[1][0] = 0.3;
      W_kj[1][1] = 0.4;

      W_ji[0][0] = 0.5;
      W_ji[1][0] = 0.5;
   } // if (populateMode == RANDOM) ... else if (populateMode == CUSTOM)
   else
   {
      loadWeights();
   }

   return;
} // void populateArrays()

/**
 * runNetworkForTraining runs the network by computing the hidden layers based on the input layer and the k-j weights,
 * and the output layer (F) based on the hidden layers and the j-i weights.
 * This version of runNetwork is only used during training, as it stores Theta_j and Theta_i values in
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
 * Compute the output layer activations (F).
 */
   for (int i = 0; i < numOutputs; i++)
   {
      Theta_i[i] = 0.0;

      for (int J = 0; J < numHidden; J++)
         Theta_i[i] += h[J] * W_ji[J][i];
      
      F[i] = f(Theta_i[i]);
   } // for (int i = 0; i < numOutputs; i++)

   return;
} // void runNetworkForTraining()

/**
 * runNetwork runs the network by computing the hidden layers based on the input layer and the k-j weights,
 * and the output layer (F) based on the hidden layers and the j-i weights.
 * This version of runNetwork is only used when running (not training), as it creates temporary Theta
 * variables to use as accumulators. This function assumes that the input layer a has already been set.
 */
void runNetwork()
{
   double Theta_j_temp, Theta_i_temp; // temporary Theta variables

/**
 * Compute the hidden layer activations (h).
 */
   for (int j = 0; j < numHidden; j++)
   {
      Theta_j_temp = 0.0;

      for (int K = 0; K < numInputs; K++)
         Theta_j_temp += a[K] * W_kj[K][j];
      
      h[j] = f(Theta_j_temp);
   } // for (int j = 0; j < numHidden; j++)

/**
 * Compute the output layer activations (F).
 */
   for (int i = 0; i < numOutputs; i++)
   {
      Theta_i_temp = 0.0;

      for (int J = 0; J < numHidden; J++)
         Theta_i_temp += h[J] * W_ji[J][i];
      
      F[i] = f(Theta_i_temp);
   } // for (int i = 0; i < numOutputs; i++)

   return;
} // void runNetwork()

/**
 * calculateDeltaWeights computes the delta weights for each layer (the k-j layer and the j-i layer).
 * This function uses gradient descent to train the network.
 * It calculates omega, psi, and dE/dW to compute deltaW.
 */
void calculateDeltaWeights()
{
/**
 * Calculate the delta for the j-i weights.
 */
   for (int i = 0; i < numOutputs; i++)
   {
      omega_i[i] = T[i] - F[i];

      psi_i[i] = omega_i[i] * fDerivative(Theta_i[i]);

      for (int j = 0; j < numHidden; j++)
      {
         dE_dWji[j][i] = -h[j] * psi_i[i];

         deltaW_ji[j][i] = -lambda * dE_dWji[j][i];
      } // for (int j = 0; j < numHidden; j++)
   } // for (int i = 0; i < numOutputs; i++)

/**
 * Calculate the delta for the k-j weights.
 */
   for (int j = 0; j < numHidden; j++)
   {
      Omega_j[j] = 0.0;

      for (int I = 0; I < numOutputs; I++)
         Omega_j[j] += psi_i[I] * W_ji[j][I];

      Psi_j[j] = Omega_j[j] * fDerivative(Theta_j[j]);

      for (int k = 0; k < numInputs; k++)
      {
         dE_dWkj[k][j] = -a[k] * Psi_j[j];
         
         deltaW_kj[k][j] = -lambda * dE_dWkj[k][j];
      } // for (int k = 0; k < numInputs; k++)
   } // for (int j = 0; j < numHidden; j++)

   return;
} // void calculateDeltaWeights()

/**
 * applyDeltaWeights applies the delta weights to each layer (the k-j layer and the j-i layer).
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
 * Update the j-i weights.
 */
   for (int j = 0; j < numHidden; j++)
      for (int i = 0; i < numOutputs; i++)
         W_ji[j][i] += deltaW_ji[j][i];
   
   return;
} // void applyDeltaWeights()

/**
 * For each iteration, the network loops over each test case and uses gradient descent to update the
 * weights on the k-j and j-i layers. Training stops either when the maximum iteration limit is reached
 * or when the average error goes under the error threshold.
 */
void trainNetwork()
{
   iterations = 0;
   averageError = DBL_MAX; // initialize averageError to be larger than the error threshold
   
   while (iterations < maxIterations && averageError > errorThreshold)
   {
      double totalError = 0.0;

      for (int test = 0; test < testCases; test++)
      {
         for (int k = 0; k < numInputs; k++)
            a[k] = inputs[test][k];

         for (int i = 0; i < numOutputs; i++)
            T[i] = truth[test][i];

         runNetworkForTraining();

         for (int i = 0; i < numOutputs; i++)
            totalError += 0.5 * (T[i] - F[i]) * (T[i] - F[i]);

         calculateDeltaWeights();

         applyDeltaWeights();
      } // for (int test = 0; test < testCases; test++)

      iterations++;
      averageError = totalError / (double) testCases;
   } // while (iterations < maxIterations && averageError > errorThreshold)

   return;
} // void trainNetwork()

/**
 * trainOrRun uses the executionMode to either train the network or run the network.
 * When training, trainNetwork() is called. The network is subsequently run to calculate average error
 * and initialize the results array for printing later (if specified in the configuration).
 * When running, runNetwork() is called for each test case.
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

   for (int test = 0; test < testCases; test++)
   {
      for (int k = 0; k < numInputs; k++)
         a[k] = inputs[test][k];

      for (int i = 0; i < numOutputs; i++)
         T[i] = truth[test][i];

      runNetwork();

      for (int i = 0; i < numOutputs; i++)
         results[test][i] = F[i];

      for (int i = 0; i < numOutputs; i++)
         totalError += 0.5 * (T[i] - F[i]) * (T[i] - F[i]);
   } // for (int test = 0; test < testCases; test++)

   averageError = totalError / (double) testCases;

   return;
} // void trainOrRun()

/**
 * Prints results from the network's execution.
 * If the network was trained, explains the reason why training was stopped and prints the number
 * of iterations and average error. If the network was run, prints the average error.
 * Only prints truth tables if specified in the configuration parameters.
 */
void reportResults()
{
   cout << "\nRESULTS:\n";

   if (executionMode == TRAIN)
   {
      cout << "The network trained on " << testCases << " test cases.\n";
      cout << "Training stopped because ";

      if (iterations >= maxIterations)
         cout << "the network reached the maximum iteration limit of " << maxIterations << " iterations";
      else
         cout << "average error was less than the error threshold (" << errorThreshold << ")";
      cout << ".\n\n";

      cout << "Iterations: " << iterations << endl;
   } // if (executionMode == TRAIN)
   else // executionMode was RUN
   {
      cout << "The network ran " << testCases << " test cases.\n";
   }

   cout << "Average Error: " << averageError << "\n\n";

   if (printTruthTables)
   {
/**
 * Print table header.
 */
      cout << "Case\t| ";

      for (int k = 0; k < numInputs; k++)
         cout << "a[" << k << "]\t| ";
      
      for (int i = 0; i < numOutputs; i++)
         cout << "F[" << i << "]\t\t| ";
      
      for (int i = 0; i < numOutputs; i++)
         cout << "T[" << i << "]\t| ";
      
      cout << "Error\n";

      cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

/**
 * Print each row of the table.
 */
      for (int test = 0; test < testCases; test++)
      {
         cout << test + 1 << "\t| "; // print a 1-indexed test case number for readability

         for (int k = 0; k < numInputs; k++)
            printf("%.1f\t| ", inputs[test][k]);   // prints a[k] with 1 decimal place
         
         for (int i = 0; i < numOutputs; i++)
            printf("%.8f\t| ", results[test][i]);  // prints F[i] with 8 decimal places
         
         for (int i = 0; i < numOutputs; i++)
            printf("%.1f\t| ", truth[test][i]);    // prints T[i] with 1 decimal place
         
         double caseError = 0.0;

         for (int i = 0; i < numOutputs; i++)
         {
            double omega_i = truth[test][i] - results[test][i];
            caseError += 0.5 * omega_i * omega_i;
         }

         printf("%.8f", caseError);               // prints error with 8 decimal places
         cout << endl;
      } // for (int test = 0; test < testCases; test++)

      cout << endl;
   } // if (printTruthTables)

   return;
} // void reportResults()

/**
 * The main function sets and echoes the configuration parameters.
 * It then allocates memory and populates the weight arrays based on the configuration parameters.
 * It then either runs or trains (again based on configuration parameters).
 * Then, results are reported (also based on configuration parameters).
 * Weights are saved if specified by the configuration.
 * Finally, memory management is performed as large arrays are garbage-collected and deleted.
 */
int main()
{
   try
   {
      setConfig();
      echoConfig();

      allocateMemory();
      populateArrays();

      trainOrRun();

      reportResults();

      if (exportWeights)
         saveWeights();

      deallocateMemory();

      return 0;
   } // try
   catch (const exception& e)
   {
      cerr << "\n" << e.what() << "\n\n";
      return 1;
   }
} // int main()
