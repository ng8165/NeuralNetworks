/**
 * A-B-C Backpropagation Network
 * Author: Nelson Gou
 * Creation Date: 3/4/24
 * 
 * Functional Description: This is an A-B-C multilayer perceptron network that uses gradient
 * descent learning with backpropagation optimization. The network has two execution modes,
 * running and training. Other configuration parameters can be set in a configuration file.
 * The configuration file must be in standard TOML format.
 * 
 * Libraries Used: toml++ (https://marzer.github.io/tomlplusplus/)
 * To install toml++, run the command "git submodule add --depth 1 https://github.com/marzer/tomlplusplus.git tomlplusplus".
 * 
 * Compile with g++ using the command "g++ -std=c++17 -O2 -Wall -I ./tomlplusplus/include abc_backprop.cpp".
 * Run using the command "./a.out [CONFIG FILE]".
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <toml++/toml.hpp>

using namespace std;

#define DEFAULT_CONFIG_FILE "config.toml"    // default configuration file
#define MILLISECONDS_IN_SECOND 1000.0        // number of milliseconds in a second
#define stringify(name) #name                // returns a variable's name as a string

/**
 * These variables are the configuration parameters.
 */
string configFile;            // the name of the configuration file
int numInputs;                // the number of input layers
int numHidden;                // the number of hidden layers
int numOutputs;               // the number of output layers
double lambda;                // learning rate
int maxIterations;            // maximum iteration limit
double errorThreshold;        // average error threshold
string populateMode;          // weight population mode (either RANDOM, CUSTOM, or IMPORT)
double randomMin;             // random number generation minimum bound
double randomMax;             // random number generation maximum bound
string weightsLoadFile;       // file name for loading weights
int testCases;                // the number of test cases to run
string testCasesFile;         // file name where test cases are located
string truthTableFile;        // file name where truth table is located
bool printTruthTables;        // print truth tables
bool exportWeights;           // exporting weights
string weightsSaveFile;       // file name for saving weights
string executionMode;         // execution mode (either TRAIN or RUN)

/**
 * These variables represent the truth tables and the stored results.
 */
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
 * Theta_j and psi_i represent intermediate arrays when running the network.
 */
double* Theta_j;  // the Theta for the j layer
double* psi_i;    // the psi for the j-i layer

/**
 * These are used to track training status.
 */
int iterations;         // the number of iterations already trained
double averageError;    // the average error of each iteration
double runtime;         // the time used in training or running

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
 * The function loads the configuration file in the TOML format and uses the toml++ library
 * to parse and extract the parameters. The function loads all relevant parameters, including
 * network structure (how many input layers, how many hidden layers, and how many output layers),
 * lambda (learning rate), maximum number of iterations, the error threshold, the mode for population
 * of weights, random number minimum and maximum boundaries, file to load weights from, the number of
 * test cases, file to load test cases from, file to load truth table from, whether truth tables should
 * be printed, whether weights should be exported, the file to save weights to, and the execution mode
 * of the network.
 */
void setConfig()
{
   auto config = toml::parse_file(configFile);

   numInputs = *(config.get(stringify(numInputs)) -> value<int>());
   numHidden = *(config.get(stringify(numHidden)) -> value<int>());
   numOutputs = *(config.get(stringify(numOutputs)) -> value<int>());
   
   lambda = *(config.get(stringify(lambda)) -> value<double>());

   maxIterations = *(config.get(stringify(maxIterations)) -> value<int>());
   errorThreshold = *(config.get(stringify(errorThreshold)) -> value<double>());

   populateMode = *(config.get(stringify(populateMode)) -> value<string>());
   randomMin = *(config.get(stringify(randomMin)) -> value<double>());
   randomMax = *(config.get(stringify(randomMax)) -> value<double>());
   weightsLoadFile = *(config.get(stringify(weightsLoadFile)) -> value<string>());

   testCases = *(config.get(stringify(testCases)) -> value<int>());
   testCasesFile = *(config.get(stringify(testCasesFile)) -> value<string>());

   truthTableFile = *(config.get(stringify(truthTableFile)) -> value<string>());
   printTruthTables = *(config.get(stringify(printTruthTables)) -> value<bool>());

   exportWeights = *(config.get(stringify(exportWeights)) -> value<bool>());
   weightsSaveFile = *(config.get(stringify(weightsSaveFile)) -> value<string>());

   executionMode = *(config.get(stringify(executionMode)) -> value<string>());

   return;
} // void setConfig()

/**
 * setConfigManual manually sets the configuration parameters for the network.
 * See setConfig's comment for a description of the parameters that are set.
 * This function should not be needed, but it is left as a failsafe.
 */
void setConfigManual()
{
   numInputs = 2;                         // the number of input layers
   numHidden = 5;                         // the number of hidden layers
   numOutputs = 3;                        // the number of output layers

   lambda = 0.3;                          // learning rate

   maxIterations = 100000;                // maximum iteration limit
   errorThreshold = 0.0002;               // average error threshold

   populateMode = "RANDOM";               // weight population mode (either RANDOM, CUSTOM, or IMPORT)
   randomMin = 0.1;                       // random number generation minimum bound
   randomMax = 1.5;                       // random number generation maximum bound
   weightsLoadFile = "weights.bin";       // file where weights are loaded from

   testCases = 4;                         // the number of test cases
   testCasesFile = "testCases.txt";       // file where test cases are located

   truthTableFile = "truthTable_2B3.txt"; // file where truth table is located
   printTruthTables = true;               // print truth tables

   exportWeights = true;                  // export weights
   weightsSaveFile = "weights.bin";       // file where weights are saved to

   executionMode = "TRAIN";               // execution mode (either TRAIN or RUN)

   return;
} // void setConfig()

/**
 * echoConfig prints out the configuration parameters for the network specified in setConfig.
 * This is used as a sanity check to ensure all parameters are as expected.
 */
void echoConfig()
{
   cout << "\nCONFIGURATION PARAMETERS: (imported from " + configFile + ")\n";

   cout << "Network Configuration: " << numInputs << "-" << numHidden << "-" << numOutputs << endl;

   cout << "Execution Mode: " << (executionMode == "TRAIN" ? "training" : "running") << endl;

   cout << "Weight Population Mode: ";

   if (populateMode == "RANDOM")
   {
      cout << "random\n";
   }
   else if (populateMode == "CUSTOM")
   {
      cout << "custom\n";
   }
   else
   {
      cout << "file\n";
      cout << "Weights Import File: " << weightsLoadFile << endl;
   }

   cout << "Print Truth Tables: " << (printTruthTables ? "enabled" : "disabled") << endl;

   cout << "Export Weights: " << (exportWeights ? "enabled" : "disabled") << endl;

   if (exportWeights)
      cout << "Weights Export File: " << weightsSaveFile << endl;

   if (populateMode == "RANDOM") // only print the random number bounds if populateMode is RANDOM
      cout << "Random Number Bounds: [" << randomMin << ", " << randomMax << "]\n";

   if (executionMode == "TRAIN") // only print training-related parameters if training mode is selected
   {
      cout << "Lambda: " << lambda << endl;
      cout << "Maximum Iterations: " << maxIterations << endl;
      cout << "Average Error Threshold: " << errorThreshold << endl;
   }

   cout << "Test Cases: " << testCases << endl;
   cout << "Test Case File: " << testCasesFile << endl;
   cout << "Truth Table File: " << truthTableFile << endl;

   return;
} // void echoConfig()

/**
 * loadWeights loads the weights from the weightsLoadFile into W_kj and W_ji.
 * weightsLoadFile is assumed to be a binary file generated using the saveWeights function.
 * If the file does not exist, an error message is printed and execution is aborted.
 */
void loadWeights()
{
   ifstream fileIn(weightsLoadFile, ios::binary | ios::in); // set an input stream to read in binary from weightsLoadFile

   if (!fileIn)
      throw runtime_error("ERROR: " + weightsLoadFile + " could not be opened to load weights.");
   
   int fileInputs, fileHidden, fileOutputs;
   fileIn.read(reinterpret_cast<char*>(&fileInputs), sizeof(int));
   fileIn.read(reinterpret_cast<char*>(&fileHidden), sizeof(int));
   fileIn.read(reinterpret_cast<char*>(&fileOutputs), sizeof(int));

   if (fileInputs != numInputs || fileHidden != numHidden || fileOutputs != numOutputs)
      throw runtime_error("ERROR: Network configuration in "+ weightsLoadFile + " does not match the current configuration.");
   
   for (int k = 0; k < numInputs; k++)
      fileIn.read(reinterpret_cast<char*>(W_kj[k]), sizeof(double) * numHidden);    // read into W_kj
   
   for (int j = 0; j < numHidden; j++)
      fileIn.read(reinterpret_cast<char*>(W_ji[j]), sizeof(double) * numOutputs);   // read into W_ji

   fileIn.close(); // close the input stream

   return;
} // void loadWeights()

/**
 * saveWeights saves W_kj and W_ji into weightsSaveFile as binary.
 * weightsSaveFile can then be imported into the network with the loadWeights function.
 */
void saveWeights()
{
   ofstream fileOut(weightsSaveFile, ios::binary | ios::out); // set an output stream to write binary to weightsSaveFile

   fileOut.write(reinterpret_cast<char*>(&numInputs), sizeof(int));  // write numInputs
   fileOut.write(reinterpret_cast<char*>(&numHidden), sizeof(int));  // write numHidden
   fileOut.write(reinterpret_cast<char*>(&numOutputs), sizeof(int)); // write numOutputs

   for (int k = 0; k < numInputs; k++)
      fileOut.write(reinterpret_cast<char*>(W_kj[k]), sizeof(double) * numHidden);  // write W_kj
   
   for (int j = 0; j < numHidden; j++)
      fileOut.write(reinterpret_cast<char*>(W_ji[j]), sizeof(double) * numOutputs); // write W_ji

   fileOut.close(); // close the output stream

   return;
} // void saveWeights()

/**
 * allocateMemory allocates memory for the network arrays (weights, activations, thetas,
 * omegas, psis, truth tables, inputs, results, etc).
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
 * Allocate memory for the h and F layer. Memory does not need to be allocated for the
 * a layer and the T (truth) array since they can just point to corresponding arrays in the truth table.
 */
   h = new double[numHidden];
   F = new double[numOutputs];

/**
 * Allocate memory for the weights for both the k-j and j-i layer.
 */
   W_kj = new double*[numInputs];

   for (int k = 0; k < numInputs; k++)
      W_kj[k] = new double[numHidden];
   
   W_ji = new double*[numHidden];
   
   for (int j = 0; j < numHidden; j++)
      W_ji[j] = new double[numOutputs];

   if (executionMode == "TRAIN")
   {
      Theta_j = new double[numHidden];
      psi_i = new double[numOutputs];
   }

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
 * Deallocate memory for the h and F layers.
 */
   delete[] h;
   delete[] F;

/**
 * Deallocate memory for the weights for both the k-j and j-i layer.
 */
   for (int k = 0; k < numInputs; k++)
      delete[] W_kj[k];

   delete[] W_kj;

   for (int j = 0; j < numHidden; j++)
      delete[] W_ji[j];

   delete[] W_ji;

   if (executionMode == "TRAIN")
   {
      delete[] Theta_j;
      delete[] psi_i;
   }

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
 * populateTestsAndTruthManual manually sets the inputs (test cases) and truth (truth table)
 * arrays with the standard 2-5-3 testing data.
 */
void populateTestsAndTruthManual()
{
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

   return;
} // void populateTestsAndTruthManual()

/**
 * populateTestsAndTruth loads the inputs (test cases) and truth (truth table) arrays from the
 * testCasesFile and truthTableFile that are specified in the configuration.
 */
void populateTestsAndTruth()
{
   ifstream testIn(testCasesFile, ios::in);
   ifstream truthIn(truthTableFile, ios::in);

   if (!testIn)
      throw runtime_error("ERROR: " + testCasesFile + " could not be opened to load the test cases.");
   
   if (!truthIn)
      throw runtime_error("ERROR: " + truthTableFile + " could not be opened to load the truth table.");

   for (int test = 0; test < testCases; test++)
   {
      for (int k = 0; k < numInputs; k++)
         testIn >> inputs[test][k];
      
      for (int i = 0; i < numOutputs; i++)
         truthIn >> truth[test][i];
   }

   testIn.close();
   truthIn.close();

   return;
} // void populateTestsAndTruth()

/**
 * populateArrays populates the truth table and test case inputs (hardcoded), as well as the weights.
 * There are two population modes for weights.
 * If populateMode is RANDOM, all weights are initialized using the randomize() function.
 * If populateMode is CUSTOM, the weights are manually set in the function to anything of the user's choice.
 * If populateMode is IMPORT, the weights are inputted from the file as binary and initialized into the weight arrays.
 */
void populateArrays()
{
   populateTestsAndTruth();

/**
 * Populate the weights (options: RANDOM, CUSTOM, or IMPORT).
 */
   if (populateMode == "RANDOM")
   {
      srand(std::time(NULL)); // seeds the random number generator
      rand();                 // needed to return random numbers correctly

      for (int k = 0; k < numInputs; k++)
         for (int j = 0; j < numHidden; j++)
            W_kj[k][j] = randomize();

      for (int j = 0; j < numHidden; j++)
         for (int i = 0; i < numOutputs; i++)
            W_ji[j][i] = randomize();
   } // if (populateMode == "RANDOM")
   else if (populateMode == "CUSTOM")
   {
      W_kj[0][0] = 0.4;
      W_kj[0][1] = 0.3;

      W_kj[1][0] = 0.3;
      W_kj[1][1] = 0.4;

      W_ji[0][0] = 0.5;
      W_ji[1][0] = 0.5;
   } // if (populateMode == "RANDOM") ... else if (populateMode == "CUSTOM")
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
      double Theta_i = 0.0;

      for (int J = 0; J < numHidden; J++)
         Theta_i += h[J] * W_ji[J][i];
      
      F[i] = f(Theta_i);

      double omega_i = T[i] - F[i];

      psi_i[i] = omega_i * fDerivative(Theta_i);
   } // for (int i = 0; i < numOutputs; i++)

   return;
} // void runNetworkForTraining()

/**
 * runNetworkWithError runs the network by computing the hidden layers based on the input layer and the k-j weights,
 * and the output layer (F) based on the hidden layers and the j-i weights. This function assumes that the
 * input layer a has already been set.
 * This version of runNetwork is only used when running (not training), as it creates temporary Theta
 * variables to use as accumulators. It also calculates case error while running and returns the error.
 */
double runNetworkWithError()
{
   double caseError = 0.0;
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

      caseError += 0.5 * (T[i] - F[i]) * (T[i] - F[i]);
   } // for (int i = 0; i < numOutputs; i++)

   return caseError;
} // double runNetworkWithError()

/**
 * runNetwork runs the network by computing the hidden layers based on the input layer and the k-j weights,
 * and the output layer (F) based on the hidden layers and the j-i weights. This function assumes that the
 * input layer a has already been set.
 * This version of runNetwork is only used when running (not training), as it creates temporary Theta
 * variables to use as accumulators. It does not calculate or return the case error.
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
} // void runNetwork()

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
         a = inputs[test];
         T = truth[test];

         runNetworkForTraining();

/**
 * Calculate and update the weights.
 */
         for (int j = 0; j < numHidden; j++)
         {
            double Omega_j = 0.0;

            for (int i = 0; i < numOutputs; i++)
            {
               Omega_j += psi_i[i] * W_ji[j][i];
               W_ji[j][i] += lambda * h[j] * psi_i[i];
            }

            double Psi_j = Omega_j * fDerivative(Theta_j[j]);
            
            for (int k = 0; k < numInputs; k++)
               W_kj[k][j] += lambda * a[k] * Psi_j;
         } // for (int j = 0; j < numHidden; j++)

         totalError += runNetworkWithError();
      } // for (int test = 0; test < testCases; test++)

      averageError = totalError / (double) testCases;
      iterations++;
   } // while (iterations < maxIterations && averageError > errorThreshold)

   return;
} // void trainNetwork()

/**
 * trainOrRun uses the executionMode to either train the network or run the network.
 * When training, trainNetwork() is called. The network is run with runNetwork() to populate the results array.
 * When running, runNetworkWithError() is called for each test case.
 */
void trainOrRun()
{
   auto start = chrono::steady_clock::now();

   if (executionMode == "TRAIN")
   {
      trainNetwork();

      for (int test = 0; test < testCases; test++)
      {
         a = inputs[test];
         T = truth[test];

         runNetwork();

         for (int i = 0; i < numOutputs; i++)
            results[test][i] = F[i];
      } // for (int test = 0; test < testCases; test++)
   } // if (executionMode == "TRAIN")
   else
   {
/**
 * The network is always run, even after training. This is to ensure that the results
 * array is correct so that reportResults will function correctly.
 */
      double totalError = 0.0; // will be used to find averageError later

      for (int test = 0; test < testCases; test++)
      {
         a = inputs[test];
         T = truth[test];

         totalError += runNetworkWithError();

         for (int i = 0; i < numOutputs; i++)
            results[test][i] = F[i];
      } // for (int test = 0; test < testCases; test++)

      averageError = totalError / (double) testCases;
   } // if (executionMode == "TRAIN") ... else

   auto end = chrono::steady_clock::now();

   runtime = chrono::duration<double>(end - start).count() * MILLISECONDS_IN_SECOND;

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

   if (executionMode == "TRAIN")
   {
      cout << "The network trained on " << testCases << " test cases.\n";
      cout << "Training stopped ";

      if (iterations >= maxIterations)
         cout << "because the network reached the maximum iteration limit of " << maxIterations << " iterations";
      
      if (iterations >= maxIterations && averageError < errorThreshold)
         cout << " and ";
      
      if (averageError < errorThreshold)
         cout << "because average error was less than the error threshold (" << errorThreshold << ")";
      
      cout << ".\n\n";

      cout << "Iterations: " << iterations << endl;
   } // if (executionMode == "TRAIN")
   else // executionMode was RUN
   {
      cout << "The network ran " << testCases << " test cases.\n";
   }

   cout << "Average Error: " << averageError << endl;

   printf("Execution Time: %.3f ms\n\n", runtime);

   if (printTruthTables)
   {
/**
 * Print table header.
 */
      cout << "Case\t| ";

      for (int k = 0; k < numInputs; k++)
         cout << "a[" << k << "]\t| ";
      
      for (int i = 0; i < numOutputs; i++)
         cout << "F[" << i << "]\t\t\t| ";
      
      for (int i = 0; i < numOutputs; i++)
         cout << "T[" << i << "]\t| ";
      
      cout << "Error\n";

      cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~";
      cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";

/**
 * Print each row of the table.
 */
      for (int test = 0; test < testCases; test++)
      {
         cout << test + 1 << "\t| "; // print a 1-indexed test case number for readability

         for (int k = 0; k < numInputs; k++)
            printf("%.1f\t| ", inputs[test][k]);   // prints a[k] with 1 decimal place
         
         for (int i = 0; i < numOutputs; i++)
            printf("%.17f\t| ", results[test][i]); // prints F[i] with 8 decimal places
         
         for (int i = 0; i < numOutputs; i++)
            printf("%.1f\t| ", truth[test][i]);    // prints T[i] with 1 decimal place
         
         double caseError = 0.0;

         for (int i = 0; i < numOutputs; i++)
         {
            double omega_i = truth[test][i] - results[test][i];
            caseError += 0.5 * omega_i * omega_i;
         }

         printf("%.17f", caseError);               // prints error with 8 decimal places
         cout << endl;
      } // for (int test = 0; test < testCases; test++)

      cout << endl;
   } // if (printTruthTables)

   return;
} // void reportResults()

/**
 * The main function first sets the configuration file based on if a file is supplied through the command line.
 * It then loads and echoes the configuration parameters from the file.
 * It then allocates memory and populates the weight arrays based on the configuration parameters.
 * It then either runs or trains (again based on configuration parameters).
 * Then, results are reported (also based on configuration parameters).
 * Weights are saved if specified by the configuration.
 * Finally, memory management is performed as large arrays are garbage-collected and deleted.
 * argc represents the number of commands entered, while argv represents an array of the arguments.
 * 
 */
int main(int argc, char **argv)
{
   try
   {
      configFile = (argc <= 1) ? DEFAULT_CONFIG_FILE : argv[1];

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
