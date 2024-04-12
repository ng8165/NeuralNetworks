/**
 * A-B-C-D Backpropagation Network
 * Author: Nelson Gou
 * Creation Date: 3/26/24
 * 
 * Functional Description: This is an A-B-C-D multilayer perceptron network that uses gradient
 * descent learning with backpropagation optimization. The network has two execution modes,
 * running and training. Other configuration parameters can be set in a configuration file.
 * The configuration file must be in standard TOML format.
 * 
 * Table of Contents:
 * - double sigmoid(double x)
 * - double sigmoidDerivative(double x)
 * - double f(double x)
 * - double fDerivative(double x)
 * - void setConfig()
 * - void setConfigManual()
 * - void echoConfig()
 * - void loadWeights()
 * - void saveWeights()
 * - void allocateMemory()
 * - void deallocateMemory()
 * - double randomize()
 * - void populateTestsAndTruthManual()
 * - void populateTestsAndTruth()
 * - void populateArrays()
 * - void runNetworkForTraining()
 * - double runNetworkWithError()
 * - void runNetwork()
 * - void trainNetwork()
 * - void trainOrRun()
 * - void reportResults()
 * - int main(int argc, char** argv)
 * 
 * Libraries Used: toml++ (https://marzer.github.io/tomlplusplus/)
 * To install toml++, run the command "git submodule add --depth 1 https://github.com/marzer/tomlplusplus.git tomlplusplus".
 * 
 * Compile with g++ using the command "g++ -std=c++17 -O2 -Wall -I ./tomlplusplus/include abcd_backprop.cpp".
 * Run using the command "./a.out [CONFIG FILE]".
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <stdexcept>
#include <toml++/toml.hpp>

using namespace std;

#define DEFAULT_CONFIG_FILE      "config.toml"  // default configuration file
#define MILLISECONDS_IN_SECOND   1000.0         // number of milliseconds in a second
#define stringify(name)          #name          // returns a variable's name as a string

#define RANDOM_WEIGHTS  "RANDOM"
#define CUSTOM_WEIGHTS  "CUSTOM"
#define FILE_WEIGHTS    "FILE"
#define TRAIN_NETWORK   "TRAIN"
#define RUN_NETWORK     "RUN"

#define NUM_TOTAL_LAYERS   4
#define NUM_CONN_LAYERS    3
#define INPUT_LAYER        0
#define HIDDEN_LAYER_1     1
#define HIDDEN_LAYER_2     2
#define OUTPUT_LAYER       3

/**
 * These variables are the configuration parameters.
 */
string configFile;            // the name of the configuration file
int numInputs;                // the number of activations in the input layer
int numHidden1;               // the number of activations in the first hidden layer
int numHidden2;               // the number of activations in the second hidden layer
int numOutputs;               // the number of activations in the output layer
double lambda;                // learning rate
int maxIterations;            // maximum iteration limit
double errorThreshold;        // average error threshold
string populateMode;          // weight population mode (either RANDOM, CUSTOM, or FILE)
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
 * a represents the activation layers. The first index represents the layer.
 * T represents the truth.
 */
double** a; // a represents the activations
double* T;  // T represents the truth

/**
 * Theta and Psi represent intermediate arrays when running the network. The first index
 * represents the layer.
 */
double** Theta;   // the Thetas for all layers
double** Psi;     // the Psis for all layers

/**
 * W is a 3D weights array. The first index represents the layer.
 */
double*** W;   // weights array

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
 * network structure (how many input activations, how many hidden activations, and how many output activations),
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
   numHidden1 = *(config.get(stringify(numHidden1)) -> value<int>());
   numHidden2 = *(config.get(stringify(numHidden2)) -> value<int>());
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
   numInputs = 2;                         // the number of activations in the input layer
   numHidden1 = 5;                        // the number of activations in the first hidden layer
   numHidden2 = 5;                        // the number of activations in the second hidden layer
   numOutputs = 3;                        // the number of activations in the output layer

   lambda = 0.3;                          // learning rate

   maxIterations = 100000;                // maximum iteration limit
   errorThreshold = 0.0002;               // average error threshold

   populateMode = RANDOM_WEIGHTS;         // weight population mode (either RANDOM, CUSTOM, or FILE)
   randomMin = 0.1;                       // random number generation minimum bound
   randomMax = 1.5;                       // random number generation maximum bound
   weightsLoadFile = "weights.bin";       // file where weights are loaded from

   testCases = 4;                         // the number of test cases
   testCasesFile = "testCases.txt";       // file where test cases are located

   truthTableFile = "truthTable_2B3.txt"; // file where truth table is located
   printTruthTables = true;               // print truth tables

   exportWeights = true;                  // export weights
   weightsSaveFile = "weights.bin";       // file where weights are saved to

   executionMode = TRAIN_NETWORK;         // execution mode (either TRAIN or RUN)

   return;
} // void setConfigManual()

/**
 * echoConfig prints out the configuration parameters for the network specified in setConfig.
 * This is used as a sanity check to ensure all parameters are as expected.
 */
void echoConfig()
{
   cout << "\nCONFIGURATION PARAMETERS: (imported from " + configFile + ")\n";

   cout << "Network Configuration: " << numInputs << "-" << numHidden1 << "-" << numHidden2 << "-" << numOutputs << endl;

   cout << "Execution Mode: " << (executionMode == TRAIN_NETWORK ? "training" : "running") << endl;

   cout << "Weight Population Mode: ";

   if (populateMode == RANDOM_WEIGHTS)
   {
      cout << "random\n";
   }
   else if (populateMode == CUSTOM_WEIGHTS)
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

   if (populateMode == RANDOM_WEIGHTS) // only print the random number bounds if populateMode is RANDOM
      cout << "Random Number Bounds: [" << randomMin << ", " << randomMax << "]\n";

   if (executionMode == TRAIN_NETWORK) // only print training-related parameters if training mode is selected
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
 * loadWeights loads the weights from the weightsLoadFile into W_mk, W_kj, and W_ji.
 * weightsLoadFile is assumed to be a binary file generated using the saveWeights function.
 * If the file does not exist, an error message is printed and execution is aborted.
 */
void loadWeights()
{
   ifstream fileIn(weightsLoadFile, ios::binary | ios::in); // set an input stream to read in binary from weightsLoadFile

   if (!fileIn)
      throw runtime_error("ERROR: " + weightsLoadFile + " could not be opened to load weights.");
   
   int fileInputs, fileHidden1, fileHidden2, fileOutputs;
   fileIn.read(reinterpret_cast<char*>(&fileInputs), sizeof(int));
   fileIn.read(reinterpret_cast<char*>(&fileHidden1), sizeof(int));
   fileIn.read(reinterpret_cast<char*>(&fileHidden2), sizeof(int));
   fileIn.read(reinterpret_cast<char*>(&fileOutputs), sizeof(int));

   if (fileInputs != numInputs || fileHidden1 != numHidden1 || fileHidden2 != numHidden2 || fileOutputs != numOutputs)
      throw runtime_error("ERROR: Network configuration in " + weightsLoadFile + " does not match the current configuration.");

   int n = INPUT_LAYER;
   for (int m = 0; m < numInputs; m++)
      fileIn.read(reinterpret_cast<char*>(W[n][m]), sizeof(double) * numHidden1);   // read into m-k weights
   
   n = HIDDEN_LAYER_1;
   for (int k = 0; k < numHidden1; k++)
      fileIn.read(reinterpret_cast<char*>(W[n][k]), sizeof(double) * numHidden2);   // read into k-j weights
   
   n = HIDDEN_LAYER_2;
   for (int j = 0; j < numHidden2; j++)
      fileIn.read(reinterpret_cast<char*>(W[n][j]), sizeof(double) * numOutputs);   // read into j-i weights

   fileIn.close(); // close the input stream

   return;
} // void loadWeights()

/**
 * saveWeights saves W_mk, W_kj, and W_ji into weightsSaveFile as binary.
 * weightsSaveFile can then be imported into the network with the loadWeights function.
 */
void saveWeights()
{
   ofstream fileOut(weightsSaveFile, ios::binary | ios::out); // set an output stream to write binary to weightsSaveFile

   fileOut.write(reinterpret_cast<char*>(&numInputs), sizeof(int));  // write numInputs
   fileOut.write(reinterpret_cast<char*>(&numHidden1), sizeof(int)); // write numHidden1
   fileOut.write(reinterpret_cast<char*>(&numHidden2), sizeof(int)); // write numHidden2
   fileOut.write(reinterpret_cast<char*>(&numOutputs), sizeof(int)); // write numOutputs

   int n = INPUT_LAYER;
   for (int m = 0; m < numInputs; m++)
      fileOut.write(reinterpret_cast<char*>(W[n][m]), sizeof(double) * numHidden1); // write m-k weights
   
   n = HIDDEN_LAYER_1;
   for (int k = 0; k < numHidden1; k++)
      fileOut.write(reinterpret_cast<char*>(W[n][k]), sizeof(double) * numHidden2); // write j-k weights
   
   n = HIDDEN_LAYER_2;
   for (int j = 0; j < numHidden2; j++)
      fileOut.write(reinterpret_cast<char*>(W[n][j]), sizeof(double) * numOutputs); // write j-i weights

   fileOut.close(); // close the output stream

   return;
} // void saveWeights()

/**
 * allocateMemory allocates memory for the network arrays (weights, activations, truth tables,
 * inputs, results, etc). It only allocates training-specific arrays (thetas and psis) if the network is training.
 */
void allocateMemory()
{
   inputs = new double*[testCases];
   truth = new double*[testCases];
   results = new double*[testCases];

   for (int test = 0; test < testCases; test++)
   {
      inputs[test] = new double[numInputs];
      truth[test] = new double[numOutputs];
      results[test] = new double[numOutputs];
   }

   a = new double*[NUM_TOTAL_LAYERS];
   W = new double**[NUM_CONN_LAYERS];

   int n = INPUT_LAYER;
   W[n] = new double*[numInputs];

   for (int m = 0; m < numInputs; m++)
      W[n][m] = new double[numHidden1];

   n = HIDDEN_LAYER_1;
   a[n] = new double[numHidden1];
   W[n] = new double*[numHidden1];

   for (int k = 0; k < numHidden1; k++)
      W[n][k] = new double[numHidden2];

   n = HIDDEN_LAYER_2;
   a[n] = new double[numHidden2];
   W[n] = new double*[numHidden2];
   
   for (int j = 0; j < numHidden2; j++)
      W[n][j] = new double[numOutputs];
   
   n = OUTPUT_LAYER;
   a[n] = new double[numOutputs];

   if (executionMode == TRAIN_NETWORK)
   {
      Theta = new double*[NUM_CONN_LAYERS];
      Psi = new double*[NUM_TOTAL_LAYERS];

      n = HIDDEN_LAYER_1;
      Theta[n] = new double[numHidden1];

      n = HIDDEN_LAYER_2;
      Theta[n] = new double[numHidden2];
      Psi[n] = new double[numHidden2];

      n = OUTPUT_LAYER;
      Psi[n] = new double[numOutputs];
   } // if (executionMode == TRAIN_NETWORK)

   return;
} // void allocateMemory()

/**
 * deallocateMemory performs garbage-collection by deallocating memory for all arrays
 * dynamically allocated in the allocateMemory function.
 */
void deallocateMemory()
{
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
   int n = INPUT_LAYER;
   delete[] W[n];

   n = HIDDEN_LAYER_1;
   delete[] a[n];

   for (int k = 0; k < numHidden1; k++)
      delete[] W[n][k];
   
   delete[] W[n];

   n = HIDDEN_LAYER_2;
   delete[] a[n];

   for (int j = 0; j < numHidden2; j++)
      delete[] W[n][j];
   
   delete[] W[n];

   n = OUTPUT_LAYER;
   delete[] a[n];

   delete[] a;
   delete[] W;

   if (executionMode == TRAIN_NETWORK)
   {
      n = HIDDEN_LAYER_1;
      delete[] Theta[n];

      n = HIDDEN_LAYER_2;
      delete[] Theta[n];
      delete[] Psi[n];

      n = OUTPUT_LAYER;
      delete[] Psi[n];

      delete[] Theta;
      delete[] Psi;
   } // if (executionMode == TRAIN_NETWORK)

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
 * arrays with the standard 2-5-5-3 testing data.
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
      for (int m = 0; m < numInputs; m++)
         testIn >> inputs[test][m];
      
      for (int i = 0; i < numOutputs; i++)
         truthIn >> truth[test][i];
   } // for (int test = 0; test < testCases; test++)

   testIn.close();
   truthIn.close();

   return;
} // void populateTestsAndTruth()

/**
 * populateArrays populates the truth table and test case inputs (hardcoded), as well as the weights.
 * There are two population modes for weights.
 * If populateMode is RANDOM, all weights are initialized using the randomize() function.
 * If populateMode is CUSTOM, the weights are manually set in the function to anything of the user's choice.
 * If populateMode is FILE, the weights are inputted from the file as binary and initialized into the weight arrays.
 */
void populateArrays()
{
   populateTestsAndTruth();

/**
 * Populate the weights (options: RANDOM, CUSTOM, or FILE).
 */
   if (populateMode == RANDOM_WEIGHTS)
   {
      srand(std::time(NULL)); // seeds the random number generator
      rand();                 // needed to return random numbers correctly

      int n = INPUT_LAYER;
      for (int m = 0; m < numInputs; m++)
         for (int k = 0; k < numHidden1; k++)
            W[n][m][k] = randomize();

      n = HIDDEN_LAYER_1;
      for (int k = 0; k < numHidden1; k++)
         for (int j = 0; j < numHidden2; j++)
            W[n][k][j] = randomize();

      n = HIDDEN_LAYER_2;
      for (int j = 0; j < numHidden2; j++)
         for (int i = 0; i < numOutputs; i++)
            W[n][j][i] = randomize();
   } // if (populateMode == RANDOM_WEIGHTS)
   else if (populateMode == CUSTOM_WEIGHTS)
   {
      W[HIDDEN_LAYER_2][0][0] = 0.4;
      W[HIDDEN_LAYER_2][0][1] = 0.3;

      W[HIDDEN_LAYER_2][1][0] = 0.3;
      W[HIDDEN_LAYER_2][1][1] = 0.4;

      W[OUTPUT_LAYER][0][0] = 0.5;
      W[OUTPUT_LAYER][1][0] = 0.5;
   } // if (populateMode == RANDOM_WEIGHTS) ... else if (populateMode == CUSTOM_WEIGHTS)
   else
   {
      loadWeights();
   }

   return;
} // void populateArrays()

/**
 * runNetworkForTraining runs the network by computing the first hidden layers based on the input layer
 * and the m-k weights, the second hidden layer based on the first hidden layer and the k-j weights, and
 * the output layer based on the second hidden layer and the j-i weights.
 * This version of runNetwork is only used during training, as it stores Thetas and Psis in global arrays that
 * are only allocated during training.
 * This function assumes that the input layer has already been set.
 */
void runNetworkForTraining()
{
/**
 * Compute the first hidden layer activations.
 */
   int n = HIDDEN_LAYER_1;

   for (int k = 0; k < numHidden1; k++)
   {
      Theta[n][k] = 0.0;

      for (int M = 0; M < numInputs; M++)
         Theta[n][k] += a[n - 1][M] * W[n - 1][M][k];
      
      a[n][k] = f(Theta[n][k]);
   } // for (int k = 0; k < numHidden1; k++)

/**
 * Compute the second hidden layer activations.
 */
   n = HIDDEN_LAYER_2;

   for (int j = 0; j < numHidden2; j++)
   {
      Theta[n][j] = 0.0;

      for (int K = 0; K < numHidden1; K++)
         Theta[n][j] += a[n - 1][K] * W[n - 1][K][j];
      
      a[n][j] = f(Theta[n][j]);
   } // for (int j = 0; j < numHidden2; j++)

/**
 * Compute the output layer activations.
 */
   n = OUTPUT_LAYER;
   double Theta_i, omega_i;

   for (int i = 0; i < numOutputs; i++)
   {
      Theta_i = 0.0;

      for (int J = 0; J < numHidden2; J++)
         Theta_i += a[n - 1][J] * W[n - 1][J][i];
      
      a[n][i] = f(Theta_i);

      omega_i = T[i] - a[n][i];

      Psi[n][i] = omega_i * fDerivative(Theta_i);
   } // for (int i = 0; i < numOutputs; i++)

   return;
} // void runNetworkForTraining()

/**
 * runNetworkWithError runs the network by computing the first hidden layers based on the input layer
 * and the m-k weights, the second hidden layer based on the first hidden layer and the k-j weights, and
 * the output layer based on the second hidden layer and the j-i weights.
 * This version of runNetwork is only used when running (not training), as it uses a temporary Theta
 * variable as an accumulator. It also calculates case error while running and returns the error.
 * This function assumes that the input layer has already been set.
 */
double runNetworkWithError()
{
   double Theta_temp, caseError = 0.0;

/**
 * Compute the first hidden layer activations.
 */
   int n = HIDDEN_LAYER_1;

   for (int k = 0; k < numHidden1; k++)
   {
      Theta_temp = 0.0;

      for (int M = 0; M < numInputs; M++)
         Theta_temp += a[n - 1][M] * W[n - 1][M][k];
      
      a[n][k] = f(Theta_temp);
   } // for (int k = 0; k < numHidden1; k++)

/**
 * Compute the second hidden layer activations.
 */
   n = HIDDEN_LAYER_2;

   for (int j = 0; j < numHidden2; j++)
   {
      Theta_temp = 0.0;

      for (int K = 0; K < numHidden1; K++)
         Theta_temp += a[n - 1][K] * W[n - 1][K][j];
      
      a[n][j] = f(Theta_temp);
   } // for (int j = 0; j < numHidden2; j++)

/**
 * Compute the output layer activations.
 */
   n = OUTPUT_LAYER;

   for (int i = 0; i < numOutputs; i++)
   {
      Theta_temp = 0.0;

      for (int J = 0; J < numHidden2; J++)
         Theta_temp += a[n - 1][J] * W[n - 1][J][i];
      
      a[n][i] = f(Theta_temp);

      double omega = T[i] - a[n][i];
      caseError += 0.5 * omega * omega;
   } // for (int i = 0; i < numOutputs; i++)

   return caseError;
} // double runNetworkWithError()

/**
 * runNetwork runs the network by computing the first hidden layers based on the input layer
 * and the m-k weights, the second hidden layer based on the first hidden layer and the k-j weights, and
 * the output layer based on the second hidden layer and the j-i weights.
 * This version of runNetwork is only used when running (not training), as it creates a temporary Theta
 * variable to use as an accumulator. It does not calculate or return the case error.
 * This function assumes that the input layer has already been set.
 */
void runNetwork()
{   
/**
 * Compute the first hidden layer activations.
 */
   int n = HIDDEN_LAYER_1;
   double Theta_temp;

   for (int k = 0; k < numHidden1; k++)
   {
      Theta_temp = 0.0;

      for (int M = 0; M < numInputs; M++)
         Theta_temp += a[n - 1][M] * W[n - 1][M][k];
      
      a[n][k] = f(Theta_temp);
   } // for (int k = 0; k < numHidden1; k++)

/**
 * Compute the second hidden layer activations.
 */
   n = HIDDEN_LAYER_2;

   for (int j = 0; j < numHidden2; j++)
   {
      Theta_temp = 0.0;

      for (int K = 0; K < numHidden1; K++)
         Theta_temp += a[n - 1][K] * W[n - 1][K][j];
      
      a[n][j] = f(Theta_temp);
   } // for (int j = 0; j < numHidden2; j++)

/**
 * Compute the output layer activations.
 */
   n = OUTPUT_LAYER;

   for (int i = 0; i < numOutputs; i++)
   {
      Theta_temp = 0.0;

      for (int J = 0; J < numHidden2; J++)
         Theta_temp += a[n - 1][J] * W[n - 1][J][i];
      
      a[n][i] = f(Theta_temp);
   } // for (int i = 0; i < numOutputs; i++)

   return;
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
         int n = INPUT_LAYER;
         a[n] = inputs[test];
         T = truth[test];

         runNetworkForTraining();

/**
 * Calculate and update the weights.
 */
         n = HIDDEN_LAYER_2;

         for (int j = 0; j < numHidden2; j++)
         {
            double Omega_j = 0.0;

            for (int i = 0; i < numOutputs; i++)
            {
               Omega_j += Psi[n + 1][i] * W[n][j][i];
               W[n][j][i] += lambda * a[n][j] * Psi[n + 1][i];
            }

            Psi[n][j] = Omega_j * fDerivative(Theta[n][j]);
         } // for (int j = 0; j < numHidden2; j++)

         n = HIDDEN_LAYER_1;

         for (int k = 0; k < numHidden1; k++)
         {
            double Omega_k = 0.0;

            for (int j = 0; j < numHidden2; j++)
            {
               Omega_k += Psi[n + 1][j] * W[n][k][j];
               W[n][k][j] += lambda * a[n][k] * Psi[n + 1][j];
            }

            double Psi_k = Omega_k * fDerivative(Theta[n][k]);

            for (int m = 0; m < numInputs; m++)
               W[n - 1][m][k] += lambda * a[n - 1][m] * Psi_k;
         } // for (int k = 0; k < numHidden1; k++)

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
   auto start = chrono::steady_clock::now(); // note starting time

   if (executionMode == TRAIN_NETWORK)
   {
      trainNetwork();

      for (int test = 0; test < testCases; test++)
      {
         int n = INPUT_LAYER;
         a[n] = inputs[test];
         T = truth[test];

         runNetwork();

         n = OUTPUT_LAYER;
         for (int i = 0; i < numOutputs; i++)
            results[test][i] = a[n][i];
      } // for (int test = 0; test < testCases; test++)
   } // if (executionMode == TRAIN_NETWORK)
   else
   {
      double totalError = 0.0; // will be used to find averageError later

      for (int test = 0; test < testCases; test++)
      {
         int n = INPUT_LAYER;
         a[n] = inputs[test];
         T = truth[test];

         totalError += runNetworkWithError();

         n = OUTPUT_LAYER;
         for (int i = 0; i < numOutputs; i++)
            results[test][i] = a[n][i];
      } // for (int test = 0; test < testCases; test++)

      averageError = totalError / (double) testCases;
   } // if (executionMode == TRAIN_NETWORK) ... else

   auto end = chrono::steady_clock::now();   // note ending time

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

   if (executionMode == TRAIN_NETWORK)
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
   } // if (executionMode == TRAIN_NETWORK)
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

      for (int m = 0; m < numInputs; m++)
         cout << "a[" << m << "]\t| ";
      
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

         for (int m = 0; m < numInputs; m++)
            printf("%.1f\t| ", inputs[test][m]);   // prints a[m] with 1 decimal place
         
         for (int i = 0; i < numOutputs; i++)
            printf("%.17f\t| ", results[test][i]); // prints F[i] with 17 decimal places
         
         for (int i = 0; i < numOutputs; i++)
            printf("%.1f\t| ", truth[test][i]);    // prints T[i] with 1 decimal place
         
         double caseError = 0.0;

         for (int i = 0; i < numOutputs; i++)
         {
            double omega_i = truth[test][i] - results[test][i];
            caseError += 0.5 * omega_i * omega_i;
         }

         printf("%.17f", caseError);               // prints error with 17 decimal places
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
 */
int main(int argc, char** argv)
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
