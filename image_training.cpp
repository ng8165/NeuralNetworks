/**
 * N-Layer Network (specialized for image training)
 * Author: Nelson Gou
 * Creation Date: 5/7/24
 * 
 * Functional Description: This is an N-layer perceptron network that uses gradient
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
 * - void echoConfig()
 * - void loadWeights()
 * - void saveWeights()
 * - void allocateMemory()
 * - void deallocateMemory()
 * - double randomize()
 * - void populateTestsAndTruth()
 * - void populateArrays()
 * - void runNetworkForTraining()
 * - double runNetworkWithError()
 * - void runNetwork()
 * - void trainNetwork()
 * - void trainOrRun()
 * - void prettyPrintTime(double milliseconds)
 * - void reportResults()
 * - int main(int argc, char** argv)
 * 
 * Libraries Used: toml++ (https://marzer.github.io/tomlplusplus/)
 * To install toml++, run the command "git submodule add --depth 1 https://github.com/marzer/tomlplusplus.git tomlplusplus".
 * 
 * Compile with g++ using the command "g++ -std=c++17 -O2 -Wall -I ./tomlplusplus/include n_layer.cpp".
 * Run using the command "./a.out [CONFIG FILE]".
 */

#include <iostream>
#include <fstream>
#include <cmath>
#include <cfloat>
#include <chrono>
#include <toml++/toml.hpp>

using namespace std;

#define DEFAULT_CONFIG_FILE      "config.toml"  // default configuration file
#define stringify(name)          #name          // returns a variable's name as a string
#define MIN_LAYERS               2              // the network must have at least two connectivity layers
#define MILLISECONDS_IN_SECOND   1000.0         // number of milliseconds in a second
#define SECONDS_IN_MINUTE        60.0           // number of seconds in a minute
#define MINUTES_IN_HOUR          60.0           // number of minutes in an hour

#define RANDOM_WEIGHTS  "RANDOM"
#define CUSTOM_WEIGHTS  "CUSTOM"
#define FILE_WEIGHTS    "FILE"
#define TRAIN_NETWORK   "TRAIN"
#define RUN_NETWORK     "RUN"

/**
 * These variables are the configuration parameters.
 */
string configFile;            // the name of the configuration file
int numLayers;                // the number of connectivity layers
int numActivationLayers;      // the number of activation layers
int* network;                 // the network configuration, which contains layer sizes
double lambda;                // learning rate
int maxIterations;            // maximum iteration limit
double errorThreshold;        // average error threshold
int keepAlive;                // number of iterations between keep alive messages (0 for no output)
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
 * These variables contain the value of n associated with the layer.
 * Since lastHiddenLayer and outputLayer depend on what is read from the configuration file,
 * they are set in setConfig and are therefore not constants.
 */
#define INPUT_LAYER           0
#define FIRST_HIDDEN_LAYER    1
#define SECOND_HIDDEN_LAYER   2
int lastHiddenLayer;
int outputLayer;

/**
 * These variables represent the truth tables and the stored results.
 */
double** inputs;  // the test cases
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
 * the network structure, lambda (learning rate), maximum number of iterations, the error threshold,
 * the number of iterations between keep alive messages, the mode for population of weights, random
 * number minimum and maximum boundaries, file to load weights from, the number of test cases, file
 * to load test cases from, file to load truth table from, whether truth tables should be printed,
 * whether weights should be exported, the file to save weights to, and the execution mode of the network.
 */
void setConfig()
{
   auto config = toml::parse_file(configFile);

   auto networkArr = config.get_as<toml::array>(stringify(network));
   numActivationLayers = networkArr -> size();
   outputLayer = numLayers = numActivationLayers - 1;
   lastHiddenLayer = outputLayer - 1;

   if (numLayers < MIN_LAYERS)
      throw new runtime_error("ERROR: The network must have at least " + to_string(MIN_LAYERS) + " connectivity layers.");
   
   executionMode = *(config.get(stringify(executionMode)) -> value<string>());
   
   network = new int[numActivationLayers];
   for (int n = INPUT_LAYER; n <= outputLayer; n++)
      network[n] = *(networkArr -> get(n) -> value<int>());
   
   lambda = *(config.get(stringify(lambda)) -> value<double>());

   maxIterations = *(config.get(stringify(maxIterations)) -> value<int>());
   errorThreshold = *(config.get(stringify(errorThreshold)) -> value<double>());
   keepAlive = *(config.get(stringify(keepAlive)) -> value<int>());

   populateMode = *(config.get(stringify(populateMode)) -> value<string>());
   randomMin = *(config.get(stringify(randomMin)) -> value<double>());
   randomMax = *(config.get(stringify(randomMax)) -> value<double>());
   weightsLoadFile = *(config.get(stringify(weightsLoadFile)) -> value<string>());

   testCases = *(config.get(stringify(testCases)) -> value<int>());
   testCasesFile = *(config.get(stringify(testCasesFile)) -> value<string>());

   auto truthTableKey = config.get(stringify(truthTableFile));
   truthTableFile = (truthTableKey == NULL) ? "" : *(truthTableKey -> value<string>());

   if (truthTableFile.empty() && executionMode == TRAIN_NETWORK)
      throw new runtime_error("ERROR: Truth table is required to train the network.");
   
   printTruthTables = *(config.get(stringify(printTruthTables)) -> value<bool>());

   exportWeights = *(config.get(stringify(exportWeights)) -> value<bool>());
   weightsSaveFile = *(config.get(stringify(weightsSaveFile)) -> value<string>());

   return;
} // void setConfig()

/**
 * echoConfig prints out the configuration parameters for the network specified in setConfig.
 * This is used as a sanity check to ensure all parameters are as expected.
 */
void echoConfig()
{
   cout << "\nCONFIGURATION PARAMETERS: (imported from " + configFile + ")\n";

   cout << "Network Configuration: ";
   cout << network[INPUT_LAYER];
   for (int n = FIRST_HIDDEN_LAYER; n <= outputLayer; n++)
      cout << "-" << network[n];

   cout << "\nExecution Mode: " << (executionMode == TRAIN_NETWORK ? "training" : "running") << endl;

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
      cout << "Keep Alive Frequency: ";

      if (keepAlive == 0)
         cout << "disabled\n";
      else
         cout << "every " << keepAlive << " iterations\n";
   } // if (executionMode == TRAIN_NETWORK)

   cout << "Test Cases: " << testCases << endl;
   cout << "Test Case File: " << testCasesFile << endl;
   cout << "Truth Table File: " << (truthTableFile.empty() ? "not given" : truthTableFile) << "\n\n";

   return;
} // void echoConfig()

/**
 * loadWeights loads the weights from the weightsLoadFile into the weights array.
 * weightsLoadFile is assumed to be a binary file generated using the saveWeights function.
 * If the file does not exist, an error message is printed and execution is aborted.
 */
void loadWeights()
{
   ifstream fileIn(weightsLoadFile, ios::binary | ios::in); // set an input stream to read in binary from weightsLoadFile

   if (!fileIn)
      throw runtime_error("ERROR: " + weightsLoadFile + " could not be opened to load weights.");

   int fileTotalLayers;
   fileIn.read(reinterpret_cast<char*>(&fileTotalLayers), sizeof(int));

   if (fileTotalLayers != numActivationLayers)
      throw runtime_error("ERROR: Network configuration in " + weightsLoadFile + " does not match the current configuration.");
   
   for (int n = INPUT_LAYER; n <= outputLayer; n++)
   {
      int fileCurrLayer;
      fileIn.read(reinterpret_cast<char*>(&fileCurrLayer), sizeof(int));

      if (fileCurrLayer != network[n])
         throw runtime_error("ERROR: Network configuration in " + weightsLoadFile + " does not match the current configuration.");
   } // for (int n = INPUT_LAYER; n <= outputLayer; n++)

   for (int n = INPUT_LAYER; n <= lastHiddenLayer; n++)
      for (int k = 0; k < network[n]; k++)
         fileIn.read(reinterpret_cast<char*>(W[n][k]), sizeof(double) * network[n + 1]);

   fileIn.close(); // close the input stream

   return;
} // void loadWeights()

/**
 * saveWeights saves the weights into weightsSaveFile as binary.
 * weightsSaveFile can then be imported into the network with the loadWeights function.
 */
void saveWeights()
{
   ofstream fileOut(weightsSaveFile, ios::binary | ios::out); // set an output stream to write binary to weightsSaveFile

   fileOut.write(reinterpret_cast<char*>(&numActivationLayers), sizeof(int));

   for (int n = INPUT_LAYER; n <= outputLayer; n++)
      fileOut.write(reinterpret_cast<char*>(network + n), sizeof(int));

   for (int n = INPUT_LAYER; n <= lastHiddenLayer; n++)
      for (int k = 0; k < network[n]; k++)
         fileOut.write(reinterpret_cast<char*>(W[n][k]), sizeof(double) * network[n + 1]);

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
      inputs[test] = new double[network[INPUT_LAYER]];
      truth[test] = new double[network[outputLayer]];
      results[test] = new double[network[outputLayer]];
   }

   a = new double*[numActivationLayers];
   W = new double**[numLayers];

   for (int n = FIRST_HIDDEN_LAYER; n <= outputLayer; n++)
      a[n] = new double[network[n]];

   for (int n = INPUT_LAYER; n <= lastHiddenLayer; n++)
   {
      W[n] = new double*[network[n]];
      for (int k = 0; k < network[n]; k++)
         W[n][k] = new double[network[n + 1]];
   }

   if (executionMode == TRAIN_NETWORK)
   {
      Theta = new double*[numLayers];
      Psi = new double*[numActivationLayers];

      for (int n = FIRST_HIDDEN_LAYER; n <= lastHiddenLayer; n++)
         Theta[n] = new double[network[n]];
      
      for (int n = SECOND_HIDDEN_LAYER; n <= outputLayer; n++)
         Psi[n] = new double[network[n]];
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

   for (int n = INPUT_LAYER; n <= lastHiddenLayer; n++)
   {
      for (int k = 0; k < network[n]; k++)
         delete[] W[n][k];
      
      delete[] W[n];
   } // for (int n = INPUT_LAYER; n <= lastHiddenLayer; n++)
   
   for (int n = FIRST_HIDDEN_LAYER; n <= outputLayer; n++)
      delete[] a[n];

   delete[] W;
   delete[] a;

   if (executionMode == TRAIN_NETWORK)
   {
      for (int n = FIRST_HIDDEN_LAYER; n <= lastHiddenLayer; n++)
         delete[] Theta[n];
      
      for (int n = SECOND_HIDDEN_LAYER; n <= outputLayer; n++)
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
 * populateTestsAndTruth loads the inputs (test cases) and truth (truth table) arrays from the
 * testCasesFile and truthTableFile that are specified in the configuration.
 */
void populateTestsAndTruth()
{
   ifstream testIn(testCasesFile, ios::in);

   if (!testIn)   // if testCasesFile cannot be opened
      throw runtime_error("ERROR: " + testCasesFile + " could not be opened to load the test cases.");
   
   for (int test = 0; test < testCases; test++)
      for (int m = 0; m < network[INPUT_LAYER]; m++)
         testIn >> inputs[test][m];

   testIn.close();

   if (!truthTableFile.empty())  // if there is a truth table, read it
   {
      ifstream truthIn(truthTableFile, ios::in);

      if (!truthIn)  // if truthTableFile cannot be opened
         throw runtime_error("ERROR: " + truthTableFile + " could not be opened to load the truth table.");
      
      for (int test = 0; test < testCases; test++)
         for (int i = 0; i < network[outputLayer]; i++)
            truthIn >> truth[test][i];

      truthIn.close();
   } // if (!truthTableFile.empty())

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

      for (int n = INPUT_LAYER; n <= lastHiddenLayer; n++)
         for (int k = 0; k < network[n]; k++)
            for (int j = 0; j < network[n + 1]; j++)
               W[n][k][j] = randomize();
   } // if (populateMode == RANDOM_WEIGHTS)
   else if (populateMode == CUSTOM_WEIGHTS)
   {
      W[0][0][0] = 0.4;
      W[0][0][1] = 0.3;

      W[0][1][0] = 0.3;
      W[0][1][1] = 0.4;

      W[1][0][0] = 0.5;
      W[1][1][0] = 0.5;
   } // if (populateMode == RANDOM_WEIGHTS) ... else if (populateMode == CUSTOM_WEIGHTS)
   else
   {
      loadWeights();
   }

   return;
} // void populateArrays()

/**
 * runNetworkForTraining runs the network by first computing all activations except those in the output layer.
 * Then, the output layer is calculated, and Psis are allocated.
 * This version of runNetwork is only used during training, as it stores Thetas and Psis in global arrays that
 * are only allocated during training.
 * This function assumes that the input layer has already been set.
 */
void runNetworkForTraining()
{
   double Theta_temp;

/**
 * Compute all activations except those in the output layer.
 */
   for (int n = FIRST_HIDDEN_LAYER; n <= lastHiddenLayer; n++)
   {
      for (int k = 0; k < network[n]; k++)
      {
         Theta[n][k] = 0.0;

         for (int m = 0; m < network[n - 1]; m++)
            Theta[n][k] += a[n - 1][m] * W[n - 1][m][k];
         
         a[n][k] = f(Theta[n][k]);
      } // for (int k = 0; k < network[n]; k++)
   } // for (int n = FIRST_HIDDEN_LAYER; n <= lastHiddenLayer; n++)

/**
 * Compute the output layer activations.
 */
   int n = outputLayer;

   for (int k = 0; k < network[n]; k++)
   {
      Theta_temp = 0.0;

      for (int m = 0; m < network[n - 1]; m++)
         Theta_temp += a[n - 1][m] * W[n - 1][m][k];
      
      a[n][k] = f(Theta_temp);
      Psi[n][k] = (T[k] - a[n][k]) * fDerivative(Theta_temp);
   } // for (int k = 0; k < network[n]; k++)

   return;
} // void runNetworkForTraining()

/**
 * runNetworkWithError runs the network by computing all activations except those in the output layer.
 * Then, the output layer is calculated, and case error is calculated.
 * This version of runNetwork is only used when running (not training), as it uses a temporary Theta
 * variable as an accumulator.
 * This function assumes that the input layer has already been set.
 */
double runNetworkWithError()
{
   double Theta_temp, omega, caseError = 0.0;

/**
 * Compute all activations except those in the output layer.
 */
   for (int n = FIRST_HIDDEN_LAYER; n <= lastHiddenLayer; n++)
   {
      for (int k = 0; k < network[n]; k++)
      {
         Theta_temp = 0.0;

         for (int m = 0; m < network[n - 1]; m++)
            Theta_temp += a[n - 1][m] * W[n - 1][m][k];
         
         a[n][k] = f(Theta_temp);
      } // for (int k = 0; k < network[n]; k++)
   } // for (int n = FIRST_HIDDEN_LAYER; n <= lastHiddenLayer; n++)

/**
 * Compute the output layer activations.
 */
   int n = outputLayer;

   for (int k = 0; k < network[n]; k++)
   {
      Theta_temp = 0.0;

      for (int m = 0; m < network[n - 1]; m++)
         Theta_temp += a[n - 1][m] * W[n - 1][m][k];
      
      a[n][k] = f(Theta_temp);
      omega = T[k] - a[n][k];
      caseError += 0.5 * omega * omega;
   } // for (int k = 0; k < network[n]; k++)

   return caseError;
} // double runNetworkWithError()

/**
 * runNetwork runs the network by computing all activations. It does not calculate or return the case error.
 * This version of runNetwork is only used when running (not training), as it creates a temporary Theta
 * variable to use as an accumulator.
 * This function assumes that the input layer has already been set.
 */
void runNetwork()
{
   double Theta_temp;

   for (int n = FIRST_HIDDEN_LAYER; n <= outputLayer; n++)
   {
      for (int k = 0; k < network[n]; k++)
      {
         Theta_temp = 0.0;

         for (int m = 0; m < network[n - 1]; m++)
            Theta_temp += a[n - 1][m] * W[n - 1][m][k];
         
         a[n][k] = f(Theta_temp);
      } // for (int k = 0; k < network[n]; k++)
   } // for (int n = FIRST_HIDDEN_LAYER; n <= outputLayer; n++)

   return;
} // void runNetwork()

/**
 * For each iteration, the network loops over each test case and uses gradient descent with backpropagation
 * optimization to update the all weights. Training stops either when the maximum iteration limit is reached
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
         a[INPUT_LAYER] = inputs[test];
         T = truth[test];

         runNetworkForTraining();

         double Omega, Psi_temp;

         for (int n = lastHiddenLayer; n >= SECOND_HIDDEN_LAYER; n--)
         {
            for (int k = 0; k < network[n]; k++)
            {
               Omega = 0.0;

               for (int j = 0; j < network[n + 1]; j++)
               {
                  Omega += Psi[n + 1][j] * W[n][k][j];
                  W[n][k][j] += lambda * a[n][k] * Psi[n + 1][j];
               }

               Psi[n][k] = Omega * fDerivative(Theta[n][k]);
            } // for (int k = 0; k < network[n]; k++)
         } // for (int n = lastHiddenLayer; n >= SECOND_HIDDEN_LAYER; n--)

         int n = FIRST_HIDDEN_LAYER;

         for (int k = 0; k < network[n]; k++)
         {
            Omega = 0.0;

            for (int j = 0; j < network[n + 1]; j++)
            {
               Omega += Psi[n + 1][j] * W[n][k][j];
               W[n][k][j] += lambda * a[n][k] * Psi[n + 1][j];
            }

            Psi_temp = Omega * fDerivative(Theta[n][k]);

            for (int m = 0; m < network[n - 1]; m++)
               W[n - 1][m][k] += lambda * a[n - 1][m] * Psi_temp;
         } // for (int k = 0; k < network[n]; k++)

         totalError += runNetworkWithError();
      } // for (int test = 0; test < testCases; test++)

      averageError = totalError / (double) testCases;
      iterations++;

      if (keepAlive && !(iterations % keepAlive))
         printf("TRAINING UPDATE: Iteration %d, Error = %.17f\n", iterations, averageError);
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
         a[INPUT_LAYER] = inputs[test];
         T = truth[test];

         runNetwork();

         for (int i = 0; i < network[outputLayer]; i++)
            results[test][i] = a[outputLayer][i];
      } // for (int test = 0; test < testCases; test++)
   } // if (executionMode == TRAIN_NETWORK)
   else if (printTruthTables && !truthTableFile.empty()) // executionMode is RUN_NETWORK
   {
      double totalError = 0.0; // will be used to find averageError later

      for (int test = 0; test < testCases; test++)
      {
         a[INPUT_LAYER] = inputs[test];
         T = truth[test];

         totalError += runNetworkWithError();

         for (int i = 0; i < network[outputLayer]; i++)
            results[test][i] = a[outputLayer][i];
      } // for (int test = 0; test < testCases; test++)

      averageError = totalError / (double) testCases;
   } // if (executionMode == TRAIN_NETWORK) ... else if (printTruthTables && !truthTableFile.empty())
   else
   {
      for (int test = 0; test < testCases; test++)
      {
         a[INPUT_LAYER] = inputs[test];

         runNetwork();

         for (int i = 0; i < network[outputLayer]; i++)
            results[test][i] = a[outputLayer][i];
      } // for (int test = 0; test < testCases; test++)
   } // if (executionMode == TRAIN_NETWORK) ... else if (printTruthTables && !truthTableFile.empty()) ... else

   auto end = chrono::steady_clock::now();   // note ending time

   runtime = chrono::duration<double>(end - start).count() * MILLISECONDS_IN_SECOND;

   return;
} // void trainOrRun()

/**
 * Pretty prints a duration of milliseconds in a more digestible unit (up to hours).
 * Used by the reportResults function to print execution time.
 */
void prettyPrintTime(double milliseconds)
{
   cout << "Execution Time: ";

   if (milliseconds < MILLISECONDS_IN_SECOND)
      printf("%.3f ms", milliseconds);
   else
   {
      double seconds = milliseconds / MILLISECONDS_IN_SECOND;

      if (seconds < SECONDS_IN_MINUTE)
         printf("%.3f s", seconds);
      else
      {
         double minutes = seconds / SECONDS_IN_MINUTE;

         if (minutes < MINUTES_IN_HOUR)
            printf("%.3f min", minutes);
         else
            printf ("%.3f hr", minutes / MINUTES_IN_HOUR);
      } // if (seconds < SECONDS_IN_MINUTE) ... else
   } // if (milliseconds < MILLISECONDS_IN_SECOND) ... else

   cout << "\n\n";

   return;
} // void prettyPrintTime(double milliseconds)

/**
 * Prints results from the network's execution.
 * If the network was trained, explains the reason why training was stopped and prints the number
 * of iterations and average error. If the network was run, prints the average error.
 * Only prints truth tables if specified in the configuration parameters.
 */
void reportResults()
{
   if (keepAlive) cout << endl;
   
   cout << "RESULTS:\n";

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

   if (!truthTableFile.empty())
      cout << "Average Error: " << averageError << endl;

   prettyPrintTime(runtime);

   if (printTruthTables)
   {
      for (int test = 0; test < testCases; test++)
      {
         for (int i = 0; i < network[outputLayer]; i++)
            printf("%.3f ", results[test][i]);  // prints F[i] with 3 decimal places
         
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
} // int main(int argc, char** argv)
