package neuralnet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {
	public static final int TRAINING_DATA_SIZE = 10000;
	public static final int TEST_DATA_SIZE = 10
			;
	public static final int INPUT_SIZE = 3
			;
	public static final float LEARNING_STEP = 0.1f;
	public static final float ACTIVATION_SAT_LIMIT = 1.73f;

	public static class Node
	{
		public float activation;
		public Node() { activation = 0;}
		public float AcitvationFunction(float n)
		{
			return (float)( 1 /( 1 + Math.exp(-n) ));
		}
		public float AcitvationFunctionDeriv(float n)
		{
			return n * (1 - n);
		}
	}

	public static class Input extends Node
	{
		public Input() { super(); }
	}

	public static class Neuron extends Node
	{
		public float [] weight;
		public float [] bias;
		public Neuron(int inputSize) {	this.weight = new float[inputSize]; this.bias = new float[inputSize]; }

		public void Execute(Node[] inputLayer, int inputLayerSize)
		{
			activation = 0;
			for(int i = 0; i < inputLayerSize; i++)
			{
				activation += weight[i] * inputLayer[i].activation + bias[i];
			}
			activation = AcitvationFunction(activation);
		}
	}

	public static class Output extends Neuron
	{
		public Output(int inputSize) {	super(inputSize); }
	}


	public static class NeuralNet
	{
		public Input[] inputs;
		public int inputSize;
		public float[] inputLayerError;
		public Neuron[] hiddenLayer;
		public int hiddenLayerSize;
		public float[] hiddenLayerError;
		public Output output;
		public float outputLayerError;

		public NeuralNet(int inputSize, int hiddenLayerSize)
		{
			this.inputs = new Input[inputSize]; this.inputSize = inputSize;
			this.hiddenLayer = new Neuron[hiddenLayerSize]; this.hiddenLayerSize = hiddenLayerSize;
			
			inputLayerError =  new float[inputSize];
			hiddenLayerError = new float[hiddenLayerSize];
			outputLayerError = 0;

			for(int i = 0; i < inputSize; i++)
			{
				inputs[i] = new Input();
			}

			for(int i = 0; i < hiddenLayerSize; i++)
			{
				hiddenLayer[i] = new Neuron(inputSize);
			}
			output = new Output(hiddenLayerSize);

			for(int i = 0; i < hiddenLayerSize; i++) // initialize with random number between (-0.5; 0.5)
			{
				for(int j = 0; j < inputSize; j++) {
					hiddenLayer[i].bias[j] = 0;
					hiddenLayer[i].weight[j] = ((float)Math.random() * 2.0f - 1 ) * ACTIVATION_SAT_LIMIT;
				}
			}
			for(int i = 0; i < hiddenLayerSize; i++)
			{
				output.bias[i] = 0;
				output.weight[i] = ((float)Math.random() * 2.0f - 1 ) * ACTIVATION_SAT_LIMIT;
			}
		}

		public void Learn(float[][] trainingData, float[] trainingOutput)
		{
			try
			{
				for(int i = 0; i < TRAINING_DATA_SIZE; i++)
				{
					// Calculate output
					Execute(trainingData[i], inputSize);

					// Calculate signal errors ------------------------------------------------------------------------------------
					// Calculate error for the output layer
					float diff = (trainingOutput[i] - output.activation);
					outputLayerError = diff*diff * output.AcitvationFunctionDeriv(output.activation);
					
					float sumError = 0;
					// Calculate error for the hidden layer
					for(int j = 0; j < hiddenLayerSize; j++)
					{
						sumError = output.weight[j]* outputLayerError;
						
						hiddenLayerError[j] = hiddenLayer[j].AcitvationFunctionDeriv(hiddenLayer[j].activation) * sumError;
					}
					// Calculate error for the input layer
					for(int j = 0; j < inputSize; j++)
					{
						sumError = 0;
						for(int k = 0; k < hiddenLayerSize; k++)
						{
							sumError += hiddenLayer[k].weight[j]* hiddenLayerError[k];
						}
						inputLayerError[j] = inputs[j].AcitvationFunctionDeriv(inputs[j].activation) * sumError;
					}
					// Backpropagate errors ------------------------------------------------------------------------------------
					
					for(int j = 0; j < hiddenLayerSize; j++)
					{
						output.weight[j] = output.weight[j] + hiddenLayerError[j] * LEARNING_STEP;
					}
					
					for(int j = 0; j < hiddenLayerSize; j++)
					{
						for(int k = 0; k < inputSize; k++)
						{
							hiddenLayer[j].weight[k] = hiddenLayer[j].weight[k] + inputLayerError[j] * LEARNING_STEP;
						}
					}
					
				}
			}
			catch(Exception e){
				System.err.println("Error during learning: " + e.toString());
			}
		}
		public float Execute(float[] inputData, int inputDataSize) throws Exception
		{
			if(inputDataSize != inputSize)	throw new Exception("Incorrect input size!");

			for (int i = 0; i < inputSize; i++) { 			// read input
				inputs[i].activation = inputData[i];
			}

			for (int i = 0; i < hiddenLayerSize; i++) { 	// run hidden layer neurons
				hiddenLayer[i].Execute(inputs, inputSize);
			}

			output.Execute(hiddenLayer, hiddenLayerSize);	// run output layer neuron
			return output.activation;
		}
	}

	public static class InputData
	{
		public float[][] trainingData;
		public float[] trainingOutput;
		public float[][] testData;
		public InputData(int trainingDataSize, int testDataSize, int dataInputSize) { testData = new float[testDataSize][dataInputSize];
			trainingData = new float[trainingDataSize][dataInputSize]; trainingOutput = new float[trainingDataSize]; }
	}
	
	public static class MinMax
	{
		public float Min; public float Max;
		public MinMax() {Min = Float.MAX_VALUE; Max = -Float.MAX_VALUE; }
	}
	
	public static MinMax[] getMinMax(float[][] data)
	{
		MinMax[] values = new MinMax[INPUT_SIZE];
		for (int j = 0; j < INPUT_SIZE; j++)
		{
			values[j] = new MinMax();
		}
		for(int i = 0; i < TRAINING_DATA_SIZE; i++)
		{
			for (int j = 0; j < INPUT_SIZE; j++)
			{
				if(data[i][j] > values[j].Max)
				{
					values[j].Max = data[i][j];
				}
				if(data[i][j] < values[j].Min)
				{
					values[j].Min = data[i][j];
				}
			}
		}
		
		return values;
	}

	public static InputData ReadData()
	{
		InputData data = new InputData(TRAINING_DATA_SIZE, TEST_DATA_SIZE, INPUT_SIZE);
		Scanner scanner = new Scanner(System.in);

		try
		{
			for(int i = 0; i < TRAINING_DATA_SIZE; i++)
			{
				String line = scanner.nextLine();
				String[] array = line.split("\t", -1);

				for (int j = 0; j < INPUT_SIZE; j++)
				{
					data.trainingData[i][j] = Float.parseFloat(array[j]);
				}
			}
			for(int i = 0; i < TRAINING_DATA_SIZE; i++)
			{
				String line = scanner.nextLine();
				data.trainingOutput[i] = Float.parseFloat(line);
			}
			for(int i = 0; i < TEST_DATA_SIZE; i++)
			{
				String line = scanner.nextLine();
				String[] array = line.split("\t", -1);

				for (int j = 0; j < INPUT_SIZE; j++)
				{
					data.testData[i][j] = Float.parseFloat(array[j]);
				}
			}
			scanner.close();
		}
		catch(NumberFormatException nfe){
			scanner.close();
			System.err.println("Invalid input format!");
		}
		
		// Normalize inputs
		/*MinMax[] minmax = getMinMax(data.trainingData);
		for(int i = 0; i < TRAINING_DATA_SIZE; i++)
		{
			for (int j = 0; j < INPUT_SIZE; j++)
			{
				data.trainingData[i][j] = (data.trainingData[i][j] - minmax[j].Min) / (minmax[j].Max - minmax[j].Min) * ACTIVATION_SAT_LIMIT;
			}
		}
		
		for(int i = 0; i < TEST_DATA_SIZE; i++)
		{
			for (int j = 0; j < INPUT_SIZE; j++)
			{
				data.testData[i][j] = (data.testData[i][j] - minmax[j].Min) / (minmax[j].Max - minmax[j].Min) * ACTIVATION_SAT_LIMIT;
			}
		}*/
		
		return data;
	}



	public static void main(String[] args) {

		// Read and normalize input data
		InputData data = ReadData();

		// Create neural network
		NeuralNet nn = new NeuralNet(INPUT_SIZE, INPUT_SIZE - 1);
		Float[] outputs = new Float[TEST_DATA_SIZE];

		//Train graph
		nn.Learn(data.trainingData, data.trainingOutput);

		try
		{
			// Execute neural network on the test data
			for(int i = 0; i < TEST_DATA_SIZE; i++)
			{
				outputs[i] = nn.Execute(data.testData[i], INPUT_SIZE);
			}
		} catch(Exception e){
			System.err.println("Error: " + e.toString());
		}


		// Write output
		for(int i = 0; i < TEST_DATA_SIZE; i++)
		{
			System.out.println(outputs[i].toString());
		}
	}
}

