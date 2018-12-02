package neuralnet;

import java.util.Scanner;

public class Main {
	public static final int TRAINING_DATA_SIZE = 10000;
	public static final int TEST_DATA_SIZE = 10;
	public static final int INPUT_SIZE = 3;
	/*public static final int TRAINING_DATA_SIZE = 3;
	public static final int TEST_DATA_SIZE = 4;
	public static final int INPUT_SIZE = 2;*/

	public static final float LEARNING_STEP = 0.2f;
	public static final float ACTIVATION_SAT_LIMIT = 1.0f;
	public static final float BIAS_HIDDEN_LAYER = 1.0f;
	public static final float BIAS_OUTPUT_LAYER = 1.0f;

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
		public Output outputLayer;
		public float outputLayerError;
		float output;
		float outputRange;

		public NeuralNet(int inputSize, int hiddenLayerSize)
		{
			this.inputs = new Input[inputSize]; this.inputSize = inputSize;
			this.hiddenLayer = new Neuron[hiddenLayerSize]; this.hiddenLayerSize = hiddenLayerSize;

			inputLayerError =  new float[hiddenLayerSize];
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
			outputLayer = new Output(hiddenLayerSize);

			for(int i = 0; i < hiddenLayerSize; i++) // initialize with random number between (-0.5; 0.5)
			{
				for(int j = 0; j < inputSize; j++) {
					hiddenLayer[i].bias[j] = BIAS_HIDDEN_LAYER;
					hiddenLayer[i].weight[j] = ((float)Math.random() * 2.0f - 1 ) * ACTIVATION_SAT_LIMIT;
				}
			}
			for(int i = 0; i < hiddenLayerSize; i++)
			{
				outputLayer.bias[i] = BIAS_OUTPUT_LAYER;
				outputLayer.weight[i] = ((float)Math.random() * 2.0f - 1 ) * ACTIVATION_SAT_LIMIT;
			}
		}

		public void Learn(float[][] trainingData, float[] trainingOutput, float range)
		{
			outputRange = range;
			try
			{
				for(int i = 0; i < TRAINING_DATA_SIZE; i++)
				{
					// Calculate output (forward pass)
					Execute(trainingData[i], inputSize);

					// Calculate signal errors ------------------------------------------------------------------------------------
					// Calculate error for the output layer
					float dOut = -2.0f * ((trainingOutput[i] / outputRange) - outputLayer.activation) * outputLayer.AcitvationFunctionDeriv(outputLayer.activation);

					// Calculate error for the hidden layer
					for(int j = 0; j < hiddenLayerSize; j++)
					{
						hiddenLayerError[j] = dOut * hiddenLayer[j].activation;
					}

					// Calculate error for the input layer
					for(int j = 0; j < hiddenLayerSize; j++)
					{
						for(int k = 0; k < inputSize; k++)
						{
							inputLayerError[j] = dOut * outputLayer.weight[j] * hiddenLayer[j].AcitvationFunctionDeriv(hiddenLayer[j].activation);
						}
					}
					// Backpropagate errors ------------------------------------------------------------------------------------

					for(int j = 0; j < hiddenLayerSize; j++)
					{
						outputLayer.weight[j] = outputLayer.weight[j] - hiddenLayerError[j] * LEARNING_STEP;
					}

					for(int j = 0; j < hiddenLayerSize; j++)
					{
						for(int k = 0; k < inputSize; k++)
						{
							hiddenLayer[j].weight[k] = hiddenLayer[j].weight[k] - inputLayerError[j] * inputs[k].activation * LEARNING_STEP;
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

			outputLayer.Execute(hiddenLayer, hiddenLayerSize);	// run output layer neuron

			output = outputLayer.activation * outputRange;
			return output;
		}
	}

	public static class MinMax
	{
		public float Min; public float Max;
		public MinMax() {Min = Float.MAX_VALUE; Max = -Float.MAX_VALUE; }
	}

	public static class InputData
	{
		public float[][] trainingData;
		public float[] trainingOutput;
		float outputRange;
		public float[][] testData;
		public InputData(int trainingDataSize, int testDataSize, int dataInputSize) { testData = new float[testDataSize][dataInputSize];
			trainingData = new float[trainingDataSize][dataInputSize]; trainingOutput = new float[trainingDataSize]; }
	}

	public static MinMax[] getInputMinMax(float[][] data)
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

	public static float getOutputRange(float[] data)
	{
		MinMax mm = new MinMax();
		for(int i = 0; i < TRAINING_DATA_SIZE; i++)
		{
			if(data[i] > mm.Max)
			{
				mm.Max = data[i];
			}
			if(data[i] < mm.Min)
			{
				mm.Min = data[i];
			}
		}
		return mm.Max - mm.Min;
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
		MinMax[] minmax = getInputMinMax(data.trainingData);

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
		}

		data.outputRange = getOutputRange(data.trainingOutput);

		return data;
	}



	public static void main(String[] args) {

		// Read and normalize input data
		InputData data = ReadData();

		// Create neural network
		NeuralNet nn = new NeuralNet(INPUT_SIZE, INPUT_SIZE);
		Float[] outputs = new Float[TEST_DATA_SIZE];

		//Train graph
		nn.Learn(data.trainingData, data.trainingOutput, data.outputRange);

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

