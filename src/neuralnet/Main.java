package neuralnet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;
import java.io.BufferedReader;
import java.io.InputStreamReader;

public class Main {
	public static final int TRAINING_DATA_SIZE = 3;
	public static final int TEST_DATA_SIZE = 4;
	public static final int INPUT_SIZE = 2;
	
	public static class Node
	{
		public float activation; 
		public Node() { activation = 0;}
		public float AcitvationFunction(float n)
		{
			return (float)( 1 /( 1 + Math.exp(-n) ));
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
		public Neuron[] hiddenLayer;
		public int hiddenLayerSize;
		public Output output;
		
		public NeuralNet(int inputSize, int hiddenLayerSize)
		{
			this.inputs = new Input[inputSize]; this.inputSize = inputSize;
			this.hiddenLayer = new Neuron[hiddenLayerSize]; this.hiddenLayerSize = hiddenLayerSize;
			
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
					hiddenLayer[i].bias[j] = (float)Math.random() - 0.5f;
					hiddenLayer[i].weight[j] = (float)Math.random() - 0.5f;
				}
			}
			for(int i = 0; i < hiddenLayerSize; i++) 
			{
				output.bias[i] = (float)Math.random() - 0.5f;
				output.weight[i] = (float)Math.random() - 0.5f;
			}
		}
		
		public void Learn(float[][] trainingData, float[] trainingOutput)
		{
			// TODO
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
		
		return data;
	}
	

	
	public static void main(String[] args) {
		
		// Read input data		
		InputData data = ReadData();
		// TODO: skalazas, normalizalas????
		
		// Create neural network
		NeuralNet nn = new NeuralNet(INPUT_SIZE, INPUT_SIZE);
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

