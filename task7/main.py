from input import receive_input
from preprocessing import preprocess_input
from model import run_model
from output import process_output

def main(): 
    # Step 1: Receive input data
    data = receive_input()

    # Step 2: Preprocess the input data
    processed_data = preprocess_input(data)

    # Step 3: Run the model on the preprocessed data
    prediction = run_model(processed_data)

    # Step 4: Process and display the output
    output = process_output(prediction)
    print(output)


if __name__ == "__main__":
    main()

