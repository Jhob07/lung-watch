import torch

def inspect_model(model_path):
    try:
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Print the keys in the state dictionary
        print("\nModel state dictionary keys:")
        for key in state_dict.keys():
            print(f"- {key}")
            
        # Print the shape of each tensor
        print("\nTensor shapes:")
        for key, tensor in state_dict.items():
            print(f"- {key}: {tensor.shape}")
            
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")

if __name__ == "__main__":
    model_path = "src/Models/new_model_final.pth"
    inspect_model(model_path) 