def run_model(data): 
    # Placeholder model 
    if sum(data) > 10: 
        return "High"
    else: 
        return "Low"

'''
Something will eventually go here that takes preprocessed data and returns a prediction. 

The rest of the pipeline shouldn't need to care about what model is being used. 
'''