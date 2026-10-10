import numpy as np
from scipy.stats import gamma

def calculate_credible_interval(data, a, b, beta, confidence_level):

    n = len(data)
    
    a_prime = n + a
    b_prime = b + np.sum(np.power(data, beta))
    
    alpha = 1 - confidence_level
    lower_tail = alpha / 2      
    upper_tail = 1 - (alpha / 2) 
    
    L = gamma.ppf(lower_tail, a_prime, scale=1/b_prime)
    U = gamma.ppf(upper_tail, a_prime, scale=1/b_prime)
    
    return L, U, a_prime, b_prime

def main():
    
    try:
        
        a_prior = float(input("Enter value of a: "))
        b_prior = float(input("Enter value of b: "))
        beta_known = float(input("Enter value of Beta: "))
        
        
        conf_input = float(input("Enter Confidence interval (for example 0.95): "))
        
        if conf_input > 1:
            conf_input = conf_input / 100
        
        data_input = input("Enter Sample Data(seperate between each line with space): ")
        sample_data = np.array([float(x) for x in data_input.split()])

        if len(sample_data) == 0:
            print("Error your must enter Sample data")
            return

    
        L, U, post_a, post_b = calculate_credible_interval(sample_data, a_prior, b_prior, beta_known, conf_input)

        
        print("\n" + "="*40)
        print(f"Final Results of Confidence interval {conf_input*100}%:")
        print(f"Credible Interval for θ: [{L:.6f} < θ < {U:.6f}]")
        print("="*40)

    except ValueError:
        print("Enter Correct Numbers :-(")

if __name__ == "__main__":
    main()