import numpy as np
from scipy.stats import gamma

def calculate_credible_interval(data, a, b, beta, confidence_level=0.95):

    n = len(data)
  

    a_prime = n + a
    b_prime = b + np.sum(np.power(data, beta))
    
    alpha = 1 - confidence_level
    lower_tail = alpha / 2      
    upper_tail = 1 - (alpha / 2) 
    
   
    L = gamma.ppf(lower_tail, a_prime, scale=1/b_prime)
    U = gamma.ppf(upper_tail, a_prime, scale=1/b_prime)
    
    return L, U, a_prime, b_prime

sample_data = np.array([1.2,0.8,1.5,2]) 

a_prior = 1       
b_prior = 1      
beta_known = 2 

L, U, post_a, post_b = calculate_credible_interval(sample_data, a_prior, b_prior, beta_known)

print(f"Posterior Parameters: a' = {post_a}, b' = {post_b:.4f}")
print(f"95% Bayesian Credible Interval for θ:")
print(f"L (Lower Limit) = {L:.4f}")
print(f"U (Upper Limit) = {U:.4f}")