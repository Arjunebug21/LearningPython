import matplotlib.pyplot as plt
import numpy as np

arr1 = np.zeros(11)
arr2 = np.zeros(11, dtype = complex)
n = 0
sum = 0

def ck_value(k):
    if k == 0:
        return 0.7
    else: 
        return (1 - np.exp(-1j*2*np.pi*k*0.7)) / (1j*2*np.pi*k)

def fourier_series(t, K, T=1):
    y = np.zeros_like(t, dtype=complex)
    for k in range(-K, K+1):  # Sum over harmonics
        y += ck_value(k) * np.exp(1j * k * 2 * np.pi * t / T)
    return y.real  # Take only the real part

# Define time values for one period
T = 1  # Period of the function
t = np.linspace(-T, T, 1000)  # Time range

# Plot Fourier series approximations for different K values
plt.figure(figsize=(8, 5))

def original_signal(t, T=1):
    t_mod = np.mod(t, T)  # Wrap time into [0, T] range
    return np.piecewise(t_mod, [t_mod < 0.7, t_mod >= 0.7], [1, 0])

for K in [2]:  # Different harmonic limits
    y = fourier_series(t, K, T)
    plt.plot(t, y, label=f"K = {K}")

# Plot the original signal
y_original = original_signal(t, T)
plt.plot(t, y_original, 'k--', linewidth=2, label="Original Signal")  # Dashed black line

plt.xlabel("Time (t)")
plt.ylabel("Function Value")
plt.title("Fourier Series Approximation vs. Original Signal")
plt.legend()
plt.grid()
plt.show()
# while sum <= 0.665:
#     sum = 0
#     for i in range (-n, n+1):
#         sum += (np.abs(ck_value(i)))**2
#     n += 1
#     print(sum)

#finalK = n-1

#print(finalK)

# for i in range (0,1,0.1):
#     if (i <= 0.7):
#         arr1[i] = 1
#     else:
#         arr1[i] = 0

