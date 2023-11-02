
## fixed step

xs, taus = gradient_descent_fixed(lst.f, lst.x0.to(device), lst.y.to(device), 2.0334, max_iter=100)
fs = correct_fs([lst.f(x, lst.y.to(device)) for x in xs])
fs

minimum = np.nanmin(fs_nonlearn2[0])
data1 = [i - minimum for i in fs if not np.isnan(i)]
data2 = [i - minimum for i in fs_nonlearn2[0][:101] if not np.isnan(i)]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title('Loglog plot of $f(x_k) - f(x^*)$ for an example function')
plt.xlabel('Iteration Number')  # Replace with your actual label
plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label

# Add occasional points for clarity
plt.loglog(data1, label='Learned step size', marker='o', markersize=5, linestyle='-')
plt.loglog(data2, label='2/($\mu$+L)', marker='s', markersize=5, linestyle='-')

# Show legend
plt.legend()

# Show the plot
plt.show()









## function and backtracking


# Calculate the minimum for valid data
minimum = np.nanmin(fs_bt)

# Calculate data arrays
data1 = [item - minimum for item in fs_func]
data2 = [item - minimum for item in fs_bt]
data3 = [item - minimum for item in fs22]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title('Loglog plot of $f(x_k) - f(x^*)$ for an example function')
plt.xlabel('Iteration Number')  # Replace with your actual label
plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label

# Add occasional points for clarity
plt.loglog(data1, label='Learned function', marker='o', markersize=5, linestyle='-')
plt.loglog(data2, label='Backtracking', marker='s', markersize=5, linestyle='-')
plt.loglog(data3, label='2/($\mu$+L)', marker='s', markersize=5, linestyle='-')

# Show legend
plt.legend()

# Show the plot
plt.show()








##################### FUNCTION VS BACKTRACKING PLOTS ######################

# LOOK AT FUNCTION VS FIXED BOTH 2/ AND 1.99

i=0
noise = round(float(noise_list[i]),4)
blur = blur_list[i]
minimum = np.nanmin(fs_bt_list[i])
# Calculate data arrays
data1 = [item - minimum for item in fs_func_list[i]]
data2 = [item - minimum for item in fs_bt_list[i]]
data3 = [item - minimum for item in fs_nonlearn2[i]]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title(f'Loglog plot of $f(x_k) - f(x^*)$, $\sigma_n$ = {noise}, $\sigma$ = {blur}')
plt.xlabel('Iteration Number')  # Replace with your actual label
plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
# Add occasional points for clarity
plt.loglog(data1, label='Learned function', marker='o', markersize=1, linestyle='-')
plt.loglog(data2, label='Backtracking', marker='o', markersize=1, linestyle='-')
plt.loglog(data3, label='2/($\mu$+L)', marker='s', markersize=1, linestyle='-')
# Show legend
plt.legend()
# Show the plot
plt.show()


# THEN LOOK AT VS BACKTRACKING, DO WORK CASE AND BEST CASE

i=0
noise = round(float(noise_list[i]),4)
blur = blur_list[i]
minimum = np.nanmin(fs_bt_list[i])
# Calculate data arrays
data1 = [item - minimum for item in fs_func_list[i]]
data2 = [item - minimum for item in fs_nonlearn2[i]]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title(f'Loglog plot of $f(x_k) - f(x^*)$, $\sigma_n$ = {noise}, $\sigma$ = {blur}')
plt.xlabel('Iteration Number')  # Replace with your actual label
plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
# Add occasional points for clarity
plt.loglog(data1, label='Learned function', marker='o', markersize=1, linestyle='-')
plt.loglog(data2, label='2/($\mu$+L)', marker='s', markersize=1, linestyle='-')
# Show legend
plt.legend()
# Show the plot
plt.show()



# PLOT TAUS

i=0
noise = round(float(noise_list[i]),4)
blur = blur_list[i]
# Calculate data arrays
data1 = taus_func_list[i]
data2 =taus_bt_list[i]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title(fr'\tau vs Iteration Number - Learned Function, $\sigma_n$ = {noise}, $\sigma$ = {blur}')
plt.xlabel('Iteration Number')
plt.ylabel(r'\tau_k')
plt.plot(data1, marker='o', markersize=1, linestyle='-')
plt.show()

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title(rf'\tau vs Iteration Number - Backtracking, $\sigma_n$ = {noise}, $\sigma$ = {blur}')
plt.xlabel('Iteration Number')
plt.ylabel(r'\tau_k')
plt.plot(data2, marker='o', markersize=1, linestyle='-')
plt.show()


################# MOMENTUM PLOTS ######################


## COMPARE MOMENTUM TO BT AND FUNCTION, BEST/WORST.

i=0
noise = round(float(noise_list[i]),4)
blur = blur_list[i]
minimum = np.nanmin(fs_bt_list[i])
# Calculate data arrays
data1 = [item - minimum for item in fs_fixed_heavy[i]]
data2 = [item - minimum for item in fs_bt_list[i]]
data3 = [item - minimum for item in fs_nonlearn2[i]]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title(f'Loglog plot of $f(x_k) - f(x^*)$, $\sigma_n$ = {noise}, $\sigma$ = {blur}')
plt.xlabel('Iteration Number')  # Replace with your actual label
plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
# Add occasional points for clarity
plt.loglog(data1, label='Heavy Ball', marker='o', markersize=1, linestyle='-')
plt.loglog(data2, label='Backtracking', marker='o', markersize=1, linestyle='-')
plt.loglog(data3, label='2/($\mu$+L)', marker='s', markersize=1, linestyle='-')
# Show legend
plt.legend()
# Show the plot
plt.show()



## ALSO COMPARE MOMENTUM TO FIXED TO FUNCTION, BEST/WORST


i=0
noise = round(float(noise_list[i]),4)
blur = blur_list[i]
minimum = np.nanmin(fs_bt_list[i])
# Calculate data arrays
data1 = [item - minimum for item in fs_fixed_heavy[i]]
data3 = [item - minimum for item in fs_heavy_func[i]]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title(f'Loglog plot of $f(x_k) - f(x^*)$, $\sigma_n$ = {noise}, $\sigma$ = {blur}')
plt.xlabel('Iteration Number')  # Replace with your actual label
plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
# Add occasional points for clarity
plt.loglog(data1, label='Heavy Ball Fixed', marker='o', markersize=1, linestyle='-')
plt.loglog(data3, label='Heavy Ball Function', marker='s', markersize=1, linestyle='-')
# Show legend
plt.legend()
# Show the plot
plt.show()






### TAUS AND BETAS

i=0
noise = round(float(noise_list[i]),4)
blur = blur_list[i]
# Calculate data arrays
data1 = taus_hb_list[i]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title(f'\tau vs Iteration Number - Learned Function, $\sigma_n$ = {noise}, $\sigma$ = {blur}')
plt.xlabel('Iteration Number')
plt.ylabel(f'\tau_k')
plt.plot(data1, marker='o', markersize=1, linestyle='-')
plt.show()

data2 = betas_hb_list[i]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title(f'\tau vs Iteration Number - Backtracking, $\sigma_n$ = {noise}, $\sigma$ = {blur}')
plt.xlabel('Iteration Number')
plt.ylabel(f'\tau_k')
plt.plot(data2, marker='o', markersize=1, linestyle='-')
plt.show()






########################### MODEL-FREE PLOTS ##############################


### JUST SHOW VS 2/MU=L


i=0
noise = round(float(noise_list[i]),4)
blur = blur_list[i]
minimum = np.nanmin(fs_bt_list[i])
# Calculate data arrays
data1 = [item - minimum for item in fs_correction_list[i]]
data2 = [item - minimum for item in fs_nonlearn2[i]]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title(f'Loglog plot of $f(x_k) - f(x^*)$, $\sigma_n$ = {noise}, $\sigma$ = {blur}')
plt.xlabel('Iteration Number')  # Replace with your actual label
plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
# Add occasional points for clarity
plt.loglog(data1, label='Learned Update "Model-free"', marker='o', markersize=1, linestyle='-')
plt.loglog(data2, label='2/($\mu$+L)', marker='s', markersize=1, linestyle='-')
# Show legend
plt.legend()
# Show the plot
plt.show()





######### ROBUSTNESS PLOTS #################

## DOES THE LEARNED FUNCTION STILL PERFORM BETTER THAN FIXED?

i=0
noise = round(float(noise_list[i]),4)
blur = blur_list[i]
minimum = np.nanmin(fs_nonlearn2[i])
# Calculate data arrays
data1 = [item - minimum for item in fs_func_list[i]]
data3 = [item - minimum for item in fs_nonlearn2[i]]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title(f'Loglog plot of $f(x_k) - f(x^*)$, $\sigma_n$ = {noise}, $\sigma$ = {blur}')
plt.xlabel('Iteration Number')  # Replace with your actual label
plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
# Add occasional points for clarity
plt.loglog(data1, label='Learned function', marker='o', markersize=1, linestyle='-')
plt.loglog(data3, label='2/($\mu$+L)', marker='s', markersize=1, linestyle='-')
# Show legend
plt.legend()
# Show the plot
plt.show()

## DOES THE MODEL-FREE BLOW UP QUICKER?



i=0
noise = round(float(noise_list[i]),4)
blur = blur_list[i]
minimum = np.nanmin(fs_bt_list[i])
# Calculate data arrays
data1 = [item - minimum for item in fs_correction_list[i]]
data2 = [item - minimum for item in fs_nonlearn2[i]]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.title(f'Loglog plot of $f(x_k) - f(x^*)$, $\sigma_n$ = {noise}, $\sigma$ = {blur}')
plt.xlabel('Iteration Number')  # Replace with your actual label
plt.ylabel('$f(x_k) - f(x^*)$')  # Replace with your actual label
# Add occasional points for clarity
plt.loglog(data1, label='Learned Update "Model-free"', marker='o', markersize=1, linestyle='-')
plt.loglog(data2, label='2/($\mu$+L)', marker='s', markersize=1, linestyle='-')
# Show legend
plt.legend()
# Show the plot
plt.show()





i=0
noise = round(float(noise_list[i]),4)
blur = blur_list[i]
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.xlabel('Iteration Number')  # Replace with your actual label
plt.ylabel('$f(x_k^{func}) - f(x_k^{fixed})$')  # Replace with your actual label
# Add occasional points for clarity
plt.loglog([fs_func[i] - fs_nonlearn2[0][i] for i in range(len(fs_func))])
# Show the plot
plt.show()




