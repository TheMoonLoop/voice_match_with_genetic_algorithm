# Codigo de ejemplo de uso, antes de run_ga()

fs_in, input_signal = load_WavData("input.wav")

# Cargar referencias desde carpeta
refs_dir = "refs"
reference_signals = []
reference_names = []

for filename in os.listdir(refs_dir):
    if filename.lower().endswith(".wav"):
        path = os.path.join(refs_dir, filename)
        try:
            fs, sig = load_WavData(path)
            reference_signals.append(sig)
            reference_names.append(filename)
        except Exception as e:
            print(f"Error cargando {filename}: {e}")

if len(reference_signals) < 2:
    raise RuntimeError("Se necesitan al menos 2 archivos de referencia para el GA")

# Alinear referencias
reference_signals = np.array(reference_signals)

# Alinear input
input_signal = input_signal[:len(reference_signals[0])]

print("\nChequeo directo:")
for i, ref in enumerate(reference_signals):
    same = same_voice(ref, input_signal)
    print(f"{reference_names[i]} → same_voice = {same}")

fitness = population_fitness(reference_signals, input_signal)

print("\nFitness:")
for name, f in zip(reference_names, fitness):
    print(f"{name}: {f:.6f}")

idx = roulette_selection(fitness, 2)
p1 = reference_signals[idx[0]]
p2 = reference_signals[idx[1]]

print("\nPadres seleccionados:")
print(reference_names[idx[0]])
print(reference_names[idx[1]])

child1, child2, cp = one_point_crossover(p1, p2)

best_child, best_f = select_best_offspring(child1, child2, input_signal)

print("\nCrossover point:", cp)
print("Best offspring fitness:", best_f)

# Mutación
mutated_child, mp = mutate_offspring(best_child)

# Fitness del mutado
mutated_fitness = mutated_offspring_fitness(
    mutated_child, input_signal
)

print("\nCrossover point:", cp)
print("Mutation point:", mp)
print("Fitness before mutation:", best_f)
print("Fitness after mutation:", mutated_fitness)

if mutated_fitness > best_f:
    current_best = mutated_child
    current_fitness = mutated_fitness
else:
    current_best = best_child
    current_fitness = best_f

# a) Threshold comparison
same_thresh = same_voice(current_best, input_signal)

# b) Euclidean comparison
same_euc, dist = same_voice_euclidean(
    current_best,
    input_signal,
    distance_threshold=1e7  # AJUSTAR EXPERIMENTALMENTE
)

print("\nThreshold match:", same_thresh)
print("Euclidean distance:", dist)
print("Euclidean match:", same_euc)

if same_thresh and same_euc:
    decision = "ACCEPT"
else:
    decision = "REJECT"

print("Final decision:", decision)
