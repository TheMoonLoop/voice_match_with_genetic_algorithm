import numpy as np
import random
import os
from scipy.io.wavfile import write
from scipy.io import wavfile

def load_WavData(path):
    sample_rate, data = wavfile.read(path)
    if len(data.shape) != 2 or data.shape[1] != 2:
        raise ValueError("El archivo no es estéreo (2 canales).")

    channel_index = random.choice([0, 1])
    signal = data[:, channel_index]
    return sample_rate, signal

def voice_similarity(signal_a, signal_b, threshold=0.09):
    if len(signal_a) != len(signal_b):
        raise ValueError("Las señales deben tener la misma longitud")

    diff = np.abs(signal_a - signal_b)
    matches = diff < threshold
    count = np.sum(matches)
    ratio = count / len(signal_a)
    return count, ratio

def same_voice(signal_a, signal_b, threshold=0.09):
    _, ratio = voice_similarity(signal_a, signal_b, threshold)
    return ratio < 0.5

def population_fitness(reference_signals, input_signal, threshold=0.09):
    fitness_values = []
    for ref in reference_signals:
        _, ratio = voice_similarity(ref, input_signal, threshold)
        fitness = ratio ** 2 # f(x) = x^2
        fitness_values.append(fitness)

    return np.array(fitness_values)

def roulette_selection(fitness_values, num_select=2):
    total_fitness = np.sum(fitness_values)
    if total_fitness == 0:
        probs = np.ones(len(fitness_values)) / len(fitness_values)
    else:
        probs = fitness_values / total_fitness

    selected_indices = np.random.choice( len(fitness_values), size=num_select, replace=False, p=probs)
    return selected_indices

def one_point_crossover(parent1, parent2):
    n = len(parent1)
    cp = int(np.log2(n + 1))
    if cp <= 0 or cp >= n:
        cp = n // 2 

    child1 = np.concatenate([parent1[:cp], parent2[cp:]])
    child2 = np.concatenate([parent2[:cp], parent1[cp:]])

    return child1, child2, cp

def select_best_offspring(child1, child2, input_signal, threshold=0.09):
    _, r1 = voice_similarity(child1, input_signal, threshold)
    _, r2 = voice_similarity(child2, input_signal, threshold)

    f1 = r1 ** 2
    f2 = r2 ** 2

    if f1 >= f2:
        return child1, f1
    else:
        return child2, f2

def mutate_offspring(offspring):
    n = len(offspring)
    mp = int(np.log2(n + 1))
    if mp <= 0 or mp >= n:
        mp = n // 2

    mutated = offspring.copy()
    mutated[mp] = -mutated[mp]
    return mutated, mp

def mutated_offspring_fitness(mutated, input_signal, threshold=0.09):
    _, ratio = voice_similarity(mutated, input_signal, threshold)
    return ratio ** 2

def normalize_signal(sig):
    sig = sig.astype(np.float32)
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = sig / max_val
    return sig

def save_wav(path, sample_rate, signal):
    max_val = np.max(np.abs(signal))
    if max_val == 0:
        return signal.astype(np.int16)
    scaled = signal / max_val
    signal_int16 = np.int16(scaled * 32767)
    write(path, sample_rate, signal_int16)

def run_ga(reference_signals, input_signal, max_generations=50, improvement_tol=1e-6):
    fitness = population_fitness(reference_signals, input_signal)
    print("\nFitness:")
    for name, f in zip(reference_names, fitness):
        print(f"{name}: {f:.6f}")

    # Mejor referencia inicial
    best_idx = np.argmax(fitness)
    current_best = reference_signals[best_idx]
    current_best_fitness = fitness[best_idx]

    # Ciclo del GA
    for gen in range(max_generations):
        idx = roulette_selection(fitness, 2)
        p1 = reference_signals[idx[0]]
        p2 = reference_signals[idx[1]]

        # Crossover
        child1, child2, cp = one_point_crossover(p1, p2)
        best_child, best_child_fitness = select_best_offspring(child1, child2, input_signal)

        # Mejoro?
        if best_child_fitness > current_best_fitness + improvement_tol:
            current_best = best_child
            current_best_fitness = best_child_fitness

        # Mutacion
        mutated_child, mp = mutate_offspring(current_best)
        mutated_fitness = mutated_offspring_fitness(mutated_child, input_signal)

        # Mejoro con mutacion?
        if mutated_fitness > current_best_fitness + improvement_tol:
            current_best = mutated_child
            current_best_fitness = mutated_fitness

    print("\nPadres seleccionados:")
    print(reference_names[idx[0]])
    print(reference_names[idx[1]])
    print("\nCrossover point:", cp)
    print("Best offspring fitness:", best_child_fitness)
    print("Mutation point:", mp)
    print("Fitness after mutation:", mutated_fitness)
    return current_best, current_best_fitness

if __name__ == "__main__":
    # 1. Cargar inputs
    inputs = {}
    for name in ["input1.wav", "input2.wav"]:
        fs, sig = load_WavData(name)
        sig = normalize_signal(sig)
        inputs[name] = sig

    # 2. Cargar referencias
    refs_dir = "refs"
    reference_signals = []
    reference_names = []

    for filename in os.listdir(refs_dir):
        if filename.lower().endswith(".wav"):
            path = os.path.join(refs_dir, filename)
            try:
                fs, sig = load_WavData(path)
                sig = normalize_signal(sig)
                reference_signals.append(sig)
                reference_names.append(filename)
            except Exception as e:
                print(f"Error cargando {filename}: {e}")

    if len(reference_signals) < 2:
        raise RuntimeError("Se necesitan al menos 2 archivos de referencia")

    # 3. Procesar cada input por separado
    for input_name, input_signal in inputs.items():
        print("\n" + "=" * 50)
        print(f"Procesando {input_name}")
        print("=" * 50)

        # Ajuste de longitudes
        min_len = min(len(input_signal), *[len(ref) for ref in reference_signals])
        input_aligned = input_signal[:min_len]
        refs_aligned = np.array([ref[:min_len] for ref in reference_signals])
        print(f"Longitud usada: {min_len} muestras")

        # 4. Ejecutar GA
        best_sound, best_fitness = run_ga(refs_aligned,input_aligned,max_generations=50)
        print("\nResultado GA:")
        print("Fitness final:", best_fitness)

        # 5. Comparación final (paper 3.3)
        # (a) Threshold
        same_thr = same_voice(best_sound, input_aligned)

        # (b) Distancia euclidiana
        euclid_dist = np.linalg.norm(best_sound - input_aligned)

        print("\nComparación final:")
        print("Threshold match:", same_thr)
        print("Euclidean distance:", euclid_dist)

        if same_thr:
            print("MISMA VOZ")
        else:
            print("NO ES LA MISMA VOZ")

    output_name = f"ga_result_{input_name.replace('.wav','')}.wav"
    save_wav(output_name, fs, best_sound)
    print(f"\nAudio GA guardado como: {output_name}")
