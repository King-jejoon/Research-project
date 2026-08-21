import numpy as np
# Activation Functions
class TanhActivation:
    def forward(self, x):
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

class LinearActivation:
    def forward(self, x):
        self.output = x
        return self.output
    
    def backward(self, grad_output):
        return grad_output

# Dense Layer
class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.bias = np.zeros((1, output_size))
        self.grad_weights = None
        self.grad_bias = None
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, grad_output):
        self.grad_weights = np.dot(self.input.T, grad_output)
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)
        return grad_input

# Neural Network Model
class NeuralDataFusionModel:
    def __init__(self, input_dim):
        self.layer1 = DenseLayer(input_dim, 4)
        self.activation1 = TanhActivation()
        self.layer2 = DenseLayer(4, 2)
        self.activation2 = TanhActivation()
        self.layer3 = DenseLayer(2, 1)
        self.activation3 = LinearActivation()
        
        self.parameters = [
            self.layer1.weights, self.layer1.bias,
            self.layer2.weights, self.layer2.bias,
            self.layer3.weights, self.layer3.bias
        ]
        self.gradients = []
    
    def forward(self, x):
        out = self.layer1.forward(x)
        out = self.activation1.forward(out)
        out = self.layer2.forward(out)
        out = self.activation2.forward(out)
        out = self.layer3.forward(out)
        out = self.activation3.forward(out)
        return out
    
    def backward(self, grad_output):
        grad = self.activation3.backward(grad_output)
        grad = self.layer3.backward(grad)
        grad = self.activation2.backward(grad)
        grad = self.layer2.backward(grad)
        grad = self.activation1.backward(grad)
        grad = self.layer1.backward(grad)
        
        self.gradients = [
            self.layer1.grad_weights, self.layer1.grad_bias,
            self.layer2.grad_weights, self.layer2.grad_bias,
            self.layer3.grad_weights, self.layer3.grad_bias
        ]
        return grad

# Delta function
def delta_function(x):
    return (x > 0).astype(float)


def compute_loss_and_gradients(model, engine_data_list, lambda0=1, lambda1=0.001, lambda2=0.001, init_threshold=0 ,
                                debug=False, epoch=0):
    N = len(engine_data_list)

    loss_term0 = 0.0  # 초기 HI 제약 (비활성화)
    loss_term1 = 0.0
    loss_term2 = 0.0
    loss_term3 = 0.0

    all_HI     = []
    all_inputs = []

    if debug:
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch} - LOSS COMPUTATION")
        print(f"{'='*80}")

    for n in range(N):
        engine_data = engine_data_list[n]
        T_n = len(engine_data)
        all_inputs.append(engine_data)

        HI_n = model.forward(engine_data)
        all_HI.append(HI_n)

        if debug:
            print(f"\n[Engine {n+1}] T_n={T_n}")
            print(f"  HI range: [{HI_n.min():.6f}, {HI_n.max():.6f}]")
            print(f"  HI[0] (first): {HI_n[0, 0]:.6f}")
            print(f"  HI[{T_n-1}] (last): {HI_n[-1, 0]:.6f}")

        # Term 0: 초기 HI 제약 (활성화)
        h_n_T0 = HI_n[0, 0]
        if h_n_T0 > init_threshold:
            term0_contribution = (h_n_T0 - init_threshold) ** 2
        else:
            term0_contribution = 0.0
        loss_term0 += term0_contribution

        # Term 1: 마지막 HI는 1에 가깝게
        h_n_Tn = HI_n[-1, 0]
        term1_contribution = (h_n_Tn - 1) ** 2
        loss_term1 += term1_contribution

        if debug:
            print(f"  Loss Term 1 contribution: {term1_contribution:.6f} (target: h_n(T_n)=1)")

        # Term 2: monotonicity
        term2_contribution = 0.0
        monotonicity_violations = 0

        for t in range(1, T_n):
            d_nt = HI_n[t-1, 0] - HI_n[t, 0]
            exp_d = np.exp(d_nt)
            penalty = max(exp_d - 1, 0)
            term2_contribution += (1.0 / (T_n - 1)) * penalty
            if d_nt > 0:
                monotonicity_violations += 1

        loss_term2 += term2_contribution

        if debug:
            print(f"  Loss Term 2 contribution: {term2_contribution:.6f}")
            print(f"  Monotonicity violations: {monotonicity_violations}/{T_n-1}")

        # Term 3: convexity (기울기가 점점 커지도록 강제)
        term3_contribution = 0.0
        convexity_violations = 0

        for t in range(2, T_n):
            current_slope = HI_n[t-1, 0] - HI_n[t, 0]
            past_slope    = HI_n[t-2, 0] - HI_n[t-1, 0]
            diff          = current_slope - past_slope
            exp_diff      = np.exp(diff)
            penalty       = max(exp_diff - 1, 0)
            term3_contribution += (1.0 / (T_n - 2)) * penalty
            if diff > 0:
                convexity_violations += 1

        loss_term3 += term3_contribution

        if debug:
            print(f"  Loss Term 3 contribution: {term3_contribution:.6f}")
            print(f"  Convexity violations: {convexity_violations}/{T_n - 2}")
            diffs = np.diff(HI_n.flatten())
            print(f"  Gradient (dHI/dt) stats:")
            print(f"    Mean: {diffs.mean():.6f}, Std: {diffs.std():.6f}")

    total_loss = lambda0 * loss_term0 + loss_term1 + lambda1 * loss_term2 + lambda2 * loss_term3
    # total_loss = loss_term1 + lambda1 * loss_term2 + lambda2 * loss_term3

    if debug:
        print(f"\n{'='*80}")
        print(f"LOSS BREAKDOWN:")
        print(f"  Term 0 (Initial HI <= {init_threshold}): {loss_term0:.6f} (×{lambda0})")
        print(f"  Term 1 (Target HI=1): {loss_term1:.6f} (×1.0)")
        print(f"  Term 2 (Monotonicity): {loss_term2:.6f} (×{lambda1})")
        print(f"  Term 3 (Convexity): {loss_term3:.6f} (×{lambda2})")
        print(f"  Total Loss: {total_loss:.6f}")
        print(f"{'='*80}")

    # Compute gradients
    total_grad_output = []

    for n in range(N):
        HI_n   = all_HI[n]
        T_n    = len(engine_data_list[n])
        grad_h = np.zeros_like(HI_n)

        # Gradient from term 0 (비활성화)
        h_n_T0 = HI_n[0, 0]
        if h_n_T0 > init_threshold:
            grad_h[0, 0] += 2 * lambda0 * (h_n_T0 - init_threshold)

        # Gradient from term 1
        h_n_Tn = HI_n[-1, 0]
        grad_h[-1, 0] += 2 * (h_n_Tn - 1)

        # Gradient from term 2
        for t in range(1, T_n):
            d_nt      = HI_n[t-1, 0] - HI_n[t, 0]
            exp_d     = np.exp(d_nt)
            delta_exp = delta_function(exp_d - 1)
            grad_d_nt = (lambda1 / (T_n - 1)) * delta_exp * exp_d
            grad_h[t,   0] -= grad_d_nt
            grad_h[t-1, 0] += grad_d_nt

        # Gradient from term 3
        for t in range(2, T_n):
            current_slope  = HI_n[t-1, 0] - HI_n[t, 0]
            past_slope     = HI_n[t-2, 0] - HI_n[t-1, 0]
            diff           = current_slope - past_slope
            exp_diff       = np.exp(diff)
            delta_exp_diff = delta_function(exp_diff - 1)

            grad_diff = (lambda2 / (T_n - 2)) * delta_exp_diff * exp_diff

            grad_h[t,   0] -= grad_diff   # d(diff)/d(HI[t])   = -1
            grad_h[t-1, 0] += 2 * grad_diff  # d(diff)/d(HI[t-1]) = +2
            grad_h[t-2, 0] -= grad_diff   # d(diff)/d(HI[t-2]) = -1

        total_grad_output.append(grad_h)

    # Backpropagation
    for n in range(N):
        engine_data = all_inputs[n]
        grad_output = total_grad_output[n]
        _ = model.forward(engine_data)
        model.backward(grad_output)

        if n == 0:
            accumulated_grads = [g.copy() for g in model.gradients]
        else:
            for i in range(len(accumulated_grads)):
                accumulated_grads[i] += model.gradients[i]

    for i in range(len(accumulated_grads)):
        accumulated_grads[i] /= N

    if debug:
        print(f"\nPARAMETER GRADIENTS (averaged over {N} engines):")
        grad_names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
        for name, grad in zip(grad_names, accumulated_grads):
            print(f"  {name}: mean={grad.mean():.8f}, std={grad.std():.8f}, max_abs={np.abs(grad).max():.8f}")

    model.gradients = accumulated_grads
    return total_loss

    if debug:
        print(f"\n{'='*80}")
        print(f"LOSS BREAKDOWN:")
        print(f"  Term 0 (Initial HI <= {init_threshold}): {loss_term0:.6f} (×{lambda_init})")
        print(f"  Term 1 (Target HI=1): {loss_term1:.6f} (×1.0)")
        print(f"  Term 2 (Monotonicity): {loss_term2:.6f} (×{lambda1})")
        print(f"  Term 3 (Convexity): {loss_term3:.6f} (×{lambda2})")
        print(f"  Total Loss: {total_loss:.6f}")
        print(f"{'='*80}")


# Adam Optimizer
class AdamOptimizer:
    def __init__(self, parameters, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha   = alpha
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.t = 0

    def step(self, parameters, gradients, debug=False):
        self.t += 1

        if debug:
            print(f"\nADAM OPTIMIZER UPDATE (step {self.t}):")

        for i in range(len(parameters)):
            g          = gradients[i]
            self.m[i]  = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i]  = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            m_hat      = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat      = self.v[i] / (1 - self.beta2 ** self.t)
            update     = self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
            parameters[i] -= update

            if debug and i < 2:
                param_names = ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']
                print(f"  {param_names[i]}: update_mean={update.mean():.8f}, update_max_abs={np.abs(update).max():.8f}")


def train_model(engine_data_list, epochs=1000, lambda0 = 1,lambda1=0.001, lambda2=0.001, init_threshold=0,
                alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                verbose=True, print_every=100, debug_epochs=None):

    num_features = engine_data_list[0].shape[1]
    model        = NeuralDataFusionModel(input_dim=num_features)
    optimizer    = AdamOptimizer(model.parameters, alpha, beta1, beta2, epsilon)
    loss_history = []

    if debug_epochs is None:
        debug_epochs = [epochs * i // 10 for i in range(1, 11)]

    if verbose:
        print(f"Debug epochs: {debug_epochs}")
        print(f"Number of engines: {len(engine_data_list)}")
        for n, ed in enumerate(engine_data_list):
            print(f"  Engine {n+1}: {ed.shape[0]} cycles")

    for epoch in range(epochs):
        debug = verbose and ((epoch + 1) in debug_epochs)
        loss = compute_loss_and_gradients(
            model, engine_data_list, lambda0, lambda1, lambda2, init_threshold,
            debug=debug, epoch=epoch+1
        )
        loss_history.append(loss)
        optimizer.step(model.parameters, model.gradients, debug=debug)

        if verbose and (epoch + 1) % print_every == 0:
            print(f"\nEpoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")

    return model, loss_history


def predict(model, engine_data_list):
    predictions = []
    for engine_data in engine_data_list:
        HI_n = model.forward(engine_data)
        predictions.append(HI_n.flatten())
    return predictions