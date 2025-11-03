// Neural Network Playground - Main Application

// ============= GLOBAL STATE =============
const state = {
    dataset: 'circle',
    noise: 0.1,
    trainTestSplit: 0.5,
    learningRate: 0.01,
    batchSize: 32,
    epochs: 100,
    optimizer: 'adam',
    activation: 'relu',
    regularization: 'none',
    regularizationRate: 0.001,
    initialization: 'xavier',
    features: ['x1', 'x2'],
    hiddenLayers: [4],
    isTraining: false,
    currentEpoch: 0,
    speed: 1,
    data: { train: [], test: [] },
    network: null,
    history: { trainLoss: [], testLoss: [], trainAcc: [], testAcc: [] }
};

let lossChart = null;
let accuracyChart = null;
let animationFrame = null;

// ============= DATASET GENERATION =============
class DataGenerator {
    static generate(type, numSamples = 200, noise = 0.1) {
        const data = [];
        const n = numSamples;
        
        switch(type) {
            case 'circle':
                return this.generateCircle(n, noise);
            case 'xor':
                return this.generateXOR(n, noise);
            case 'gaussian':
                return this.generateGaussian(n, noise);
            case 'spiral':
                return this.generateSpiral(n, noise);
            case 'plane':
                return this.generatePlane(n, noise);
            case 'multigaussian':
                return this.generateMultiGaussian(n, noise);
            default:
                return this.generateCircle(n, noise);
        }
    }
    
    static generateCircle(n, noise) {
        const data = [];
        for (let i = 0; i < n; i++) {
            const r = Math.random();
            const angle = Math.random() * 2 * Math.PI;
            const x = r * Math.cos(angle) + (Math.random() - 0.5) * noise;
            const y = r * Math.sin(angle) + (Math.random() - 0.5) * noise;
            const label = r < 0.5 ? 0 : 1;
            data.push({ x, y, label });
        }
        return data;
    }
    
    static generateXOR(n, noise) {
        const data = [];
        for (let i = 0; i < n; i++) {
            const x = Math.random() * 2 - 1;
            const y = Math.random() * 2 - 1;
            const label = (x * y >= 0) ? 1 : 0;
            data.push({ 
                x: x + (Math.random() - 0.5) * noise, 
                y: y + (Math.random() - 0.5) * noise, 
                label 
            });
        }
        return data;
    }
    
    static generateGaussian(n, noise) {
        const data = [];
        for (let i = 0; i < n; i++) {
            const label = Math.random() > 0.5 ? 1 : 0;
            const centerX = label === 1 ? 0.5 : -0.5;
            const centerY = label === 1 ? 0.5 : -0.5;
            const x = centerX + this.gaussianRandom() * 0.3 + (Math.random() - 0.5) * noise;
            const y = centerY + this.gaussianRandom() * 0.3 + (Math.random() - 0.5) * noise;
            data.push({ x, y, label });
        }
        return data;
    }
    
    static generateSpiral(n, noise) {
        const data = [];
        const points = n / 2;
        for (let i = 0; i < points; i++) {
            const t = i / points * 4 * Math.PI;
            for (let label = 0; label < 2; label++) {
                const r = t / (4 * Math.PI);
                const angle = t + label * Math.PI;
                const x = r * Math.cos(angle) + (Math.random() - 0.5) * noise * 0.5;
                const y = r * Math.sin(angle) + (Math.random() - 0.5) * noise * 0.5;
                data.push({ x, y, label });
            }
        }
        return data;
    }
    
    static generatePlane(n, noise) {
        const data = [];
        for (let i = 0; i < n; i++) {
            const x = Math.random() * 2 - 1;
            const y = Math.random() * 2 - 1;
            const value = 0.5 * x + 0.3 * y + (Math.random() - 0.5) * noise;
            data.push({ x, y, label: value });
        }
        return data;
    }
    
    static generateMultiGaussian(n, noise) {
        const data = [];
        const centers = [[-0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        for (let i = 0; i < n; i++) {
            const center = centers[Math.floor(Math.random() * centers.length)];
            const x = center[0] + this.gaussianRandom() * 0.2 + (Math.random() - 0.5) * noise;
            const y = center[1] + this.gaussianRandom() * 0.2 + (Math.random() - 0.5) * noise;
            const value = center[0] + center[1] + (Math.random() - 0.5) * noise;
            data.push({ x, y, label: value });
        }
        return data;
    }
    
    static gaussianRandom() {
        let u = 0, v = 0;
        while(u === 0) u = Math.random();
        while(v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
}

// ============= NEURAL NETWORK =============
class NeuralNetwork {
    constructor(layers, activation = 'relu', learningRate = 0.01, optimizer = 'adam') {
        this.layers = layers;
        this.activation = activation;
        this.learningRate = learningRate;
        this.optimizer = optimizer;
        this.weights = [];
        this.biases = [];
        this.activations = [];
        this.zValues = [];
        
        // Optimizer state
        this.m_w = [];
        this.v_w = [];
        this.m_b = [];
        this.v_b = [];
        this.t = 0;
        
        this.initializeWeights(state.initialization);
    }
    
    initializeWeights(method = 'xavier') {
        for (let i = 0; i < this.layers.length - 1; i++) {
            const inputSize = this.layers[i];
            const outputSize = this.layers[i + 1];
            let scale = 0.1;
            
            if (method === 'xavier') {
                scale = Math.sqrt(2.0 / (inputSize + outputSize));
            } else if (method === 'he') {
                scale = Math.sqrt(2.0 / inputSize);
            }
            
            const w = [];
            for (let j = 0; j < outputSize; j++) {
                const row = [];
                for (let k = 0; k < inputSize; k++) {
                    row.push((Math.random() * 2 - 1) * scale);
                }
                w.push(row);
            }
            this.weights.push(w);
            
            const b = new Array(outputSize).fill(0).map(() => Math.random() * 0.01);
            this.biases.push(b);
            
            // Initialize optimizer state
            this.m_w.push(w.map(row => row.map(() => 0)));
            this.v_w.push(w.map(row => row.map(() => 0)));
            this.m_b.push(b.map(() => 0));
            this.v_b.push(b.map(() => 0));
        }
    }
    
    activate(x, activation = this.activation) {
        switch(activation) {
            case 'relu':
                return x.map(val => Math.max(0, val));
            case 'sigmoid':
                return x.map(val => 1 / (1 + Math.exp(-val)));
            case 'tanh':
                return x.map(val => Math.tanh(val));
            case 'linear':
                return x;
            default:
                return x.map(val => Math.max(0, val));
        }
    }
    
    activateDerivative(x, activation = this.activation) {
        switch(activation) {
            case 'relu':
                return x.map(val => val > 0 ? 1 : 0);
            case 'sigmoid':
                const sig = this.activate(x, 'sigmoid');
                return sig.map(val => val * (1 - val));
            case 'tanh':
                return x.map(val => 1 - Math.tanh(val) ** 2);
            case 'linear':
                return x.map(() => 1);
            default:
                return x.map(val => val > 0 ? 1 : 0);
        }
    }
    
    forward(input) {
        this.activations = [input];
        this.zValues = [];
        
        let current = input;
        for (let i = 0; i < this.weights.length; i++) {
            const z = this.matmul(this.weights[i], current, this.biases[i]);
            this.zValues.push(z);
            
            const isOutputLayer = i === this.weights.length - 1;
            const act = isOutputLayer ? 'sigmoid' : this.activation;
            current = this.activate(z, act);
            this.activations.push(current);
        }
        
        return current;
    }
    
    matmul(weights, input, bias) {
        const result = [];
        for (let i = 0; i < weights.length; i++) {
            let sum = bias[i];
            for (let j = 0; j < input.length; j++) {
                sum += weights[i][j] * input[j];
            }
            result.push(sum);
        }
        return result;
    }
    
    backward(target) {
        const gradients = [];
        const output = this.activations[this.activations.length - 1];
        
        // Output layer error
        let delta = output.map((o, i) => o - target[i]);
        
        // Backpropagate
        for (let i = this.weights.length - 1; i >= 0; i--) {
            const grad_w = [];
            const grad_b = [...delta];
            
            for (let j = 0; j < this.weights[i].length; j++) {
                const row = [];
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    row.push(delta[j] * this.activations[i][k]);
                }
                grad_w.push(row);
            }
            
            gradients.unshift({ w: grad_w, b: grad_b });
            
            if (i > 0) {
                const newDelta = new Array(this.activations[i].length).fill(0);
                for (let j = 0; j < this.activations[i].length; j++) {
                    for (let k = 0; k < delta.length; k++) {
                        newDelta[j] += delta[k] * this.weights[i][k][j];
                    }
                }
                
                const actDeriv = this.activateDerivative(this.zValues[i - 1], this.activation);
                delta = newDelta.map((d, idx) => d * actDeriv[idx]);
            }
        }
        
        return gradients;
    }
    
    updateWeights(gradients) {
        this.t++;
        const lr = this.learningRate;
        const beta1 = 0.9;
        const beta2 = 0.999;
        const epsilon = 1e-8;
        
        for (let i = 0; i < this.weights.length; i++) {
            for (let j = 0; j < this.weights[i].length; j++) {
                for (let k = 0; k < this.weights[i][j].length; k++) {
                    const grad = gradients[i].w[j][k];
                    
                    if (this.optimizer === 'adam') {
                        this.m_w[i][j][k] = beta1 * this.m_w[i][j][k] + (1 - beta1) * grad;
                        this.v_w[i][j][k] = beta2 * this.v_w[i][j][k] + (1 - beta2) * grad * grad;
                        
                        const m_hat = this.m_w[i][j][k] / (1 - Math.pow(beta1, this.t));
                        const v_hat = this.v_w[i][j][k] / (1 - Math.pow(beta2, this.t));
                        
                        this.weights[i][j][k] -= lr * m_hat / (Math.sqrt(v_hat) + epsilon);
                    } else if (this.optimizer === 'rmsprop') {
                        this.v_w[i][j][k] = beta2 * this.v_w[i][j][k] + (1 - beta2) * grad * grad;
                        this.weights[i][j][k] -= lr * grad / (Math.sqrt(this.v_w[i][j][k]) + epsilon);
                    } else {
                        this.weights[i][j][k] -= lr * grad;
                    }
                }
            }
            
            for (let j = 0; j < this.biases[i].length; j++) {
                const grad = gradients[i].b[j];
                
                if (this.optimizer === 'adam') {
                    this.m_b[i][j] = beta1 * this.m_b[i][j] + (1 - beta1) * grad;
                    this.v_b[i][j] = beta2 * this.v_b[i][j] + (1 - beta2) * grad * grad;
                    
                    const m_hat = this.m_b[i][j] / (1 - Math.pow(beta1, this.t));
                    const v_hat = this.v_b[i][j] / (1 - Math.pow(beta2, this.t));
                    
                    this.biases[i][j] -= lr * m_hat / (Math.sqrt(v_hat) + epsilon);
                } else if (this.optimizer === 'rmsprop') {
                    this.v_b[i][j] = beta2 * this.v_b[i][j] + (1 - beta2) * grad * grad;
                    this.biases[i][j] -= lr * grad / (Math.sqrt(this.v_b[i][j]) + epsilon);
                } else {
                    this.biases[i][j] -= lr * grad;
                }
            }
        }
    }
    
    train(data, epochs = 1) {
        let totalLoss = 0;
        let correct = 0;
        
        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let i = 0; i < data.length; i++) {
                const point = data[i];
                const input = this.extractFeatures(point);
                const target = typeof point.label === 'number' && point.label < 1 && point.label > 0 ? 
                    [point.label] : [point.label === 1 ? 1 : 0];
                
                const output = this.forward(input);
                const gradients = this.backward(target);
                this.updateWeights(gradients);
                
                const loss = (output[0] - target[0]) ** 2;
                totalLoss += loss;
                
                if ((output[0] > 0.5 ? 1 : 0) === (target[0] > 0.5 ? 1 : 0)) {
                    correct++;
                }
            }
        }
        
        return {
            loss: totalLoss / (data.length * epochs),
            accuracy: correct / (data.length * epochs)
        };
    }
    
    predict(point) {
        const input = this.extractFeatures(point);
        const output = this.forward(input);
        return output[0];
    }
    
    extractFeatures(point) {
        const features = [point.x, point.y];
        
        if (state.features.includes('x1sq')) features.push(point.x * point.x);
        if (state.features.includes('x2sq')) features.push(point.y * point.y);
        if (state.features.includes('x1x2')) features.push(point.x * point.y);
        if (state.features.includes('sinx1')) features.push(Math.sin(point.x * Math.PI));
        if (state.features.includes('sinx2')) features.push(Math.sin(point.y * Math.PI));
        
        return features;
    }
}

// ============= VISUALIZATION =============
class Visualizer {
    static drawData(canvas, data, network = null) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw decision boundary if network exists
        if (network) {
            const resolution = 50;
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const x = (i / resolution) * 2 - 1;
                    const y = (j / resolution) * 2 - 1;
                    const prediction = network.predict({ x, y });
                    
                    const px = ((x + 1) / 2) * width;
                    const py = ((y + 1) / 2) * height;
                    
                    const color = prediction > 0.5 ? 
                        `rgba(0, 217, 255, ${prediction * 0.3})` : 
                        `rgba(255, 107, 53, ${(1 - prediction) * 0.3})`;
                    
                    ctx.fillStyle = color;
                    ctx.fillRect(px, py, width / resolution, height / resolution);
                }
            }
        }
        
        // Draw data points
        data.forEach(point => {
            const px = ((point.x + 1) / 2) * width;
            const py = ((point.y + 1) / 2) * height;
            
            ctx.beginPath();
            ctx.arc(px, py, 4, 0, 2 * Math.PI);
            
            const isPositive = typeof point.label === 'number' ? point.label > 0 : point.label === 1;
            ctx.fillStyle = isPositive ? '#00d9ff' : '#ff6b35';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 1;
            ctx.stroke();
        });
    }
    
    static drawNetwork(svg, network) {
        if (!network) return;
        
        svg.innerHTML = '';
        const width = svg.clientWidth;
        const height = svg.clientHeight;
        
        const layers = network.layers;
        const layerSpacing = width / (layers.length + 1);
        const maxNeurons = Math.max(...layers);
        
        const neuronPositions = [];
        
        // Draw connections first
        for (let i = 0; i < layers.length - 1; i++) {
            const currentLayer = layers[i];
            const nextLayer = layers[i + 1];
            
            for (let j = 0; j < currentLayer; j++) {
                for (let k = 0; k < nextLayer; k++) {
                    const x1 = layerSpacing * (i + 1);
                    const y1 = (height / (currentLayer + 1)) * (j + 1);
                    const x2 = layerSpacing * (i + 2);
                    const y2 = (height / (nextLayer + 1)) * (k + 1);
                    
                    const weight = network.weights[i][k][j];
                    const thickness = Math.min(Math.abs(weight) * 2, 3);
                    const color = weight > 0 ? '#00d9ff' : '#ff6b35';
                    const opacity = Math.min(Math.abs(weight), 1);
                    
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.setAttribute('x1', x1);
                    line.setAttribute('y1', y1);
                    line.setAttribute('x2', x2);
                    line.setAttribute('y2', y2);
                    line.setAttribute('stroke', color);
                    line.setAttribute('stroke-width', thickness);
                    line.setAttribute('opacity', opacity * 0.5);
                    line.setAttribute('class', 'connection-line');
                    svg.appendChild(line);
                }
            }
        }
        
        // Draw neurons
        for (let i = 0; i < layers.length; i++) {
            const layerSize = layers[i];
            const x = layerSpacing * (i + 1);
            
            for (let j = 0; j < layerSize; j++) {
                const y = (height / (layerSize + 1)) * (j + 1);
                
                const activation = network.activations[i] ? network.activations[i][j] : 0;
                const intensity = Math.min(Math.abs(activation), 1);
                
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', x);
                circle.setAttribute('cy', y);
                circle.setAttribute('r', 15);
                circle.setAttribute('fill', `rgba(0, 217, 255, ${intensity})`);
                circle.setAttribute('stroke', '#00d9ff');
                circle.setAttribute('stroke-width', 2);
                circle.setAttribute('class', 'neuron-circle');
                svg.appendChild(circle);
                
                const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                text.setAttribute('x', x);
                text.setAttribute('y', y + 4);
                text.setAttribute('class', 'neuron-label');
                text.textContent = activation ? activation.toFixed(2) : '0';
                svg.appendChild(text);
            }
            
            // Layer label
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', x);
            label.setAttribute('y', 20);
            label.setAttribute('class', 'layer-label');
            label.textContent = i === 0 ? 'Input' : i === layers.length - 1 ? 'Output' : `Hidden ${i}`;
            svg.appendChild(label);
        }
    }
}

// ============= UI INITIALIZATION =============
function initializeUI() {
    // Dataset selection
    document.querySelectorAll('.dataset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.dataset-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.dataset = btn.dataset.dataset;
            generateData();
            updateVisualization();
        });
    });
    
    // Noise slider
    const noiseSlider = document.getElementById('noise-slider');
    const noiseValue = document.getElementById('noise-value');
    noiseSlider.addEventListener('input', (e) => {
        state.noise = parseFloat(e.target.value) / 100;
        noiseValue.textContent = e.target.value;
        generateData();
        updateVisualization();
    });
    
    // Split slider
    const splitSlider = document.getElementById('split-slider');
    const splitValue = document.getElementById('split-value');
    splitSlider.addEventListener('input', (e) => {
        state.trainTestSplit = parseFloat(e.target.value) / 100;
        splitValue.textContent = e.target.value;
        generateData();
    });
    
    // Learning rate
    const lrSlider = document.getElementById('lr-slider');
    const lrValue = document.getElementById('lr-value');
    lrSlider.addEventListener('input', (e) => {
        state.learningRate = Math.pow(10, parseFloat(e.target.value));
        lrValue.textContent = state.learningRate.toFixed(5);
    });
    
    // Batch size
    document.getElementById('batch-size').addEventListener('change', (e) => {
        state.batchSize = parseInt(e.target.value);
    });
    
    // Epochs
    document.getElementById('epochs').addEventListener('change', (e) => {
        state.epochs = parseInt(e.target.value);
        document.getElementById('total-epochs').textContent = state.epochs;
    });
    
    // Optimizer
    document.getElementById('optimizer').addEventListener('change', (e) => {
        state.optimizer = e.target.value;
        resetNetwork();
    });
    
    // Activation
    document.getElementById('activation').addEventListener('change', (e) => {
        state.activation = e.target.value;
        updateFormulaPanel();
        resetNetwork();
    });
    
    // Regularization
    document.getElementById('regularization').addEventListener('change', (e) => {
        state.regularization = e.target.value;
    });
    
    // Regularization rate
    const regSlider = document.getElementById('reg-slider');
    const regValue = document.getElementById('reg-value');
    regSlider.addEventListener('input', (e) => {
        state.regularizationRate = parseFloat(e.target.value);
        regValue.textContent = state.regularizationRate.toFixed(4);
    });
    
    // Initialization
    document.getElementById('initialization').addEventListener('change', (e) => {
        state.initialization = e.target.value;
        resetNetwork();
    });
    
    // Features
    document.querySelectorAll('.feature-checkbox input').forEach(checkbox => {
        checkbox.addEventListener('change', (e) => {
            const feature = e.target.dataset.feature;
            if (e.target.checked) {
                if (!state.features.includes(feature)) {
                    state.features.push(feature);
                }
            } else {
                state.features = state.features.filter(f => f !== feature);
            }
            resetNetwork();
        });
    });
    
    // Layer controls
    document.getElementById('add-layer').addEventListener('click', () => {
        state.hiddenLayers.push(4);
        updateLayerControls();
        resetNetwork();
    });
    
    document.getElementById('remove-layer').addEventListener('click', () => {
        if (state.hiddenLayers.length > 1) {
            state.hiddenLayers.pop();
            updateLayerControls();
            resetNetwork();
        }
    });
    
    // Training controls
    document.getElementById('play-btn').addEventListener('click', startTraining);
    document.getElementById('pause-btn').addEventListener('click', pauseTraining);
    document.getElementById('step-btn').addEventListener('click', stepTraining);
    document.getElementById('reset-btn').addEventListener('click', resetNetwork);
    
    // Speed
    const speedSlider = document.getElementById('speed-slider');
    const speedValue = document.getElementById('speed-value');
    speedSlider.addEventListener('input', (e) => {
        state.speed = parseInt(e.target.value);
        speedValue.textContent = `${state.speed}x`;
    });
    
    // Initialize charts
    initializeCharts();
    
    // Initial setup
    generateData();
    resetNetwork();
    updateVisualization();
    updateLayerControls();
    updateFormulaPanel();
}

function updateLayerControls() {
    const container = document.getElementById('layer-neuron-controls');
    container.innerHTML = '';
    
    state.hiddenLayers.forEach((neurons, index) => {
        const control = document.createElement('div');
        control.className = 'layer-control';
        
        const label = document.createElement('label');
        label.textContent = `Layer ${index + 1}:`;
        
        const minusBtn = document.createElement('button');
        minusBtn.className = 'neuron-btn';
        minusBtn.textContent = '−';
        minusBtn.addEventListener('click', () => {
            if (state.hiddenLayers[index] > 1) {
                state.hiddenLayers[index]--;
                updateLayerControls();
                resetNetwork();
            }
        });
        
        const count = document.createElement('span');
        count.className = 'neuron-count';
        count.textContent = neurons;
        
        const plusBtn = document.createElement('button');
        plusBtn.className = 'neuron-btn';
        plusBtn.textContent = '+';
        plusBtn.addEventListener('click', () => {
            if (state.hiddenLayers[index] < 8) {
                state.hiddenLayers[index]++;
                updateLayerControls();
                resetNetwork();
            }
        });
        
        control.appendChild(label);
        control.appendChild(minusBtn);
        control.appendChild(count);
        control.appendChild(plusBtn);
        container.appendChild(control);
    });
}

function updateFormulaPanel() {
    const formulas = {
        relu: { formula: 'max(0, x)', range: '[0, ∞)', desc: 'Fast and effective for hidden layers' },
        sigmoid: { formula: '1/(1+e^-x)', range: '(0, 1)', desc: 'For binary classification output' },
        tanh: { formula: 'tanh(x)', range: '(-1, 1)', desc: 'Zero-centered alternative to sigmoid' },
        linear: { formula: 'x', range: '(-∞, ∞)', desc: 'For regression output' }
    };
    
    const info = formulas[state.activation];
    const content = document.getElementById('formula-content');
    content.innerHTML = `
        <div class="formula-item"><strong>${state.activation.toUpperCase()}:</strong> ${info.formula}</div>
        <div class="formula-item"><strong>Range:</strong> ${info.range}</div>
        <div class="formula-item"><em>${info.desc}</em></div>
    `;
}

function generateData() {
    const allData = DataGenerator.generate(state.dataset, 300, state.noise);
    const splitIndex = Math.floor(allData.length * state.trainTestSplit);
    
    // Shuffle
    for (let i = allData.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [allData[i], allData[j]] = [allData[j], allData[i]];
    }
    
    state.data.train = allData.slice(0, splitIndex);
    state.data.test = allData.slice(splitIndex);
}

function resetNetwork() {
    const inputSize = state.features.length;
    const layers = [inputSize, ...state.hiddenLayers, 1];
    
    state.network = new NeuralNetwork(layers, state.activation, state.learningRate, state.optimizer);
    state.currentEpoch = 0;
    state.history = { trainLoss: [], testLoss: [], trainAcc: [], testAcc: [] };
    
    updateVisualization();
    updateMetrics();
    updateCharts();
    
    document.getElementById('current-epoch').textContent = '0';
    document.getElementById('info-content').innerHTML = '<p>Network reset! Ready to train with new configuration.</p>';
}

function startTraining() {
    state.isTraining = true;
    document.getElementById('play-btn').style.display = 'none';
    document.getElementById('pause-btn').style.display = 'inline-flex';
    
    trainLoop();
}

function pauseTraining() {
    state.isTraining = false;
    document.getElementById('play-btn').style.display = 'inline-flex';
    document.getElementById('pause-btn').style.display = 'none';
    
    if (animationFrame) {
        cancelAnimationFrame(animationFrame);
    }
}

function stepTraining() {
    if (state.currentEpoch < state.epochs) {
        const trainMetrics = state.network.train(state.data.train, 1);
        const testMetrics = evaluateNetwork(state.data.test);
        
        state.currentEpoch++;
        state.history.trainLoss.push(trainMetrics.loss);
        state.history.testLoss.push(testMetrics.loss);
        state.history.trainAcc.push(trainMetrics.accuracy);
        state.history.testAcc.push(testMetrics.accuracy);
        
        updateVisualization();
        updateMetrics();
        updateCharts();
        
        document.getElementById('current-epoch').textContent = state.currentEpoch;
        updateInfoPanel();
    }
}

function trainLoop() {
    if (!state.isTraining || state.currentEpoch >= state.epochs) {
        pauseTraining();
        return;
    }
    
    for (let i = 0; i < state.speed; i++) {
        if (state.currentEpoch < state.epochs) {
            stepTraining();
        }
    }
    
    animationFrame = requestAnimationFrame(trainLoop);
}

function evaluateNetwork(data) {
    let totalLoss = 0;
    let correct = 0;
    
    data.forEach(point => {
        const prediction = state.network.predict(point);
        const target = typeof point.label === 'number' && point.label < 1 && point.label > 0 ? 
            point.label : (point.label === 1 ? 1 : 0);
        
        totalLoss += (prediction - target) ** 2;
        
        if ((prediction > 0.5 ? 1 : 0) === (target > 0.5 ? 1 : 0)) {
            correct++;
        }
    });
    
    return {
        loss: totalLoss / data.length,
        accuracy: correct / data.length
    };
}

function updateVisualization() {
    const canvas = document.getElementById('viz-canvas');
    const svg = document.getElementById('network-svg');
    
    Visualizer.drawData(canvas, [...state.data.train, ...state.data.test], state.network);
    Visualizer.drawNetwork(svg, state.network);
}

function updateMetrics() {
    const trainMetrics = evaluateNetwork(state.data.train);
    const testMetrics = evaluateNetwork(state.data.test);
    
    document.getElementById('train-loss').textContent = trainMetrics.loss.toFixed(3);
    document.getElementById('test-loss').textContent = testMetrics.loss.toFixed(3);
    document.getElementById('train-accuracy').textContent = `${(trainMetrics.accuracy * 100).toFixed(1)}%`;
    document.getElementById('test-accuracy').textContent = `${(testMetrics.accuracy * 100).toFixed(1)}%`;
}

function updateInfoPanel() {
    const trainLoss = state.history.trainLoss[state.history.trainLoss.length - 1];
    const testLoss = state.history.testLoss[state.history.testLoss.length - 1];
    const trainAcc = state.history.trainAcc[state.history.trainAcc.length - 1];
    
    let message = `<p><strong>Epoch ${state.currentEpoch}:</strong> Training in progress...</p>`;
    
    if (trainLoss < 0.1) {
        message += '<p>✓ Low training loss - model is learning well!</p>';
    } else if (trainLoss > 0.5) {
        message += '<p>⚠ High training loss - consider adjusting learning rate or architecture.</p>';
    }
    
    if (testLoss - trainLoss > 0.2) {
        message += '<p>⚠ Model may be overfitting. Try adding regularization.</p>';
    }
    
    if (trainAcc > 0.95) {
        message += '<p>✓ Excellent accuracy achieved!</p>';
    }
    
    document.getElementById('info-content').innerHTML = message;
}

function initializeCharts() {
    const lossCtx = document.getElementById('loss-chart').getContext('2d');
    const accCtx = document.getElementById('accuracy-chart').getContext('2d');
    
    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: '#00d9ff',
                    backgroundColor: 'rgba(0, 217, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4
                },
                {
                    label: 'Test Loss',
                    data: [],
                    borderColor: '#ff6b35',
                    backgroundColor: 'rgba(255, 107, 53, 0.1)',
                    borderWidth: 2,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, labels: { color: '#e0e0e0' } },
                title: { display: true, text: 'Loss Over Time', color: '#e0e0e0' }
            },
            scales: {
                x: { display: false },
                y: { ticks: { color: '#a0a0a0' }, grid: { color: '#2a2a3e' } }
            }
        }
    });
    
    accuracyChart = new Chart(accCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Accuracy',
                    data: [],
                    borderColor: '#00d9ff',
                    backgroundColor: 'rgba(0, 217, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4
                },
                {
                    label: 'Test Accuracy',
                    data: [],
                    borderColor: '#ff6b35',
                    backgroundColor: 'rgba(255, 107, 53, 0.1)',
                    borderWidth: 2,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: true, labels: { color: '#e0e0e0' } },
                title: { display: true, text: 'Accuracy Over Time', color: '#e0e0e0' }
            },
            scales: {
                x: { display: false },
                y: { 
                    ticks: { color: '#a0a0a0' }, 
                    grid: { color: '#2a2a3e' },
                    min: 0,
                    max: 1
                }
            }
        }
    });
}

function updateCharts() {
    if (!lossChart || !accuracyChart) return;
    
    const labels = state.history.trainLoss.map((_, i) => i + 1);
    
    lossChart.data.labels = labels;
    lossChart.data.datasets[0].data = state.history.trainLoss;
    lossChart.data.datasets[1].data = state.history.testLoss;
    lossChart.update('none');
    
    accuracyChart.data.labels = labels;
    accuracyChart.data.datasets[0].data = state.history.trainAcc;
    accuracyChart.data.datasets[1].data = state.history.testAcc;
    accuracyChart.update('none');
}

// Initialize app
document.addEventListener('DOMContentLoaded', initializeUI);