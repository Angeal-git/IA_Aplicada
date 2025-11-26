class Perceptron {
    constructor(w1, w2, theta) {
        this.w1 = w1;
        this.w2 = w2;
        this.theta = theta;
        this.logs = [];
    }
    // Activation function: 1 if Net >= 0, else 0 (Standard Step)
    // Note: User treats Theta as something to be ADDED to Net (like Bias) based on update rule "Theta + y"
    activation(net) {
        return net > 0 ? 1 : -1;
    }
    train(patterns) {
        this.logs = [];
        let converged = false;
        let epoch = 0;
        const maxEpochs = 100; // Hardcoded limit to prevent infinite loops
        while (!converged && epoch < maxEpochs) {
            converged = true;
            epoch++;
            for (let i = 0; i < patterns.length; i++) {
                const p = patterns[i];
                const x1 = p.x1;
                const x2 = p.x2;
                const target = p.y;

                // Calculate Net: x1*w1 + x2*w2 + theta
                const net = (x1 * this.w1) + (x2 * this.w2) + this.theta;

                // Determine output
                const output = this.activation(net);
                const error = target - output;

                const stepLog = {
                    epoch: epoch,
                    patternIndex: i + 1,
                    inputs: { x1, x2, y: target },
                    weightsBefore: { w1: this.w1, w2: this.w2, theta: this.theta },
                    net: net.toFixed(4),
                    output: output,
                    error: error,
                    delta: { dw1: 0, dw2: 0, dTheta: 0 },
                    weightsAfter: {}
                };

                if (error !== 0) {
                    converged = false;
                    // Update Rule: delta = target * input
                    const dw1 = target * x1;
                    const dw2 = target * x2;
                    const dTheta = target;

                    this.w1 += dw1;
                    this.w2 += dw2;
                    this.theta += dTheta;
                    stepLog.delta = { dw1, dw2, dTheta };
                }
                stepLog.weightsAfter = { w1: this.w1, w2: this.w2, theta: this.theta };
                this.logs.push(stepLog);
            }
        }
        return {
            converged,
            epochs: epoch,
            finalWeights: { w1: this.w1, w2: this.w2, theta: this.theta },
            logs: this.logs
        };
    }
}
// UI Logic
document.addEventListener('DOMContentLoaded', () => {
    const dataTableBody = document.querySelector('#dataTable tbody');
    const addRowBtn = document.getElementById('addRowBtn');
    const trainBtn = document.getElementById('trainBtn');
    // Initialize with some default rows
    addPatternRow();
    addPatternRow();
    addPatternRow();
    addPatternRow();
    // Event Listeners
    addRowBtn.addEventListener('click', () => addPatternRow());
    trainBtn.addEventListener('click', runTraining);
    function addPatternRow() {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td><input type="number" class="input-x1" value="0"></td>
            <td><input type="number" class="input-x2" value="0"></td>
            <td><input type="number" class="input-y" value="0"></td>
            <td><button class="btn btn-sm btn-danger delete-row-btn">×</button></td>
        `;
        tr.querySelector('.delete-row-btn').addEventListener('click', () => {
            if (dataTableBody.children.length > 1) {
                tr.remove();
            }
        });
        dataTableBody.appendChild(tr);
    }
    function getPatterns() {
        const rows = dataTableBody.querySelectorAll('tr');
        const patterns = [];
        rows.forEach(row => {
            const x1 = parseFloat(row.querySelector('.input-x1').value) || 0;
            const x2 = parseFloat(row.querySelector('.input-x2').value) || 0;
            const y = parseFloat(row.querySelector('.input-y').value) || 0;
            patterns.push({ x1, x2, y });
        });
        return patterns;
    }
    function runTraining() {
        const initW1 = parseFloat(document.getElementById('initW1').value) || 0;
        const initW2 = parseFloat(document.getElementById('initW2').value) || 0;
        const initTheta = parseFloat(document.getElementById('initTheta').value) || 0;
        const patterns = getPatterns();
        const perceptron = new Perceptron(initW1, initW2, initTheta);
        const result = perceptron.train(patterns);
        displayResults(result);
    }
    function displayResults(result) {
        const resultsSection = document.getElementById('resultsSection');
        resultsSection.classList.remove('hidden');

        // Summary
        const statusEl = document.getElementById('statusValue');
        statusEl.textContent = result.converged ? 'Convergió ✅' : 'No Convergió ❌';
        statusEl.style.color = result.converged ? 'var(--success)' : 'var(--danger)';
        document.getElementById('epochsValue').textContent = result.epochs;
        const w = result.finalWeights;
        document.getElementById('weightsValue').textContent = `w1: ${w.w1}, w2: ${w.w2}, θ: ${w.theta}`;

        // Group logs by epoch
        const epochsContainer = document.getElementById('epochsContainer');
        epochsContainer.innerHTML = '';

        const logsByEpoch = {};
        result.logs.forEach(log => {
            if (!logsByEpoch[log.epoch]) {
                logsByEpoch[log.epoch] = [];
            }
            logsByEpoch[log.epoch].push(log);
        });

        // Create epoch cards
        Object.keys(logsByEpoch).forEach(epochNum => {
            const epochLogs = logsByEpoch[epochNum];

            const epochCard = document.createElement('div');
            epochCard.className = 'epoch-card';

            const epochHeader = document.createElement('div');
            epochHeader.className = 'epoch-header';
            epochHeader.innerHTML = `<h3>🔄 Época ${epochNum}</h3>`;
            epochCard.appendChild(epochHeader);

            const table = document.createElement('table');
            table.className = 'epoch-table';

            // Table header
            const thead = document.createElement('thead');
            thead.innerHTML = `
                <tr>
                    <th>Patrón</th>
                    <th>Entradas (x1, x2)</th>
                    <th>Pesos Ant. (w1, w2, θ)</th>
                    <th>Sustitución</th>
                    <th>Net</th>
                    <th>Salida</th>
                    <th>Error</th>
                    <th>Deltas (Δw1, Δw2, Δθ)</th>
                    <th>Pesos Nuevos</th>
                </tr>
            `;
            table.appendChild(thead);

            // Table body
            const tbody = document.createElement('tbody');
            epochLogs.forEach(log => {
                const tr = document.createElement('tr');
                const inputsStr = `(${log.inputs.x1}, ${log.inputs.x2})`;
                const wBeforeStr = `(${log.weightsBefore.w1}, ${log.weightsBefore.w2}, ${log.weightsBefore.theta})`;
                const wAfterStr = `(${log.weightsAfter.w1}, ${log.weightsAfter.w2}, ${log.weightsAfter.theta})`;
                const deltasStr = log.error !== 0
                    ? `(${log.delta.dw1}, ${log.delta.dw2}, ${log.delta.dTheta})`
                    : '-';
                const substitution = `(${log.inputs.x1})(${log.weightsBefore.w1}) + (${log.inputs.x2})(${log.weightsBefore.w2}) + ${log.weightsBefore.theta}`;

                tr.innerHTML = `
                    <td>${log.patternIndex}</td>
                    <td>${inputsStr}</td>
                    <td>${wBeforeStr}</td>
                    <td style="font-size: 0.85rem; color: var(--text-muted);">${substitution}</td>
                    <td>${log.net}</td>
                    <td>${log.output}</td>
                    <td>${log.error}</td>
                    <td>${deltasStr}</td>
                    <td>${wAfterStr}</td>
                `;
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);

            epochCard.appendChild(table);
            epochsContainer.appendChild(epochCard);
        });
    }
});
