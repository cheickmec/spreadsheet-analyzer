// Global variables
let viz;
let config;

// Initialize the application
async function init() {
    try {
        // Fetch configuration
        const configResponse = await fetch('/api/config');
        config = await configResponse.json();

        // Fetch predefined queries
        const queriesResponse = await fetch('/api/queries');
        const { queries } = await queriesResponse.json();

        // Populate query dropdown
        const querySelect = document.getElementById('querySelect');
        queries.forEach(query => {
            const option = document.createElement('option');
            option.value = query.cypher;
            option.textContent = query.name;
            option.title = query.description;
            querySelect.appendChild(option);
        });

        // Initialize with a simple query
        initializeVisualization("MATCH (n:Cell) WHERE n.formula <> '' RETURN n LIMIT 25");

    } catch (error) {
        console.error('Failed to initialize:', error);
        showError('Failed to connect to server');
    }
}

// Initialize Neovis visualization
function initializeVisualization(initialCypher) {
    const vizConfig = {
        containerId: "viz",
        neo4j: config.neo4j,
        labels: {
            Cell: {
                label: 'ref',
                size: 'pagerank',
                color: '#3498db',
                font: {
                    size: 12,
                    color: '#333'
                }
            },
            Range: {
                label: 'ref',
                size: 'size',
                color: '#e74c3c',
                font: {
                    size: 12,
                    color: '#333'
                }
            }
        },
        relationships: {
            DEPENDS_ON: {
                thickness: 'weight',
                color: '#95a5a6',
                arrows: 'to'
            }
        },
        initialCypher: initialCypher,
        physics: {
            enabled: true,
            barnesHut: {
                gravitationalConstant: -8000,
                springConstant: 0.001,
                springLength: 200
            }
        }
    };

    // Show loading indicator
    showLoading();

    viz = new NeoVis.default(vizConfig);

    // Register event handlers
    viz.registerOnEvent('completed', () => {
        hideLoading();
        updateNodeCount();
    });

    viz.registerOnEvent('error', (error) => {
        console.error('Visualization error:', error);
        hideLoading();
        showError('Failed to load visualization');
    });

    viz.registerOnEvent('clickNode', (event) => {
        const nodeId = event.node;
        const nodeData = viz._nodes[nodeId];
        displayNodeInfo(nodeData);
    });

    viz.render();
}

// Display node information
function displayNodeInfo(nodeData) {
    const infoDiv = document.getElementById('nodeInfo');

    if (!nodeData) {
        infoDiv.innerHTML = 'Click on a node to see details';
        return;
    }

    let html = '';

    // Display key properties
    const properties = [
        { name: 'Type', value: nodeData.labels?.[0] || 'Unknown' },
        { name: 'Reference', value: nodeData.ref || nodeData.key || 'N/A' },
        { name: 'Sheet', value: nodeData.sheet || 'N/A' },
        { name: 'Formula', value: nodeData.formula ? truncateFormula(nodeData.formula) : 'No formula' },
        { name: 'Depth', value: nodeData.depth || '0' },
        { name: 'PageRank', value: nodeData.pagerank ? nodeData.pagerank.toFixed(4) : 'N/A' }
    ];

    // Add range-specific properties
    if (nodeData.labels?.[0] === 'Range') {
        properties.push(
            { name: 'Range Type', value: nodeData.type || 'N/A' },
            { name: 'Size', value: nodeData.size || 'N/A' },
            { name: 'Start Cell', value: nodeData.start_cell || 'N/A' },
            { name: 'End Cell', value: nodeData.end_cell || 'N/A' }
        );
    }

    properties.forEach(prop => {
        html += `
            <div class="property">
                <span class="property-name">${prop.name}:</span>
                <span class="property-value">${prop.value}</span>
            </div>
        `;
    });

    infoDiv.innerHTML = html;
}

// Truncate long formulas for display
function truncateFormula(formula, maxLength = 50) {
    if (formula.length <= maxLength) return formula;
    return formula.substring(0, maxLength) + '...';
}

// Update node count
function updateNodeCount() {
    const nodeCount = Object.keys(viz._nodes || {}).length;
    document.getElementById('nodeCount').textContent = `Nodes: ${nodeCount}`;
}

// Show loading indicator
function showLoading() {
    const vizDiv = document.getElementById('viz');
    const existing = vizDiv.querySelector('.loading');
    if (!existing) {
        const loading = document.createElement('div');
        loading.className = 'loading';
        loading.textContent = 'Loading visualization...';
        vizDiv.appendChild(loading);
    }
}

// Hide loading indicator
function hideLoading() {
    const loading = document.querySelector('.loading');
    if (loading) loading.remove();
}

// Show error message
function showError(message) {
    const vizDiv = document.getElementById('viz');
    const existing = vizDiv.querySelector('.error');
    if (existing) existing.remove();

    const error = document.createElement('div');
    error.className = 'error';
    error.textContent = message;
    vizDiv.appendChild(error);

    setTimeout(() => error.remove(), 5000);
}

// Event handlers
document.getElementById('runQuery').addEventListener('click', () => {
    const query = document.getElementById('querySelect').value;
    if (query) {
        viz.renderWithCypher(query);
        showLoading();
    }
});

document.getElementById('runCustom').addEventListener('click', () => {
    const query = document.getElementById('customQuery').value.trim();
    if (query) {
        viz.renderWithCypher(query);
        showLoading();
    }
});

document.getElementById('clearViz').addEventListener('click', () => {
    viz.clearNetwork();
    document.getElementById('nodeInfo').innerHTML = 'Click on a node to see details';
    updateNodeCount();
});

document.getElementById('stabilize').addEventListener('click', () => {
    viz.stabilize();
});

document.getElementById('centerGraph').addEventListener('click', () => {
    viz._network.fit();
});

// Keyboard shortcuts
document.addEventListener('keydown', (event) => {
    if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
            case 'Enter':
                // Run custom query
                document.getElementById('runCustom').click();
                event.preventDefault();
                break;
            case 'l':
                // Clear visualization
                document.getElementById('clearViz').click();
                event.preventDefault();
                break;
            case 's':
                // Stabilize
                document.getElementById('stabilize').click();
                event.preventDefault();
                break;
            case 'c':
                // Center
                document.getElementById('centerGraph').click();
                event.preventDefault();
                break;
        }
    }
});

// Initialize on page load
window.addEventListener('DOMContentLoaded', init);
