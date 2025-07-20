const express = require('express');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files
app.use(express.static('public'));

// Environment variables for Neo4j connection
const neo4jConfig = {
  uri: process.env.NEO4J_URI || 'bolt://localhost:7687',
  user: process.env.NEO4J_USER || 'neo4j',
  password: process.env.NEO4J_PASSWORD || 'neo4j'
};

// API endpoint to get Neo4j config
app.get('/api/config', (req, res) => {
  res.json({
    neo4j: {
      serverUrl: neo4jConfig.uri,
      serverUser: neo4jConfig.user,
      serverPassword: neo4jConfig.password
    }
  });
});

// API endpoint for saved queries
app.get('/api/queries', (req, res) => {
  res.json({
    queries: [
      {
        name: "All Formulas",
        description: "Show all cells with formulas",
        cypher: "MATCH (n:Cell) WHERE n.formula <> '' RETURN n LIMIT 100"
      },
      {
        name: "High Complexity Nodes",
        description: "Show cells with highest PageRank scores",
        cypher: "MATCH (n) WHERE n.pagerank IS NOT NULL RETURN n ORDER BY n.pagerank DESC LIMIT 50"
      },
      {
        name: "Circular References",
        description: "Show circular reference chains",
        cypher: "MATCH p=(n:Cell)-[:DEPENDS_ON*]->(n) RETURN p LIMIT 10"
      },
      {
        name: "Deep Dependencies",
        description: "Show cells with deep dependency chains",
        cypher: "MATCH (n:Cell) WHERE n.depth > 5 RETURN n, n.depth ORDER BY n.depth DESC LIMIT 50"
      },
      {
        name: "Cross-Sheet Dependencies",
        description: "Show dependencies between different sheets",
        cypher: "MATCH (n1:Cell)-[:DEPENDS_ON]->(n2:Cell) WHERE n1.sheet <> n2.sheet RETURN n1, n2 LIMIT 100"
      },
      {
        name: "Range Nodes",
        description: "Show all range nodes and their connections",
        cypher: "MATCH (r:Range)<-[:DEPENDS_ON]-(c:Cell) RETURN r, c LIMIT 50"
      },
      {
        name: "Volatile Formulas",
        description: "Show cells with volatile formulas (NOW, RAND, etc.)",
        cypher: "MATCH (n:Cell) WHERE n.formula =~ '.*(?i)(NOW|TODAY|RAND|RANDBETWEEN|INDIRECT|OFFSET).*' RETURN n LIMIT 50"
      },
      {
        name: "External References",
        description: "Show cells referencing external workbooks",
        cypher: "MATCH (n:Cell) WHERE n.formula CONTAINS '[' AND n.formula CONTAINS ']' RETURN n LIMIT 50"
      }
    ]
  });
});

// Serve the main page
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`Spreadsheet visualization server running on port ${PORT}`);
  console.log(`Neo4j connection: ${neo4jConfig.uri}`);
});
