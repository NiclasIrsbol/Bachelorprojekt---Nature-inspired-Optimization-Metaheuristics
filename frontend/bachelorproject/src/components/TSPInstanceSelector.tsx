import { useEffect, useState } from "react";

interface TSPInstance {
  name: string;
  num_cities: number;
}

interface TSPInstanceSelectorProps {
  selectedInstance: string | null;
  onInstanceChange: (instance: string) => void;
}

const API_BASE = "http://localhost:8000";

export default function TSPInstanceSelector({
  selectedInstance,
  onInstanceChange,
}: TSPInstanceSelectorProps) {
  const [instances, setInstances] = useState<TSPInstance[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch(`${API_BASE}/tsp-instances`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json: { instances: TSPInstance[] }) => {
        setInstances(json.instances);
        // Auto-select first instance if none selected
        if (!selectedInstance && json.instances.length > 0) {
          onInstanceChange(json.instances[0].name);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error(err);
        setError("Could not load TSP instances");
        setLoading(false);
      });
  }, []);

  const filteredInstances = instances.filter((instance) =>
    instance.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  if (loading) return <div className="card">Loading instances...</div>;
  if (error) return <div className="card error">{error}</div>;

  return (
    <div className="card controlPanel">
      <div className="tspSelectorContainer">
        <span className="label">TSP Instance</span>
        <input
          type="text"
          className="searchInput"
          placeholder="Search instances..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        <div className="instanceList">
          {filteredInstances.map((instance) => (
            <div
              key={instance.name}
              className={`instanceItem ${
                selectedInstance === instance.name ? "selected" : ""
              }`}
              onClick={() => {
                onInstanceChange(instance.name);
                setSearchTerm("");
              }}
            >
              <div className="instanceName">{instance.name}</div>
              <div className="instanceCities">{instance.num_cities} cities</div>
            </div>
          ))}
          {filteredInstances.length === 0 && (
            <div className="noResults">No instances found</div>
          )}
        </div>
        {selectedInstance && (
          <div className="selectedInstance">
            Selected: <strong>{selectedInstance}</strong>
          </div>
        )}
      </div>
    </div>
  );
}
