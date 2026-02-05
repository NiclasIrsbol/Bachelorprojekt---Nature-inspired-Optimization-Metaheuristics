import { useEffect, useState } from "react";
import './app.css'

function App() {
  const [run, setRun] = useState<any>(null);

  useEffect(() => {
    fetch("/data/latest_run.json")
      .then(res => res.json())
      .then(data => setRun(data));
  }, []);

  if (!run) return <div>Loading…</div>;

  return (
    <div className="screen">
      <div className="left-container">
        <h3>Problem</h3>
        <h3>Algorithms</h3>
        <h3>Metrics</h3>
      </div>
      <div className="right-container">
      <h1>{run.problem} - {run.algorithm}</h1>
      <ul>
        {run.history.map((step: any) => (
          <li key={step.iteration}>
            Iter {step.iteration}: cost {step.cost}
          </li>
        ))}
      </ul>
      </div>
    </div>
  );
}

export default App;