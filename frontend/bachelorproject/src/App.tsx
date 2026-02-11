import { useEffect, useState } from "react"

function App() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch("http://localhost:8000/latest-run")
      .then(r => r.json())
      .then(json => {
        setData(json)
        setLoading(false)
      })
      .catch(err => {
        console.error(err)
        setLoading(false)
      })
  }, [])

  if (loading) return <p>Loading...</p>

  return (
    <div>
      <h1>Latest experiment run</h1>
      <span>{JSON.stringify(data)}</span>
    </div>
  )
}

export default App
