import { useState } from 'react'
import Lobby from './Lobby.jsx'
import Call from './Call.jsx'

export default function App() {
  const [setup, setSetup] = useState(null)
  return setup ? <Call setup={setup} onEnd={() => setSetup(null)} /> : <Lobby onStart={setSetup} />
}
