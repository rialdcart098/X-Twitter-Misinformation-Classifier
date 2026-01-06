import { useState } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'
import './App.css'
import axios from 'axios'

function App() {
    const [link, setLink] = useState('')
    const classify = async e => {
        e.preventDefault()
        axios.post('http://localhost:8000/classify', { link: link })
    }
    return (
        <div>
            <h1>Twitter Misinformation Classifier</h1>
            <form onSubmit={classify}>
                <input type="text" placeholder="Link" />
                <button type='submit'>Classify</button>
            </form>

        </div>
    )
}

export default App
