// import { useState } from 'react'
import './App.css'
import axios from 'axios'

function App() {
    const classify = async e => {
        e.preventDefault()
        const link = e.target.message.value
        axios.post('http://localhost:1337/predict', { tweet: link })
            .then(response => {
                alert(`The tweet is classified as: ${response.data.prediction ? 'Legitimate' : 'Misinformation'} with a confidence of ${response.data.confidence}%`)
            })
    }
    return (
        <div>
            <h1>Twitter Misinformation Classifier</h1>
            <form onSubmit={classify}>
                <textarea name="message" rows='10' placeholder="Type here..."></textarea>
                <button type='submit'>Classify</button>
            </form>

        </div>
    )
}

export default App
