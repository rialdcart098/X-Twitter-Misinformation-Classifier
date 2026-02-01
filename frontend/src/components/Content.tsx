import axios from "axios";
import { useState } from "react";

function Content() {
    const [prediction, setPrediction] = useState<string | null>(null);
    const [confidence, setConfidence] = useState<number | null>(null);

    async function classify(e) {
    e.preventDefault()
    const link = e.target.message.value
    axios.post('http://localhost:1337/predict', { tweet: link })
        .then(response => {
            setPrediction(response.data.prediction);
            setConfidence(response.data.confidence);
        })
    }
    return (
        <div className='flex flex-col items-center justify-center gap-8'>
            <h2 className='text-3xl text-green-300 font-almarai font-medium'>Enter the link of the tweet you want to classify:</h2>
            <form onSubmit={classify}>
                <input type='text' name='message' placeholder='Enter tweet link here' className='py-1.5 pr-3 pl-1 p-2 rounded w-96 gap-2 bg-purple-200 focus:outline-none ' />
                <button type='submit' className='cursor-pointer m-4 bg-green-300 p-3 font-semibold text-xl text-purple-950 font-almarai rounded-lg hover:text-purple-800 transition-all ease-in-out'>Classify</button>
            </form>
            {prediction !== null && confidence !== null && (
                <div className='text-center'>
                    <h3 className='text-2xl text-green-300 font-almarai font-medium'>Our model predicted this post as {prediction ? 'legitimate' : 'misinformation'}</h3>
                    <p className='text-green-300 font-almarai font-thin'>Confidence: {confidence}%</p>
                </div>
            )}
        </div>
    )
}
export default Content;